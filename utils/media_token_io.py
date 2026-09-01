"""Media token IO for inference.

Loads pre-tokenized AToken indices/coords for images, videos and 3D assets.
Local absolute paths are read directly from disk; object-storage URIs go through
the optional petrel SDK (configure via PETREL_CONF_PATH; see README).
"""

import io
import json
import os
import re

import torch

# ============================================================
# Monkey-patch petrel S3Client 以注入自定义 botocore Config
# 解决 Ceph 高并发下 read_timeout=60s 导致的超时问题
# Lazy-import petrel so local-abs-path workloads can start without
# loading the native petrel SDK at module import time.
# ============================================================
_PETREL_CONNECT_TIMEOUT = float(os.environ.get("PETREL_CONNECT_TIMEOUT", "10"))
_PETREL_READ_TIMEOUT = float(os.environ.get("PETREL_READ_TIMEOUT", "10"))
_PETREL_MAX_ATTEMPTS = int(os.environ.get("PETREL_MAX_ATTEMPTS", "2"))
_PETREL_MAX_POOL_CONNECTIONS = int(os.environ.get("PETREL_MAX_POOL_CONNECTIONS", "50"))
_PETREL_PATCHED = False


def _patch_petrel_s3_client():
    """Optional boto S3Client timeout patch (old petrel builds only).

    Current conda env ships s3cpp-only petrel_client (no petrel_client.ceph.s3);
    patch is a no-op there. Client() still works via s3cpp.
    """
    global _PETREL_PATCHED
    if _PETREL_PATCHED:
        return
    _PETREL_PATCHED = True
    try:
        from petrel_client.ceph.s3 import s3_client as s3_module
    except ModuleNotFoundError:
        # s3cpp backend — nothing to monkey-patch.
        return
    try:
        from botocore.config import Config as BotocoreConfig
        import boto3
        from botocore import UNSIGNED
        from petrel_client.ceph.ceph import Ceph

        def _patched_init(self, cluster, conf, anonymous_access, *args, **kwargs):
            custom_config = BotocoreConfig(
                connect_timeout=_PETREL_CONNECT_TIMEOUT,
                read_timeout=_PETREL_READ_TIMEOUT,
                retries={"max_attempts": _PETREL_MAX_ATTEMPTS},
                max_pool_connections=_PETREL_MAX_POOL_CONNECTIONS,
            )

            if anonymous_access:
                base_config = BotocoreConfig(signature_version=UNSIGNED)
                merged_config = base_config.merge(custom_config)
                s3_args = {"config": merged_config}
            else:
                s3_args = {
                    "aws_access_key_id": conf["access_key"],
                    "aws_secret_access_key": conf["secret_key"],
                    "config": custom_config,
                }

            s3_args["endpoint_url"] = conf["endpoint_url"]
            s3_args["verify"] = conf.get_boolean("verify_ssl", False)

            Ceph.__init__(self, cluster, conf, *args, **kwargs)

            self._cluster = cluster
            self._conf = conf
            self._session = boto3.session.Session()
            self._s3_resource = self._session.resource("s3", **s3_args)

        s3_module.S3Client.__init__ = _patched_init
        print(
            f"[PETREL_PATCH] S3Client patched: connect_timeout={_PETREL_CONNECT_TIMEOUT}s, "
            f"read_timeout={_PETREL_READ_TIMEOUT}s, max_attempts={_PETREL_MAX_ATTEMPTS}, "
            f"max_pool_connections={_PETREL_MAX_POOL_CONNECTIONS}",
            flush=True,
        )
    except Exception as e:
        print(f"[PETREL_PATCH] Failed to patch S3Client: {e}", flush=True)


# ============================================================

# 已有的正则不变
_REMOTE_PREFIX = re.compile(r"^[a-z0-9._-]+:", re.IGNORECASE)  # e.g. my-remote:

# rclone remote name → petrel cluster name 映射。
# 当 token_path 以 rclone remote 前缀开头、且该 bucket 不在 petrel 默认集群时，
# 需要在此注册映射，使 petrel 能路由到正确的集群。
# 通过环境变量 RCLONE_TO_PETREL_CLUSTER_JSON 注入，例如:
#   export RCLONE_TO_PETREL_CLUSTER_JSON='{"my-remote": "my-cluster"}'
# 本地路径数据不需要配置。
_RCLONE_TO_PETREL_CLUSTER = json.loads(os.environ.get("RCLONE_TO_PETREL_CLUSTER_JSON", "{}"))

_WARNED_REMOTES = set()


def _parse_bucket_key(remote_path: str):
    """
    支持:
      - my-remote:s3://image_tokens/path/to/...
      - s3://image_tokens/path/to/...
      - image_tokens/path/to/...
    返回 (rclone_remote_or_empty, bucket, key)。
    """
    p = remote_path.strip()
    rclone_remote = ""
    if _REMOTE_PREFIX.match(p):           # 去掉 rclone remote 前缀
        rclone_remote, p = p.split(":", 1)
    if p.startswith("s3://"):             # 去掉 s3://
        p = p[5:]
    p = p.lstrip("/")                     # 防止 key 以 / 开头
    parts = p.split("/", 1)
    bucket = parts[0] if parts else ""
    key = parts[1] if len(parts) == 2 else ""
    return rclone_remote, bucket, key


def _to_petrel_path(remote_path: str) -> str:
    """
    在保持 _parse_bucket_key 行为的基础上，把路径转成
    petrel Client 可用的标准 s3://bucket/key 形式。
    若 rclone remote 在 _RCLONE_TO_PETREL_CLUSTER 中有映射，
    则加上 petrel cluster 前缀（如 cluster2:s3://...）。
    """
    rclone_remote, bucket, key = _parse_bucket_key(remote_path)
    if not bucket or not key:
        raise ValueError(
            f"Invalid token_path for S3: {remote_path!r} -> bucket={bucket!r}, key={key!r}"
        )
    petrel_cluster = _RCLONE_TO_PETREL_CLUSTER.get(rclone_remote, "")
    if petrel_cluster:
        return f"{petrel_cluster}:s3://{bucket}/{key}"
    if rclone_remote and rclone_remote not in _WARNED_REMOTES:
        _WARNED_REMOTES.add(rclone_remote)
        print(
            f"[PETREL] rclone remote {rclone_remote!r} has no cluster mapping; falling back to "
            f"the default cluster. Add it to _RCLONE_TO_PETREL_CLUSTER if reads 404.",
            flush=True,
        )
    return f"s3://{bucket}/{key}"


def _get_petrel_conf_path() -> str:
    """从环境变量读取 petrel 配置文件路径，未设置时返回空字符串表示使用默认路径。"""
    return (
        os.environ.get("PETREL_CONF_PATH")
        or os.environ.get("PETRELOSS_CONF")
        or ""
    ).strip()


_PETREL_CLIENTS = {}


def _get_petrel_client(_conf_path: str = ""):
    """
    Per-process petrel Client (fork-safe).

    DataLoader workers fork after import; a parent-created Client / @lru_cache
    handle is not safe to use in children and can segfault.
    """
    # petrel bundles its own aws-sdk-cpp; pyarrow ships another copy. Whichever is
    # dlopen'ed first wins for the shared aws-c-* symbols, and petrel only survives
    # the "arrow first" order. Force that order before touching petrel_client.
    try:
        import pyarrow  # noqa: F401
    except ImportError:
        pass

    from petrel_client.client import Client

    pid = os.getpid()
    key = (pid, _conf_path)
    client = _PETREL_CLIENTS.get(key)
    if client is None:
        _patch_petrel_s3_client()
        client = Client(_conf_path) if _conf_path else Client()
        _PETREL_CLIENTS[key] = client
    return client


def load_pkl_from_rclone(token_path: str):
    """
    读取 token pkl：

    支持路径形式：
      - my-remote:s3://image_tokens/path/to/.../xxx.webp.pkl
      - s3://image_tokens/path/to/.../xxx.webp.pkl
      - image_tokens/path/to/.../xxx.webp.pkl
      - 以及本地绝对路径：/mnt/xxx/xxx.pkl

    返回值：torch.load 反序列化后的对象（通常是一个 dict，含 indices / coords）
    """

    # 1) 本地路径（绝对或相对当前工作目录）：直接走磁盘。
    # `token_path` is commonly passed as a relative .pkl/.pt path from the
    # repository's launch directory; do not mistake it for an object-store key.
    if os.path.isfile(token_path):
        # Token files are trusted data artifacts and may contain NumPy arrays;
        # PyTorch 2.6's weights-only default rejects those legacy payloads.
        return torch.load(token_path, map_location="cpu", weights_only=False)
    if os.path.isabs(token_path):
        raise FileNotFoundError(f"{token_path} not found.")

    # 2) 远程：通过 petrel sdk 从对象存储读取
    petrel_path = _to_petrel_path(token_path)  # 会做 bucket/key 校验
    client = _get_petrel_client(_get_petrel_conf_path())

    last_err = None
    try:
        # petrel Client.get 返回 bytes
        data = client.get(petrel_path)
        if not data:
            raise FileNotFoundError(
                f"{token_path} empty in object storage: {petrel_path}"
            )

        # 和原逻辑一样，用 BytesIO 包一层再 torch.load
        bio = io.BytesIO(data)
        # Token files are trusted data artifacts and may contain NumPy arrays;
        # PyTorch 2.6's weights-only default rejects those legacy payloads.
        obj = torch.load(bio, map_location="cpu", weights_only=False)
        return obj

    except Exception as e:
        last_err = e

    # 读取失败时，沿用之前“抛出最后一次错误”的风格
    raise last_err if last_err is not None else RuntimeError(
        f"failed to load token from object storage (petrel): {token_path}"
    )


def _validate_media_token_schema(data_pkl, *, media_kind: str):
    """Validate and normalize a serialized media-token payload."""
    coord_columns = {"image": {4, 5}, "3d": {3, 5}, "video": {5}}
    if media_kind not in coord_columns:
        raise ValueError(f"Unsupported media kind: {media_kind!r}")
    if not isinstance(data_pkl, dict):
        raise ValueError(f"{media_kind} token payload must be a dict")

    missing = {"indices", "coords"} - data_pkl.keys()
    if missing:
        raise ValueError(
            f"{media_kind} token payload is missing required key(s): {sorted(missing)}"
        )

    def _integer_matrix(value, *, name, allowed_columns):
        try:
            matrix = torch.as_tensor(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{media_kind} {name} must be a rectangular integer matrix"
            ) from exc
        if matrix.ndim != 2:
            raise ValueError(
                f"{media_kind} {name} must be 2D, got shape {tuple(matrix.shape)}"
            )
        if matrix.shape[0] == 0:
            raise ValueError(f"{media_kind} {name} must contain at least one row")
        if matrix.shape[1] not in allowed_columns:
            expected = "/".join(str(columns) for columns in sorted(allowed_columns))
            raise ValueError(
                f"{media_kind} {name} must have {expected} columns, "
                f"got shape {tuple(matrix.shape)}"
            )
        if (
            matrix.dtype == torch.bool
            or torch.is_floating_point(matrix)
            or torch.is_complex(matrix)
        ):
            raise ValueError(f"{media_kind} {name} must use an integer dtype")
        return matrix.to(dtype=torch.long, device="cpu")

    indices = _integer_matrix(data_pkl["indices"], name="indices", allowed_columns={8})
    coords = _integer_matrix(
        data_pkl["coords"],
        name="coords",
        allowed_columns=coord_columns[media_kind],
    )
    if indices.shape[0] != coords.shape[0]:
        raise ValueError(
            f"{media_kind} indices and coords row counts must match, got "
            f"{indices.shape[0]} and {coords.shape[0]}"
        )

    min_token_id = int(indices.min().item())
    max_token_id = int(indices.max().item())
    if min_token_id < 0 or max_token_id >= 4096:
        raise ValueError(
            f"{media_kind} token ids must be in [0, 4096), got "
            f"min={min_token_id}, max={max_token_id}"
        )
    min_coord = int(coords.min().item())
    if min_coord < 0:
        raise ValueError(
            f"{media_kind} coords must be non-negative, got min={min_coord}"
        )

    return indices, coords


def load_image_tokens_from_rclone(token_path):
    data_pkl = load_pkl_from_rclone(token_path)
    image_tokens, image_coords = _validate_media_token_schema(
        data_pkl, media_kind="image"
    )
    max_height = image_coords[:, 2].max().item() + 1  # height: 第三列（索引2）
    max_width = image_coords[:, 3].max().item() + 1   # width: 第四列（索引3）
    return image_tokens, max_height, max_width


def load_3d_tokens_from_rclone(token_path):
    """
    加载 atoken 3D token pkl/pt：indices + coords。

    支持两种 coords 格式：
      - 5 列 [batch, t, x, y, z]：旧格式，col2=x, col3=y, col4=z
      - 3 列 [x, y, z]：新格式，会自动补齐 batch=0, t=0 成 5 列

    与 2D 命名对应：x→width(W), y→height(H), z→depth(D)。
    """
    data_pkl = load_pkl_from_rclone(token_path)
    image_tokens, image_coords = _validate_media_token_schema(
        data_pkl, media_kind="3d"
    )

    num_cols = image_coords.shape[1]
    if num_cols == 3:
        # 新格式：[x, y, z] → 补齐为 [batch=0, t=0, x, y, z]
        n = image_coords.shape[0]
        batch_col = torch.zeros((n, 1), dtype=image_coords.dtype, device=image_coords.device)
        t_col = torch.zeros((n, 1), dtype=image_coords.dtype, device=image_coords.device)
        image_coords = torch.cat([batch_col, t_col, image_coords], dim=1)
        # 取 x, y, z
        max_width = image_coords[:, 2].max().item() + 1   # x
        max_height = image_coords[:, 3].max().item() + 1  # y
        max_depth = image_coords[:, 4].max().item() + 1   # z
    elif num_cols == 5:
        # 旧格式：[batch, t, x, y, z]
        max_width = image_coords[:, 2].max().item() + 1   # x
        max_height = image_coords[:, 3].max().item() + 1  # y
        max_depth = image_coords[:, 4].max().item() + 1   # z
    else:
        raise ValueError(f"Unexpected coords shape: {image_coords.shape}, expected (N, 3) or (N, 5)")

    return image_tokens, image_coords, max_height, max_width, max_depth


def load_video_tokens_from_rclone(token_path):
    """
    加载 atoken 视频 token pkl：indices + coords。

    coords 格式为 [N, 5]：[batch, t, x, y, z]，其中 t=时序帧组, x=空间宽, y=空间高, z=0。
    返回 (indices, coords, T, H, W)：T=帧组数, H=空间高, W=空间宽。
    """
    data_pkl = load_pkl_from_rclone(token_path)
    video_tokens, video_coords = _validate_media_token_schema(
        data_pkl, media_kind="video"
    )

    # coords: [batch, t, x, y, z]
    max_T = video_coords[:, 1].max().item() + 1   # t
    max_W = video_coords[:, 2].max().item() + 1   # x
    max_H = video_coords[:, 3].max().item() + 1   # y
    return video_tokens, video_coords, max_T, max_H, max_W

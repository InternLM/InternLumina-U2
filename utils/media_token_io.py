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
# Monkey-patch petrel's S3Client to inject a custom botocore Config, which
# avoids the read_timeout=60s stalls seen under high object-store concurrency.
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

_REMOTE_PREFIX = re.compile(r"^[a-z0-9._-]+:", re.IGNORECASE)  # e.g. my-remote:

# Maps an rclone remote name to a petrel cluster name. Register a mapping when a
# token_path starts with an rclone remote prefix and that bucket does not live in
# petrel's default cluster, so petrel can route to the right one. Injected through
# the RCLONE_TO_PETREL_CLUSTER_JSON environment variable, e.g.
#   export RCLONE_TO_PETREL_CLUSTER_JSON='{"my-remote": "my-cluster"}'
# Local paths need no configuration.
_RCLONE_TO_PETREL_CLUSTER = json.loads(os.environ.get("RCLONE_TO_PETREL_CLUSTER_JSON", "{}"))

_WARNED_REMOTES = set()


def _parse_bucket_key(remote_path: str):
    """
    Accepts:
      - my-remote:s3://image_tokens/path/to/...
      - s3://image_tokens/path/to/...
      - image_tokens/path/to/...
    Returns (rclone_remote_or_empty, bucket, key).
    """
    p = remote_path.strip()
    rclone_remote = ""
    if _REMOTE_PREFIX.match(p):           # strip the rclone remote prefix
        rclone_remote, p = p.split(":", 1)
    if p.startswith("s3://"):             # strip s3://
        p = p[5:]
    p = p.lstrip("/")                     # keys must not start with /
    parts = p.split("/", 1)
    bucket = parts[0] if parts else ""
    key = parts[1] if len(parts) == 2 else ""
    return rclone_remote, bucket, key


def _to_petrel_path(remote_path: str) -> str:
    """
    Convert a path into the standard s3://bucket/key form a petrel Client accepts,
    keeping _parse_bucket_key's behaviour. If the rclone remote has a mapping in
    _RCLONE_TO_PETREL_CLUSTER, the petrel cluster prefix is prepended
    (e.g. my-cluster:s3://...).
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
    """Petrel config path from the environment; empty string means use the default."""
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
    Read a token pkl.

    Accepted path forms:
      - my-remote:s3://image_tokens/path/to/.../xxx.webp.pkl
      - s3://image_tokens/path/to/.../xxx.webp.pkl
      - image_tokens/path/to/.../xxx.webp.pkl
      - local absolute paths, e.g. /data/tokens/xxx.pkl

    Returns whatever torch.load deserializes (usually a dict with indices / coords).
    """

    # 1) Local path (absolute or relative to the cwd): read straight from disk.
    # `token_path` is commonly passed as a relative .pkl/.pt path from the
    # repository's launch directory; do not mistake it for an object-store key.
    if os.path.isfile(token_path):
        # Token files are trusted data artifacts and may contain NumPy arrays;
        # PyTorch 2.6's weights-only default rejects those legacy payloads.
        return torch.load(token_path, map_location="cpu", weights_only=False)
    if os.path.isabs(token_path):
        raise FileNotFoundError(f"{token_path} not found.")

    # 2) Remote: read from object storage through the petrel SDK.
    petrel_path = _to_petrel_path(token_path)  # validates bucket/key
    client = _get_petrel_client(_get_petrel_conf_path())

    last_err = None
    try:
        # petrel Client.get returns bytes
        data = client.get(petrel_path)
        if not data:
            raise FileNotFoundError(
                f"{token_path} empty in object storage: {petrel_path}"
            )

        # Wrap in BytesIO before torch.load, as before
        bio = io.BytesIO(data)
        # Token files are trusted data artifacts and may contain NumPy arrays;
        # PyTorch 2.6's weights-only default rejects those legacy payloads.
        obj = torch.load(bio, map_location="cpu", weights_only=False)
        return obj

    except Exception as e:
        last_err = e

    # On failure, raise the last error seen
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
    max_height = image_coords[:, 2].max().item() + 1  # height: column index 2
    max_width = image_coords[:, 3].max().item() + 1   # width: column index 3
    return image_tokens, max_height, max_width


def load_3d_tokens_from_rclone(token_path):
    """
    Load an AToken 3D token pkl/pt: indices + coords.

    Two coords layouts are accepted:
      - 5 columns [batch, t, x, y, z]: col2=x, col3=y, col4=z
      - 3 columns [x, y, z]: padded to 5 columns with batch=0, t=0

    Matching the 2D naming: x -> width (W), y -> height (H), z -> depth (D).
    """
    data_pkl = load_pkl_from_rclone(token_path)
    image_tokens, image_coords = _validate_media_token_schema(
        data_pkl, media_kind="3d"
    )

    num_cols = image_coords.shape[1]
    if num_cols == 3:
        # 3-column form: [x, y, z] -> [batch=0, t=0, x, y, z]
        n = image_coords.shape[0]
        batch_col = torch.zeros((n, 1), dtype=image_coords.dtype, device=image_coords.device)
        t_col = torch.zeros((n, 1), dtype=image_coords.dtype, device=image_coords.device)
        image_coords = torch.cat([batch_col, t_col, image_coords], dim=1)
        # Take x, y, z
        max_width = image_coords[:, 2].max().item() + 1   # x
        max_height = image_coords[:, 3].max().item() + 1  # y
        max_depth = image_coords[:, 4].max().item() + 1   # z
    elif num_cols == 5:
        # 5-column form: [batch, t, x, y, z]
        max_width = image_coords[:, 2].max().item() + 1   # x
        max_height = image_coords[:, 3].max().item() + 1  # y
        max_depth = image_coords[:, 4].max().item() + 1   # z
    else:
        raise ValueError(f"Unexpected coords shape: {image_coords.shape}, expected (N, 3) or (N, 5)")

    return image_tokens, image_coords, max_height, max_width, max_depth


def load_video_tokens_from_rclone(token_path):
    """
    Load an AToken video token pkl: indices + coords.

    coords is [N, 5]: [batch, t, x, y, z], where t is the temporal frame group,
    x is spatial width, y is spatial height and z is 0.
    Returns (indices, coords, T, H, W): T frame groups, H height, W width.
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

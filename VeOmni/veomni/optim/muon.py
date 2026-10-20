"""Per-Head Muon, the optimizer described in the Kimi K3 technical report
(arXiv 2607.24653).

Basis (quoted from the report)
------------------------------
§2.5 Per-Head Muon
    "Following Kimi K2, Kimi K3 adopts Muon as the optimizer for its matrix parameters.
     For attention projections, we further refine it into a per-head variant: instead of
     applying Newton-Schulz orthogonalization to the full Q, K, and V projection matrices,
     we partition their momentum matrices along the head dimension and orthogonalize each
     head's block separately."
    The motivation is stated there too: orthogonalizing the whole block treats every
    head as one coupled unit, so heads with large momentum dominate the shared update
    direction while small ones are under-normalized. Splitting per head equalizes the
    update scale across heads, which is more stable at scale, and the NS iteration is
    cheaper on the tall, narrow per-head blocks.

§3.3 Training Recipe
    "We optimize the model using the Per-Head Muon optimizer (§2.5) together with the
     weight-clipping mechanism introduced in Kimi K2 ... cosine learning rate schedule with
     a 1% linear warmup. Weight decay is set to 0.1 throughout."

What the report does *not* restate (NS coefficients, the RMS matching factor, momentum)
follows the public Muon / Moonlight / K2 line: NS5 coefficients
(3.4445, -4.7750, 2.0315) for 5 iterations; the update is scaled by
0.2 * sqrt(max(A, B)) so its RMS lands at 0.2, matching AdamW's typical update RMS --
which is why an AdamW learning rate can be reused unchanged; Nesterov momentum 0.95.
All of these are constructor arguments, so changing them needs no code edit.

§5.2.2 notes that K3 uses P2P rather than all-gather to assemble full matrices. The
portable default here is DTensor redistribute (equivalent to an on-demand all-gather),
with an owner-based all-to-all fast path available. That fast path is experimental on
the target Torch 2.6 / NCCL and is off by default; enable it only after single-node and
multi-node GPU smoke runs.

Relation to the public implementations
--------------------------------------
The K3 repository (github.com/MoonshotAI/Kimi-K3) publishes weights and the report but
no optimizer code, so the line-by-line references are:

  [J] github.com/KellerJordan/Muon/blob/master/muon.py            (original Muon author)
  [M] github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py (Moonshot, same
      line as K2/K3)

The core update follows [M]'s accumulating-momentum form:

    buf.mul_(momentum).add_(g)        # accumulating momentum, not [J]'s lerp_ EMA
    g = g.add(buf, alpha=momentum)    # Nesterov
    u = NS5(g, steps=5)
    p.mul_(1 - lr * wd)               # weight decay uses the BASE lr
    p.add_(u, alpha=-lr * 0.2 * sqrt(max(A, B)))   # the update uses the scaled lr

([J]'s lerp_ variant differs by a constant factor (1 - beta); NS normalizes the scale
away, so the direction is identical.)

Four deliberate departures from the public implementations, each with a reason:

1. **Tensors with more than 2 dims are batched, not flattened.** [M] applies
   `g.view(g.size(0), -1)` for ndim > 2 (the conv-filter convention). This model's MoE
   weights are fused as (num_experts, in, out); flattening would orthogonalize all 256
   experts as a single matrix, erasing the relative scale between experts, which is
   semantically wrong. This follows [J]'s newer batched version instead: the last two
   dims are the matrix, everything before them is batch, so each expert is
   orthogonalized on its own.

2. **The Frobenius norm is computed in fp32.** [J] and [M] both cast to bfloat16 before
   `X.norm()`. With 8 mantissa bits, summing squares over a 2048x2048 matrix carries
   real error, and once that step is off, all five iterations run at the wrong scale.
   The cast to bf16 happens after normalization; the iteration itself stays bf16.

3. **The AdamW branch matches torch.optim.AdamW, not [M].** [M] places eps inside
   `buf1/(eps + buf2.sqrt())` and rescales, which is equivalent to
   eps/sqrt(bias_correction2); torch computes `m/bc1 / (sqrt(v/bc2) + eps)`. This code
   base switches over from torch AdamW, so the update rule for non-Muon parameters
   (embeddings, lm_head, norms) must not change at all -- otherwise a change in results
   cannot be attributed to Muon rather than to AdamW drift.

4. **Per-head RMS scaling uses the head block's edge lengths, not the full matrix's.**
   The report omits this detail. Either choice equalizes scale across heads (the blocks
   are the same shape); they differ in the update magnitude of the whole tensor. Using
   the block edges keeps RMS(update) == 0.2 for every parameter, so an AdamW learning
   rate transfers unchanged; using the full-matrix edges would make qkv updates
   sqrt(3072/2048) = 1.22x larger than elsewhere. The former is used.

On ns_steps = 5
    The fixed point of these coefficients puts singular values in [0.682, 1.134], but
    whether 5 steps suffice depends on the aspect ratio:
      (128, 2048) per-head blocks / (512, 2048) experts / (5120, 2048) -> converged in 5
      (2048, 2048) square matrices such as attention.dense still leave ~3% of singular
      values below 0.5 after 5 steps on Gaussian input; 8 steps reach 0.1%, 12 steps
      converge fully.
    Square Gaussian input is the worst-conditioned case, though. With a decaying
    spectrum closer to real gradients (sv proportional to 1/sqrt(k)), 5 steps converge
    completely. The default therefore stays at the published 5; models dominated by
    square matrices may want 8.

Which parameters use Muon
-------------------------
K2's rule is that hidden matrix parameters use Muon and the rest (embeddings, LM head,
biases) use AdamW. `classify_muon_params` decides by **module type and tensor rank**
rather than by name substring: in this model `img_embeddings` (nn.Embedding, AdamW) and
`img_embeddings_proj` (Linear, Muon) differ by a suffix only, and substring matching
would misclassify the latter into AdamW.
"""

import math
import warnings
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.optim.optimizer import Optimizer


try:
    from torch.distributed.tensor import DTensor, Replicate, Shard
except ImportError:  # torch < 2.5
    from torch.distributed._tensor import DTensor, Replicate, Shard


__all__ = ["Muon", "classify_muon_params", "zeropower_via_newtonschulz"]


# --------------------------------------------------------------------- NS iteration
@torch.no_grad()
def zeropower_via_newtonschulz(
    G: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-7,
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Orthogonalize the last two dims of `G` (batched); returns G's shape.

    Five Newton-Schulz iterations with coefficients (a, b, c) =
    (3.4445, -4.7750, 2.0315). These are deliberately tuned to converge quickly near
    zero at the cost of pulling singular values only into ~[0.7, 1.3] rather than
    exactly 1 -- Muon does not need exact orthogonality, only a flattened singular
    value spectrum, so 5 steps in bf16 suffice.

    Every dim before the last two is treated as batch: MoE's (num_experts, in, out)
    and the per-head split's (num_heads, head_dim, in) take the same path.
    """
    if G.ndim < 2:
        raise ValueError(f"Newton-Schulz needs at least 2 dims, got {G.ndim}")
    a, b, c = 3.4445, -4.7750, 2.0315

    # Compute the norm in fp32: bf16 has 8 mantissa bits, so summing squares over a
    # large matrix carries real error, and once this step is off all five iterations
    # run at the wrong scale.
    # pow/sum rather than torch.linalg.matrix_norm: the latter is not guaranteed to
    # have a backend implementation on every accelerator, while pow/sum/sqrt always
    # do. The two are numerically equivalent (both Frobenius).
    G32 = G if G.dtype in (torch.float32, torch.float64) else G.float()
    nrm = G32.pow(2).sum(dim=(-2, -1), keepdim=True).sqrt().clamp_min(eps)

    X = (G32 / nrm).to(dtype)

    # The NS iteration wants rows <= cols; otherwise X @ X^T is the large side,
    # which is both slower and less stable.
    transposed = X.size(-2) > X.size(-1)
    if transposed:
        X = X.mT

    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X

    if transposed:
        X = X.mT
    return X


def _split_heads(u: torch.Tensor, head_rows: int, head_axis: int) -> Tuple[torch.Tensor, Any]:
    """Split one of the last two dims into per-head blocks.

    Returns (split tensor, info needed to undo the split).

    head_axis=-2 splits by *row* (Q/K/V, whose out_features are heads concatenated).
    head_axis=-1 splits by *column* (o_proj, whose in_features are heads concatenated).
    """
    if head_rows <= 0:
        return u, None
    if head_axis not in (-2, -1):
        raise ValueError(f"head_axis must be -2 or -1, got {head_axis}")

    orig_shape = u.shape
    transposed = head_axis == -1
    if transposed:
        u = u.mT  # splitting columns == transposing then splitting rows; NS is
        # equivariant under transposition
    rows = u.size(-2)
    if rows % head_rows:
        raise ValueError(f"per-head split needs {rows} divisible by head_rows={head_rows}")
    u = u.reshape(*u.shape[:-2], rows // head_rows, head_rows, u.size(-1))
    return u, (orig_shape, transposed)


def _merge_heads(o: torch.Tensor, info: Any) -> torch.Tensor:
    if info is None:
        return o
    orig_shape, transposed = info
    o = o.reshape(*o.shape[:-3], o.size(-3) * o.size(-2), o.size(-1))
    if transposed:
        o = o.mT
    return o.reshape(orig_shape)


@torch.no_grad()
def muon_update(
    u: torch.Tensor,
    head_rows: int = 0,
    head_axis: int = -2,
    ns_steps: int = 5,
    rms_scale: float = 0.2,
    ns_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Muon's update from the (Nesterov-mixed) momentum, excluding lr and weight decay.

    RMS matching: a semi-orthogonal O (m x n, m <= n) satisfies O @ O^T = I, so its
    sum of squares is m and RMS(O) = sqrt(m / (m * n)) = 1 / sqrt(max(m, n)). After
    multiplying by rms_scale * sqrt(max(m, n)) the RMS is exactly rms_scale (0.2 by
    default, the typical RMS of an AdamW update), which is why a learning rate tuned
    for AdamW transfers directly.

    In per-head mode max(m, n) uses each *head block's* edge lengths, not the full
    matrix's. That is where K3's "per-head orthogonalization equalizes the update
    scale across heads" becomes concrete: smaller blocks give a smaller scale factor,
    so every head receives the same update magnitude.
    """
    # Only 2-D (Linear, optionally split per head) or 3-D (fused MoE expert stack)
    # parameters belong here. For 4-D and above NS would treat the leading dims as
    # batch, which is semantically wrong. classify_muon_params already routes those to
    # AdamW; this is a second line of defence against a group config that bypasses the
    # classifier.
    if u.ndim not in (2, 3):
        raise ValueError(
            f"muon_update supports 2-D/3-D parameters only, got {u.ndim}-D "
            f"(shape={tuple(u.shape)}). 4-D and above (e.g. conv) must use AdamW."
        )
    u_split, info = _split_heads(u, head_rows, head_axis)
    o = zeropower_via_newtonschulz(u_split, steps=ns_steps, dtype=ns_dtype)
    m, n = o.size(-2), o.size(-1)
    o = o * (rms_scale * math.sqrt(max(m, n)))
    return _merge_heads(o, info)


# --------------------------------------------------------------------- distributed
def _loc(t: torch.Tensor) -> torch.Tensor:
    """Local shard of a DTensor (shared storage, so in-place writes land back);
    plain tensors are returned unchanged."""
    return t.to_local() if isinstance(t, DTensor) else t


def _matrix_full_local(x: torch.Tensor, gather_dtype: Optional[torch.dtype] = None):
    """Materialize a DTensor's *last two dims*.

    Returns (local tensor, the placements used to materialize it, or None).

    This is the only communication point in distributed Muon, and the rule is simple:

      a Shard(d) on some mesh dim splits a *matrix* dim when d falls in the last two
      dims, so the NS iteration would not see a complete matrix and it must be
      gathered first. A Shard on an earlier dim splits "which matrix" instead (the
      expert dim of a fused MoE stack is Shard(0)); each matrix is already complete
      locally and nothing is communicated.

    That distinction saves a great deal here. Most Muon-eligible parameters are MoE
    experts shaped (num_experts, in, out); with expert parallelism off, FSDP2 shards
    dim 0, which is the expert dim, so no bytes move. Only the 2-D tensors need an
    all-gather, and they are a small fraction of the model. `gather_dtype=torch.bfloat16`
    halves that traffic by casting before the gather; the Frobenius accumulation that
    follows still runs in fp32.

    K3 §5.2.2 replaces all-gather with P2P because its sharding is ZeRO-style over
    whole parameters, so the traffic scales with the full parameter count. Here the
    expert dim already lines up, so the simple and reliable redistribute stays the
    default and the owner-based all-to-all is an explicit, opt-in fast path for 2-D
    tensors on a single mesh dim.
    """
    if not isinstance(x, DTensor):
        return x, None

    cur = list(x.placements)
    tgt = []
    for p in cur:
        # Shard on a batch (expert) dim: leave it alone, no communication.
        if isinstance(p, Shard) and p.dim < x.ndim - 2:
            tgt.append(p)
        else:
            # Shard on a matrix dim, or Partial: both must be made complete first.
            tgt.append(Replicate())

    if tgt == cur:
        # Do not apply gather_dtype on the no-communication branch: this is the MoE
        # expert path, and casting for a gather that never happens only wastes time
        # and an equally large temporary buffer.
        return x.to_local(), None

    if gather_dtype is not None and x.dtype != gather_dtype:
        x = x.to(gather_dtype)  # cast before the all-gather, or traffic is unchanged
    return x.redistribute(placements=tgt).to_local(), tgt


def _scatter_back(full_local: torch.Tensor, ref: torch.Tensor, tgt) -> torch.Tensor:
    """Slice a materialized result back into `ref`'s sharding; returns the local tensor.

    Replicate -> Shard is a purely local slice in DTensor and communicates nothing.
    """
    if tgt is None:
        return full_local
    d = DTensor.from_local(full_local, ref.device_mesh, tgt, run_check=False)
    return d.redistribute(placements=list(ref.placements)).to_local()


# ------------------------------------------------------- all-to-all fast path
# The redistribute path above materializes every 2-D matrix on *every* rank and runs
# NS there, so the communication is a full all-gather and the NS work is repeated per
# rank. Both grow linearly with parameter count and rank count.
#
# The fast path below handles the FSDP2/HSDP layout where a 2-D parameter is Shard(0)
# on exactly one mesh dim and Replicate elsewhere, removing that redundancy with an
# owner-based all-to-all: matrices are grouped shard-world at a time, the j-th matrix
# in a group is owned by rank j, one all-to-all assembles its full [M, K] from the row
# shards, the owner runs NS once, and a second all-to-all returns the row shards. The
# matrix is the same, so this is numerically equivalent to the redistribute path, but
# NS is no longer repeated per rank and the traffic becomes a balanced all-to-all.
# Other layouts (3-D MoE expert stacks, several shard dims, column shards, Partial)
# still fall back to redistribute.


def _fsdp_all2all_fast_path_mesh_dim(u: torch.Tensor) -> Optional[int]:
    """The single 2-D ``Shard(0)`` mesh dim of `u`, or ``None`` if the fast path
    does not apply.

    Every other HSDP mesh dim must be Replicate; 3-D tensors, several shard dims,
    column shards, Partial and world <= 1 all fall back to redistribute. A trailing
    rank receiving 0 rows (fewer rows than the shard world) is allowed.
    """
    if not isinstance(u, DTensor) or u.ndim != 2:
        return None
    mesh = u.device_mesh
    shard_mesh_dims: List[int] = []
    for mesh_dim, placement in enumerate(u.placements):
        if isinstance(placement, Shard) and placement.dim == 0:
            shard_mesh_dims.append(mesh_dim)
        elif not isinstance(placement, Replicate):
            return None
    if len(shard_mesh_dims) != 1:
        return None
    mesh_dim = shard_mesh_dims[0]
    return mesh_dim if mesh.size(mesh_dim) > 1 else None


def _shard_row_sizes(full_rows: int, world: int) -> List[int]:
    """Rows per rank for a DTensor `Shard(0)` (ceil split; trailing ranks may get 0).

    This has to match DTensor's own split rank by rank: the owner relies on it to
    reassemble the row shards in rank order, and the reverse direction relies on it to
    slice the result back. A mismatch produces a wrongly ordered matrix, not an error.
    """
    per = -(-full_rows // world)  # ceil(full_rows / world)
    sizes, remaining = [], full_rows
    for _ in range(world):
        take = min(per, max(remaining, 0))
        sizes.append(take)
        remaining -= take
    return sizes


def _all_to_all_flat(
    send_chunks: List[torch.Tensor],
    recv_numels: List[int],
    pg: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> List[torch.Tensor]:
    """One `all_to_all_single`: exchange world flat send chunks for world recv chunks,
    sized by element count.

    `all_to_all_single` rather than the list-based `all_to_all` because gloo supports
    only the former, which is what makes this path testable with multiple CPU
    processes; nccl supports both. Each chunk is flattened to 1-D and split by numel,
    with the caller restoring the shape, so matrices with different column counts are
    fine.
    """
    input_splits = [int(c.numel()) for c in send_chunks]
    inp = torch.cat(send_chunks) if send_chunks else torch.empty(0, device=device, dtype=dtype)
    out = torch.empty(int(sum(recv_numels)), device=device, dtype=dtype)
    dist.all_to_all_single(out, inp, list(recv_numels), input_splits, group=pg)
    chunks, off = [], 0
    for numel in recv_numels:
        chunks.append(out[off : off + numel])
        off += numel
    return chunks


# ------------------------------------------------------------- parameter routing
def classify_muon_params(
    model: nn.Module,
    per_head: bool = True,
    per_head_o_proj: bool = False,
    extra_adamw_patterns: Optional[Iterable[str]] = None,
    extra_muon_patterns: Optional[Iterable[str]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Decide whether each parameter uses Muon or AdamW, and how to split per head.

    Returns {param name: {"use_muon", "head_rows", "head_axis", "reason"}}.

    Rules, in priority order:
      1. matches `extra_muon_patterns`      -> Muon (manual override)
      2. matches `extra_adamw_patterns`     -> AdamW (manual override)
      3. ndim is not 2 or 3                 -> AdamW (norm / bias / conv)
      4. belongs to an nn.Embedding         -> AdamW
      5. output head (final Linear to vocab) -> AdamW
      6. everything else that is 2-D or 3-D -> Muon

    Rule 4 deliberately does not match on names: in this model `img_embeddings`
    (nn.Embedding) and `img_embeddings_proj` (Linear) differ by a suffix only, so
    substring matching would pull the projection into AdamW as well -- silently, with
    no error, leaving that layer on the wrong update rule.

    Rule 5 has to use names: output heads are ordinary Linear layers, structurally
    indistinguishable from hidden ones.
    """
    extra_adamw = list(extra_adamw_patterns or [])
    extra_muon = list(extra_muon_patterns or [])

    # Output heads to vocab / codebooks, plus bare embedding-like Parameters that
    # belong to no Module. K2's rule: embeddings, LM head and biases use AdamW.
    HEAD_LIKE = (
        "lm_head.",              # text output head
        "lm_img_head.",          # image codebook head (note: does not contain "lm_head")
        "channel_embed",         # bare nn.Parameter in multi_codebook_head, embedding-like
        "channel_mask",
        "timesteps_embeddings",
    )

    param_to_module: Dict[int, nn.Module] = {}
    for _, mod in model.named_modules():
        for p in mod.parameters(recurse=False):
            param_to_module[id(p)] = mod

    # per-head: read head_dim from what the attention module declares, never guess.
    # This targets projections whose out_features are heads concatenated -- the fused
    # query_key_value ((n_heads + 2*n_kv) * head_dim, hidden) and the ViT-style qkv
    # ((3*dim, dim)) inside multi_codebook_head. Both split by head_dim rows.
    head_spec: Dict[int, Tuple[int, int]] = {}
    if per_head:
        for _, mod in model.named_modules():
            hd = getattr(mod, "head_dim", None)
            if not isinstance(hd, int) or hd <= 0:
                continue
            for child_name in ("query_key_value", "qkv", "q_proj", "k_proj", "v_proj"):
                child = getattr(mod, child_name, None)
                if isinstance(child, nn.Linear) and child.weight.size(0) % hd == 0:
                    head_spec[id(child.weight)] = (hd, -2)
            if per_head_o_proj:
                # For o_proj it is *in_features* that concatenates heads, so split
                # by column. K3 names only Q/K/V, so this is off by default.
                for child_name in ("dense", "o_proj", "proj", "out_proj"):
                    child = getattr(mod, child_name, None)
                    if isinstance(child, nn.Linear) and child.weight.size(1) % hd == 0:
                        head_spec[id(child.weight)] = (hd, -1)

    out: Dict[str, Dict[str, Any]] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        mod = param_to_module.get(id(p))

        if any(s in name for s in extra_muon):
            if p.ndim not in (2, 3):
                raise ValueError(
                    f"extra_muon_patterns matched {name!r} (ndim={p.ndim}); Muon "
                    f"supports 2-D (plain Linear) and 3-D (fused MoE expert stack) "
                    f"parameters only. For 4-D and above (e.g. conv) NS would silently "
                    f"treat the leading dims as batch, which is wrong; use AdamW."
                )
            spec = (True, "muon_pattern")
        elif any(s in name for s in extra_adamw):
            spec = (False, "adamw_pattern")
        elif p.ndim not in (2, 3):
            # ndim < 2: vectors such as norm / bias / expert_bias, AdamW by design.
            # ndim > 3: 4-D and above (e.g. conv); NS would orthogonalize each leading
            #           slice as batch, which is wrong, so these fall back to AdamW.
            #           This model has none; the branch is defensive.
            spec = (False, "ndim not in (2,3)")
        elif isinstance(mod, nn.Embedding):
            spec = (False, "nn.Embedding")
        elif any(s in name for s in HEAD_LIKE):
            spec = (False, "output_head/embedding_param")
        else:
            spec = (True, "matrix")

        use_muon, reason = spec
        hr, ax = head_spec.get(id(p), (0, -2)) if use_muon else (0, -2)
        out[name] = {
            "use_muon": use_muon,
            "head_rows": hr,
            "head_axis": ax,
            "reason": reason + (f"+per_head({hr},axis={ax})" if hr else ""),
        }
    return out


# --------------------------------------------------------------------- optimizer
class Muon(Optimizer):
    """Per-Head Muon and AdamW in one optimizer instance, so a single lr scheduler
    and DCP state cover both.

    Each param_group picks its update rule with `use_muon`:

      use_muon=True  —— Muon:
          M   <- momentum * M + G
          U   <- G + momentum * M      (nesterov=True) or M
          O   <- NS5(U), applied per head block in per-head mode
          W   <- W - lr * (rms_scale * sqrt(max(m,n)) * O + weight_decay * W)
      use_muon=False -- standard decoupled AdamW.

    Muon keys per group: `momentum`, `nesterov`, `ns_steps`, `rms_scale`, `head_rows`,
    `head_axis`. AdamW keys: `betas`, `eps`. Both share `lr` and `weight_decay`.
    `owner_alltoall` is the master switch for the experimental distributed fast path
    and is off by default.

    On HSDP: dp_replicate replicas hold the same shard and each runs NS on the same
    input. Gradients come from an all-reduce and are bit-identical, and NS is only
    deterministic matmuls, so the replicas agree. This is the same assumption AdamW
    already makes when it recomputes redundantly under HSDP.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_decay: float = 0.1,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        rms_scale: float = 0.2,
        ns_dtype: torch.dtype = torch.bfloat16,
        gather_dtype: Optional[torch.dtype] = None,
        owner_alltoall: bool = False,
    ):
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            rms_scale=rms_scale,
            use_muon=True,
            head_rows=0,
            head_axis=-2,
        )
        super().__init__(params, defaults)
        for group in self.param_groups:
            if not group.get("use_muon", True):
                continue
            head_rows = int(group.get("head_rows", 0))
            head_axis = int(group.get("head_axis", -2))
            if head_rows > 0 and head_axis not in (-2, -1):
                raise ValueError(f"Muon head_axis must be -2 or -1, got {head_axis}")
            for param in group["params"]:
                if param.ndim not in (2, 3):
                    raise ValueError(
                        f"A Muon param_group supports 2-D/3-D parameters only, got "
                        f"shape={tuple(param.shape)}; route norm/bias/embedding/conv "
                        f"to AdamW."
                    )
                if head_rows > 0 and param.size(head_axis) % head_rows:
                    raise ValueError(
                        f"Muon per-head dim {param.size(head_axis)} is not divisible by head_rows={head_rows}"
                    )
        self.ns_dtype = ns_dtype
        # None means use the parameter's own dtype (fp32 under FSDP2 mixed precision).
        # Setting bfloat16 halves the all-gather traffic, at the cost of quantizing
        # elements before the gather.
        self.gather_dtype = gather_dtype
        # all_to_all_single is still marked experimental in torch 2.6. A CPU/Gloo
        # regression proves the numerics and the collective pairing, not that the
        # target NCCL is stable for cross-node variable or zero-sized splits.
        self.owner_alltoall = bool(owner_alltoall)

    # ------------------------------------------------------------------ resume
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Load optimizer state while *keeping this run's Muon routing config*.

        `use_muon`, `head_rows` and `head_axis` describe how a matrix is split and
        which update rule it takes. They are configuration recomputed at construction
        from the current model structure, not resumable state (momentum_buffer is the
        state). torch's `Optimizer.load_state_dict` restores every param_group
        hyperparameter from the checkpoint verbatim, so a checkpoint written with a
        different per-head config (head_rows=4, say) would silently change this run's
        head_rows=2 back to 4 with no error, and NS would split on the wrong head
        boundary. This wraps the load to force this run's config to win.

        param_group order is stable across a DCP resume (groups are rebuilt from the
        live optimizer before delegating here), so aligning by position is safe.
        """
        configured = [
            (g.get("use_muon", True), g.get("head_rows", 0), g.get("head_axis", -2))
            for g in self.param_groups
        ]
        super().load_state_dict(state_dict)
        mismatched = 0
        for g, (use_muon, hr, ax) in zip(self.param_groups, configured):
            if (g.get("use_muon"), g.get("head_rows"), g.get("head_axis")) != (use_muon, hr, ax):
                mismatched += 1
            g["use_muon"], g["head_rows"], g["head_axis"] = use_muon, hr, ax
        if mismatched:
            # warnings rather than a logger: this file deliberately depends on torch only.
            warnings.warn(
                f"[Muon] {mismatched} param_group(s) in the checkpoint disagree with "
                f"this run on (use_muon/head_rows/head_axis); this run's config was "
                f"kept. Ignore this if the per-head split changed on purpose; "
                f"otherwise check that the checkpoint matches.",
                stacklevel=2,
            )

    # ------------------------------------------------------------------ Muon
    @torch.no_grad()
    def _step_muon(self, group: Dict[str, Any]) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        mom = group["momentum"]
        nesterov = group["nesterov"]
        # The per-head / NS config is uniform within a group (build_muon_param_groups
        # groups by exactly these keys), so pack it once and hand it to both paths; the
        # owner runs NS with it after assembling the full matrix.
        ns_cfg = dict(
            head_rows=group.get("head_rows", 0),
            head_axis=group.get("head_axis", -2),
            ns_steps=group["ns_steps"],
            rms_scale=group["rms_scale"],
            ns_dtype=self.ns_dtype,
        )

        # Staging buckets for the all-to-all fast path, keyed by (mesh, shard mesh
        # dim, comm dtype): flush once shard-world matrices have accumulated, which
        # bounds the temporary buffer at shard-world matrices.
        a2a_buckets: Dict[Tuple[Any, int, torch.dtype], List[Tuple[torch.Tensor, torch.Tensor]]] = {}

        for p in group["params"]:
            g = p.grad
            # Skipping on grad is None is safe: under FSDP2 "this parameter has no
            # gradient this step" holds on every rank at once (a parameter that did
            # not take part in the forward is None everywhere), so no rank can skip a
            # collective that others enter.
            if g is None:
                continue
            state = self.state[p]
            if "momentum_buffer" not in state:
                # zeros_like(p) keeps the buffer a DTensor when p is one, which is
                # what lets DCP's get_optimizer_state_dict shard and save it.
                state["momentum_buffer"] = torch.zeros_like(p)

            buf = state["momentum_buffer"]
            g_l, buf_l = _loc(g), _loc(buf)

            # Momentum accumulates entirely on the local shard; no communication.
            # An empty local shard must still run the rest of this loop: with an uneven
            # split (a 256-row router across 512 ranks, say) trailing ranks get 0 rows
            # but must still enter the collectives. Skipping here would leave the other
            # ranks waiting forever, as a silent hang that is very hard to trace. All
            # of these ops are no-ops on an empty tensor.
            buf_l.mul_(mom).add_(g_l)
            # Nesterov: U = G + mom * M. Same placements as g, so elementwise and
            # still no communication.
            u = g.add(buf, alpha=mom) if nesterov else buf

            mesh_dim = _fsdp_all2all_fast_path_mesh_dim(u) if self.owner_alltoall else None
            if mesh_dim is not None:
                # 2-D parameter on exactly one Shard(0) mesh dim: batch it inside
                # that shard group for the all-to-all.
                mesh = u.device_mesh
                comm_dtype = self.gather_dtype or _loc(u).dtype
                bucket = a2a_buckets.setdefault((mesh, mesh_dim, comm_dtype), [])
                bucket.append((p, u))
                if len(bucket) == mesh.size(mesh_dim):
                    self._flush_a2a_chunk(bucket, mesh, mesh_dim, comm_dtype, ns_cfg, lr, wd)
            else:
                # 3-D MoE, several shard dims, unsharded, or column shards: the
                # redistribute slow path, numerically equivalent.
                self._apply_redistribute(p, u, ns_cfg, lr, wd)

        # Drain the partial tail of each bucket. Bucket contents are identical on
        # every rank (guaranteed by group['params'] order plus every Muon parameter
        # having a gradient on every rank), and so is the iteration order.
        for (mesh, mesh_dim, comm_dtype), bucket in a2a_buckets.items():
            self._flush_a2a_chunk(bucket, mesh, mesh_dim, comm_dtype, ns_cfg, lr, wd)

    @torch.no_grad()
    def _apply_redistribute(
        self, p: torch.Tensor, u: torch.Tensor, ns_cfg: Dict[str, Any], lr: float, wd: float
    ) -> None:
        """Slow path: redistribute to the full matrix, NS on each rank, slice back."""
        u_full, tgt = _matrix_full_local(u, gather_dtype=self.gather_dtype)
        if tgt is None and u_full.numel() == 0:
            # Only the no-communication branch may return early (this rank was
            # assigned no experts, for instance). When tgt is not None, u_full is the
            # materialized full matrix and cannot be empty.
            return
        o_full = muon_update(u_full, **ns_cfg)
        o_l = _scatter_back(o_full, u, tgt)
        self._apply_update(p, o_l, lr, wd)

    @staticmethod
    def _apply_update(p: torch.Tensor, o_l: torch.Tensor, lr: float, wd: float) -> None:
        """Apply the orthogonalized local-shard update (decoupled weight decay uses
        the base lr)."""
        p_l = _loc(p)
        if wd:
            p_l.mul_(1.0 - lr * wd)
        p_l.add_(o_l.to(p_l.dtype), alpha=-lr)

    @torch.no_grad()
    def _flush_a2a_chunk(
        self,
        bucket: List[Tuple[torch.Tensor, torch.Tensor]],
        mesh: Any,
        mesh_dim: int,
        comm_dtype: torch.dtype,
        ns_cfg: Dict[str, Any],
        lr: float,
        wd: float,
    ) -> None:
        if not bucket:
            return
        updates = [u for _p, u in bucket]
        local_orthos = self._ortho_fsdp_all2all(updates, mesh, mesh_dim, comm_dtype, ns_cfg)
        for (p, _u), o_l in zip(bucket, local_orthos):
            self._apply_update(p, o_l, lr, wd)
        bucket.clear()

    @torch.no_grad()
    def _ortho_fsdp_all2all(
        self,
        updates: List[torch.Tensor],
        mesh: Any,
        mesh_dim: int,
        comm_dtype: torch.dtype,
        ns_cfg: Dict[str, Any],
    ) -> List[torch.Tensor]:
        """Orthogonalize a batch of 2-D Shard(0) updates with an owner-based
        all-to-all; returns each rank's local row shard.

        The j-th matrix in a group is owned by rank j: the forward all-to-all
        assembles its full [M, K] from the per-rank row shards, the owner runs NS once,
        and the reverse all-to-all sends the resulting row shards back. This computes
        the same full matrix as the redistribute path and is numerically equivalent,
        but NS runs once instead of per rank and the traffic is a balanced all-to-all.

        Invariant: every rank must enter with the *same updates order and length*.
        all-to-all pairs by position, so any misalignment between ranks deadlocks --
        the same precondition the all-gather path relies on.
        """
        world = mesh.size(mesh_dim)
        if len(updates) > world:
            raise ValueError(f"an all-to-all group holds at most {world} matrices, got {len(updates)}")
        n = len(updates)
        my_idx = int(mesh.get_coordinate()[mesh_dim])
        pg = mesh.get_group(mesh_dim)
        device = _loc(updates[0]).device
        ortho_dtype = self.ns_dtype  # muon_update output dtype, used for the reverse
        # exchange; identical on every rank

        # The matrix this rank owns (group base 0, so the owner index == my_idx).
        owned = updates[my_idx] if my_idx < n else None
        owned_k = int(owned.shape[1]) if owned is not None else 0
        owned_rows = _shard_row_sizes(int(owned.shape[0]), world) if owned is not None else [0] * world

        # --- Forward: send this rank's row shard of matrix j to owner j, and as the
        #     owner, receive all world row shards of its own matrix. ---
        send = [
            (_loc(updates[j]).contiguous().reshape(-1).to(comm_dtype))
            if j < n
            else torch.empty(0, device=device, dtype=comm_dtype)
            for j in range(world)
        ]
        recv_numels = [owned_rows[i] * owned_k for i in range(world)]
        recv = _all_to_all_flat(send, recv_numels, pg, device, comm_dtype)

        if owned is not None:
            full_grad = torch.cat([recv[i].reshape(owned_rows[i], owned_k) for i in range(world)], dim=0)
            full_ortho = muon_update(full_grad, **ns_cfg).to(ortho_dtype)
            ortho_shards = list(torch.split(full_ortho, owned_rows, dim=0))
        else:
            ortho_shards = [torch.empty(0, 0, device=device, dtype=ortho_dtype)] * world

        # --- Reverse: the owner returns each rank its row shard; this rank collects
        #     its shard of every matrix. ---
        send2 = [ortho_shards[i].contiguous().reshape(-1) for i in range(world)]
        my_rc: List[Tuple[int, int]] = []
        for j in range(world):
            if j < n:
                my_rc.append((_shard_row_sizes(int(updates[j].shape[0]), world)[my_idx], int(updates[j].shape[1])))
            else:
                my_rc.append((0, 0))
        recv2 = _all_to_all_flat(send2, [r * k for r, k in my_rc], pg, device, ortho_dtype)

        return [recv2[j].reshape(my_rc[j][0], my_rc[j][1]) for j in range(n)]

    # ------------------------------------------------------------------ AdamW
    @torch.no_grad()
    def _step_adamw(self, group: Dict[str, Any]) -> None:
        lr = group["lr"]
        wd = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for p in group["params"]:
            g = p.grad
            if g is None:
                continue
            state = self.state[p]
            if len(state) == 0:
                # Keep step on CPU: bias correction needs .item() every step, and on
                # device that becomes one device->host sync per parameter, i.e.
                # hundreds of bubbles. torch's own AdamW also keeps step on CPU on the
                # non-capturable path.
                state["step"] = torch.zeros((), dtype=torch.float32)
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            p_l, g_l = _loc(p), _loc(g)
            # step counts updates of the *global* parameter, not elements of this
            # rank's shard. Under an uneven Shard a trailing rank may hold an empty
            # shard, but its step must still advance with the others; otherwise
            # resuming at a different world size gives AdamW a different timestep for
            # bias correction.
            state["step"] += 1
            if p_l.numel() == 0:
                continue
            m_l, v_l = _loc(state["exp_avg"]), _loc(state["exp_avg_sq"])

            t = float(state["step"].item())

            if wd:
                p_l.mul_(1.0 - lr * wd)
            m_l.mul_(beta1).add_(g_l, alpha=1 - beta1)
            v_l.mul_(beta2).addcmul_(g_l, g_l, value=1 - beta2)

            bc1 = 1 - beta1**t
            bc2 = 1 - beta2**t
            denom = (v_l.sqrt() / math.sqrt(bc2)).add_(eps)
            p_l.addcdiv_(m_l, denom, value=-lr / bc1)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            if group.get("use_muon", True):
                self._step_muon(group)
            else:
                self._step_adamw(group)
        return loss


def build_muon_param_groups(
    model: nn.Module,
    base_groups: List[Dict[str, Any]],
    per_head: bool = True,
    per_head_o_proj: bool = False,
    extra_adamw_patterns: Optional[Iterable[str]] = None,
    extra_muon_patterns: Optional[Iterable[str]] = None,
) -> Tuple[List[Dict[str, Any]], List[str]]:
    """Subdivide existing param_groups (already split by lr / weight decay) by Muon
    routing.

    Each input group becomes several groups that agree on
    (use_muon, head_rows, head_axis); every other key (lr, weight_decay, ...) is
    inherited unchanged. Returns (new groups, human-readable summary lines).

    Ending up with more groups is expected: the lr scheduler scales each group's
    initial_lr independently, so the split does not change any lr value, only which
    update rule each group takes.
    """
    spec = classify_muon_params(
        model,
        per_head=per_head,
        per_head_o_proj=per_head_o_proj,
        extra_adamw_patterns=extra_adamw_patterns,
        extra_muon_patterns=extra_muon_patterns,
    )
    by_param = {id(p): (n, spec[n]) for n, p in model.named_parameters() if n in spec}

    new_groups: List[Dict[str, Any]] = []
    lines: List[str] = []
    for gi, g in enumerate(base_groups):
        buckets: Dict[Tuple[bool, int, int], List[torch.nn.Parameter]] = {}
        counts: Dict[Tuple[bool, int, int], int] = {}
        for p in g["params"]:
            entry = by_param.get(id(p))
            if entry is None:
                # Fallback: anything missing from the spec goes to AdamW. Better to
                # fall back to known behaviour than to update an unrecognized tensor
                # with an unverified rule.
                key = (False, 0, -2)
            else:
                s = entry[1]
                key = (s["use_muon"], s["head_rows"], s["head_axis"])
            buckets.setdefault(key, []).append(p)
            counts[key] = counts.get(key, 0) + p.numel()

        for key in sorted(buckets, key=lambda k: (not k[0], k[1], k[2])):
            use_muon, hr, ax = key
            ng = dict(g)
            ng["params"] = buckets[key]
            ng["use_muon"] = use_muon
            ng["head_rows"] = hr
            ng["head_axis"] = ax
            new_groups.append(ng)
            tag = "muon" + (f"/per-head({hr},ax{ax})" if hr else "") if use_muon else "adamw"
            lines.append(
                f"  group[{gi}] -> {tag:26s} lr={g.get('lr', float('nan')):.2e} "
                f"wd={g.get('weight_decay', 0.0):g} tensors {len(buckets[key]):4d} params {counts[key]:,}"
            )
    return new_groups, lines

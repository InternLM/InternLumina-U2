from .clip_grad_norm import clip_grad_norm

# Apply the torch-2.6 FSDP2 all-gather version-counter fix on import (no-op on
# torch >= 2.7, where upstream carries it). See _fsdp_common.py for the story.
from ._fsdp_common import apply_patch as _apply_fsdp2_vc_patch


_apply_fsdp2_vc_patch()

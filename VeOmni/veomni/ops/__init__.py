# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch

from ..utils.device import get_device_type
from ..utils.import_utils import is_fused_moe_available
from .attention import flash_attention_forward
from .loss import causallm_loss_function


_fused_moe_kernel = None


def _get_fused_moe_kernel():
    """Resolve the fused MoE kernel matching the current device, once per process."""
    global _fused_moe_kernel
    if _fused_moe_kernel is None:
        if get_device_type() == "npu":
            from .npu_group_gemm import npu_fused_moe_forward

            _fused_moe_kernel = npu_fused_moe_forward
        elif is_fused_moe_available():
            from .fused_moe import fused_moe_forward as triton_fused_moe_forward

            _fused_moe_kernel = triton_fused_moe_forward
        else:
            raise RuntimeError(
                "No fused MoE kernel is available, it requires either torch_npu on Ascend NPU "
                "or triton on a CUDA device."
            )

    return _fused_moe_kernel


def fused_moe_forward(
    module: torch.nn.Module,
    num_experts: int,
    routing_weights: torch.Tensor,
    selected_experts: torch.Tensor,
    hidden_states: torch.Tensor,
    fc1_1_weight: torch.Tensor,
    fc1_2_weight: torch.Tensor,
    fc2_weight: torch.Tensor,
):
    return _get_fused_moe_kernel()(
        module,
        num_experts,
        routing_weights,
        selected_experts,
        hidden_states,
        fc1_1_weight,
        fc1_2_weight,
        fc2_weight,
    )


__all__ = [
    "flash_attention_forward",
    "fused_moe_forward",
    "causallm_loss_function",
]

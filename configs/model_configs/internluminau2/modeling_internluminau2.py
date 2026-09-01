# coding=utf-8
# Copyright 2025 Antgroup and The HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaDA2MoE model."""

import math
import os
import warnings
import threading
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_attn_mask_utils import (
    AttentionMaskConverter,
    _prepare_4d_attention_mask,
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)
from transformers.modeling_outputs import (
    MoeModelOutputWithPast,
    MoeCausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
from transformers.modeling_utils import PreTrainedModel, ALL_ATTENTION_FUNCTIONS
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_13
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.utils.import_utils import is_torch_fx_available
from .configuration_internluminau2 import LLaDA2MoeConfig
from transformers.generation.utils import GenerationMixin
from veomni.ops import fused_moe_forward
from veomni.distributed.parallel_state import get_parallel_state
from veomni.utils.import_utils import is_liger_kernel_available
from veomni.utils import logging

from .tokenbridge import CausalBlock, FinalLayer
from functools import partial


if is_liger_kernel_available():
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    from liger_kernel.transformers.rms_norm import LigerRMSNorm
    from liger_kernel.transformers.rope import liger_rotary_pos_emb

# This makes `_prepare_4d_causal_attention_mask` a leaf function in the FX graph.
# It means that the function will not be traced through and simply appear as a node in the graph.
if is_torch_fx_available():
    if not is_torch_greater_or_equal_than_1_13:
        import torch.fx

    _prepare_4d_causal_attention_mask = torch.fx.wrap(_prepare_4d_causal_attention_mask)


logger = logging.get_logger(__name__)

# Global state for expert tracking
_EXPERT_COUNTS = None
_TRACKING_ENABLED = False
_TRACKING_LOCK = threading.Lock()

def enable_expert_tracking(num_layers, num_experts):
    global _EXPERT_COUNTS, _TRACKING_ENABLED
    with _TRACKING_LOCK:
        _EXPERT_COUNTS = torch.zeros(num_layers, num_experts, dtype=torch.long)
        _TRACKING_ENABLED = True

def disable_expert_tracking():
    global _TRACKING_ENABLED
    with _TRACKING_LOCK:
        _TRACKING_ENABLED = False

def get_expert_stats():
    global _EXPERT_COUNTS
    with _TRACKING_LOCK:
        if _EXPERT_COUNTS is not None:
            return _EXPERT_COUNTS.clone()
        return None

def update_expert_counts(layer_idx, topk_idx):
    global _EXPERT_COUNTS, _TRACKING_ENABLED
    if not _TRACKING_ENABLED or layer_idx < 0:
        return
    
    # topk_idx shape: [batch, seq, top_k] or [batch*seq, top_k]
    flat_indices = topk_idx.view(-1)
    
    # Use torch.bincount for efficiency
    counts = torch.bincount(flat_indices, minlength=_EXPERT_COUNTS.shape[1])
    
    with _TRACKING_LOCK:
        if _EXPERT_COUNTS is not None:
            _EXPERT_COUNTS[layer_idx] += counts.cpu()

    
#     # router_logits is actually logits, convert to probs
#     probs = torch.softmax(router_logits, dim=-1)
    
#     # P_i: mean probability assigned to expert i
#     # shape: (num_experts)
#     P_i = torch.mean(probs, dim=0)
    
    
#     # Load balancing loss = N * sum(f_i * P_i)
#     loss = num_experts * torch.sum(f_i * P_i)
#     return loss


def load_balancing_loss_func(router_logits, num_experts, topk_idx=None):
    if router_logits is None or num_experts <= 1:
        return 0.0

    if router_logits.dim() == 3:
        router_logits = router_logits.reshape(-1, router_logits.size(-1))
    if topk_idx is not None and topk_idx.dim() == 3:
        topk_idx = topk_idx.reshape(-1, topk_idx.size(-1))

    probs = torch.softmax(router_logits, dim=-1)          # (B*S, E)
    P_i = probs.mean(dim=0)                               # (E,) 且 sum(P_i)=1

    if topk_idx is None:
        topk_idx = torch.topk(router_logits, k=1, dim=-1)[1]  # (B*S, 1)

    tokens_per_expert = torch.bincount(topk_idx.view(-1), minlength=num_experts)
    f_i = tokens_per_expert.float() / topk_idx.numel()    # (E,) 且 sum(f_i)=1

    loss = num_experts * torch.sum(f_i * P_i)             # 上界为 num_experts
    return loss

_CONFIG_FOR_DOC = "LLaDA2MoeConfig"


def _get_unpad_data(attention_mask):
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


class LLaDA2MoeRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LLaDA2MoeRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


ALL_LAYERNORM_LAYERS.append(LLaDA2MoeRMSNorm)


class LLaDA2MoeRotaryEmbedding(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    @torch.no_grad()
    @dynamic_rope_update  # power user: used with advanced RoPE types (e.g. dynamic rope)
    def forward(self, x, position_ids):
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):  # Force float32
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos() * self.attention_scaling
            sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.
    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`):
            The position indices of the tokens corresponding to the query and key tensors. For example, this can be
            used to pass offsetted position ids when working with a KV-cache.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    # Keep half or full tensor for later concatenation
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    # Apply rotary embeddings on the first half or full tensor
    q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)

    # Concatenate back to full shape
    q_embed = torch.cat([q_embed, q_pass], dim=-1)
    k_embed = torch.cat([k_embed, k_pass], dim=-1)
    return q_embed, k_embed


class LLaDA2MoeMLP(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size

        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        if is_liger_kernel_available():
            return self.down_proj(LigerSiLUMulFunction.apply(self.gate_proj(x), self.up_proj(x)))
        else:
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LLaDA2MoeGate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts

        self.n_group = config.n_group
        self.topk_group = config.topk_group

        # topk selection algorithm
        self.gating_dim = config.hidden_size
        self.weight = nn.Parameter(torch.empty((self.num_experts, self.gating_dim)))
        self.routed_scaling_factor = config.routed_scaling_factor

        self.register_buffer("expert_bias", torch.zeros((self.num_experts)))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        import torch.nn.init as init

        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def group_limited_topk(
        self,
        scores: torch.Tensor,
    ):
        num_tokens, _ = scores.size()
        # Organize the experts into groups
        group_scores = scores.view(num_tokens, self.n_group, -1).topk(2, dim=-1)[0].sum(dim=-1)
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Mask the experts based on selection groups
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(num_tokens, self.n_group, self.num_experts // self.n_group)
            .reshape(num_tokens, -1)
        )

        masked_scores = scores.masked_fill(~score_mask.bool(), float('-inf'))
        probs, top_indices = torch.topk(masked_scores, k=self.top_k, dim=-1)

        return probs, top_indices

    def forward(self, hidden_states):
        # compute gating score
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        logits = F.linear(hidden_states.type(torch.float32), self.weight.type(torch.float32))
        # logits = F.linear(hidden_states, self.weight)

        scores = torch.sigmoid(logits.float()).type_as(logits)
        # scores = torch.sigmoid(logits).type_as(logits)

        scores_for_routing = scores + self.expert_bias
        _, topk_idx = self.group_limited_topk(scores_for_routing)

        scores = torch.gather(scores, dim=1, index=topk_idx).type_as(logits)

        topk_weight = scores / (scores.sum(dim=-1, keepdim=True) + 1e-20) if self.top_k > 1 else scores
        topk_weight = topk_weight * self.routed_scaling_factor

        return topk_idx, topk_weight, logits


class LLaDA2MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_size = config.moe_intermediate_size
        self.gate_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.intermediate_size, self.hidden_dim),
            requires_grad=True,
        )
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_size),
            requires_grad=True,
        )
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, hidden_states, expert_idx=None, routing_weights=None, selected_experts=None):
        if expert_idx is not None:
            assert not get_parallel_state().ep_enabled, "_moe_implementation=`eager` does not support EP"
            gate_proj_out = torch.matmul(hidden_states, self.gate_proj[expert_idx].transpose(0, 1))
            up_proj_out = torch.matmul(hidden_states, self.up_proj[expert_idx].transpose(0, 1))

            out = self.act_fn(gate_proj_out) * up_proj_out
            out = torch.matmul(out, self.down_proj[expert_idx].transpose(0, 1))
        else:
            assert routing_weights is not None and selected_experts is not None, (
                "routing_weights and selected_experts must be provided when expert_idx is None"
            )

            out = fused_moe_forward(
                module=self,
                num_experts=self.num_experts,
                routing_weights=routing_weights,
                selected_experts=selected_experts,
                hidden_states=hidden_states,
                fc1_1_weight=self.gate_proj,
                fc1_2_weight=self.up_proj,
                fc2_weight=self.down_proj,
            )
        return out

    def reset_parameters(self):
        """
        Initialize the parameters of all expert networks.
        Uses different initialization strategies for different projection layers.
        """
        for expert_id in range(self.num_experts):
            nn.init.kaiming_uniform_(self.gate_proj[expert_id], a=math.sqrt(5))            
            nn.init.kaiming_uniform_(self.up_proj[expert_id], a=math.sqrt(5))
            nn.init.xavier_uniform_(self.down_proj[expert_id])


class LLaDA2MoeSparseMoeBlock(nn.Module):
    """
    A mixed expert module containing shared experts.
    """

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int = -1):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.num_experts_per_tok = config.num_experts_per_tok
        if self.config.model_type == "llada2_moe_veomni":
            self._setup_fuse_moe_experts()
        else:
            self._setup_experts()

        self.gate = LLaDA2MoeGate(config)
        if config.num_shared_experts is not None:
            self.shared_experts = LLaDA2MoeMLP(
                config=config, intermediate_size=config.moe_intermediate_size * config.num_shared_experts
            )
    
    def _setup_fuse_moe_experts(self):
        self.experts = LLaDA2MoeExperts(self.config)

    def _setup_experts(self):
        self.experts = nn.ModuleList(
            [
                LLaDA2MoeMLP(config=self.config, intermediate_size=self.config.moe_intermediate_size)
                for _ in range(self.config.num_experts)
            ]
        )

    def _fuse_moe_forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        if _TRACKING_ENABLED:
            update_expert_counts(self.layer_idx, topk_idx)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        y = self.experts(
            hidden_states, routing_weights=topk_weight, selected_experts=topk_idx
        ).reshape(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        router_logits_out = router_logits.to(dtype=identity.dtype).view(bsz, seq_len, -1)
        return y, (router_logits_out, topk_idx.view(bsz, seq_len, -1))

    def _forward(self, hidden_states):
        identity = hidden_states
        bsz, seq_len, h = hidden_states.shape
        topk_idx, topk_weight, router_logits = self.gate(hidden_states)
        if _TRACKING_ENABLED:
            update_expert_counts(self.layer_idx, topk_idx)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        flat_topk_idx = topk_idx.view(-1)
        if self.training:
            hidden_states = hidden_states.repeat_interleave(self.num_experts_per_tok, dim=0)
            y = torch.empty_like(hidden_states)
            for i, expert in enumerate(self.experts):
                y[flat_topk_idx == i] = expert(hidden_states[flat_topk_idx == i])
            y = (y.view(*topk_weight.shape, -1) * topk_weight.unsqueeze(-1)).sum(dim=1)
            y = y.to(hidden_states.dtype).view(bsz, seq_len, h)
        else:
            y = self.moe_infer(hidden_states, topk_idx, topk_weight).view(bsz, seq_len, h)
        if self.config.num_shared_experts is not None:
            y = y + self.shared_experts(identity)
        router_logits_out = router_logits.to(dtype=identity.dtype).view(bsz, seq_len, -1)
        return y, (router_logits_out, topk_idx.view(bsz, seq_len, -1))

    def forward(self, hidden_states):
        # TODO (zhiguang): make a flag here for selecting different forward in different type
        if self.config.model_type == "llada2_moe_veomni":
            return self._fuse_moe_forward(hidden_states)
        else:
            return self._forward(hidden_states)

    @torch.no_grad()
    def moe_infer(self, x, topk_ids, topk_weight):
        cnts = topk_ids.new_zeros((topk_ids.shape[0], len(self.experts)))
        cnts.scatter_(1, topk_ids, 1)
        tokens_per_expert = cnts.sum(dim=0)
        idxs = topk_ids.view(-1).argsort()
        sorted_tokens = x[idxs // topk_ids.shape[1]]
        sorted_tokens_shape = sorted_tokens.shape
        tokens_per_expert = tokens_per_expert.cpu().numpy()
        outputs = []
        start_idx = 0
        for i, num_tokens in enumerate(tokens_per_expert):
            end_idx = start_idx + num_tokens
            if num_tokens == 0:
                continue
            expert = self.experts[i]
            tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
            expert_out = expert(tokens_for_this_expert)
            outputs.append(expert_out.to(x.device))
            start_idx = end_idx

        outs = torch.cat(outputs, dim=0) if len(outputs) else sorted_tokens.new_empty(0)
        new_x = torch.empty_like(outs)
        new_x[idxs] = outs
        final_out = (
            new_x.view(*topk_ids.shape, -1)
            .type(topk_weight.dtype)
            .mul_(topk_weight.unsqueeze(dim=-1))
            .sum(dim=1)
            .type(new_x.dtype)
        )
        return final_out


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaAttention with Llama->LLaDA2Moe
class LLaDA2MoeAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    # Number of forwards that took the segmented path; parity harnesses assert on it so
    # that "no difference" can never be confused with "the segmented path never ran".
    _seg_calls = 0

    def __init__(self, config: LLaDA2MoeConfig, layer_idx: Optional[int] = None):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        if layer_idx is None:
            logger.warning_once(
                f"Instantiating {self.__class__.__name__} without passing `layer_idx` is not recommended and will "
                "to errors during the forward call, if caching is used. Please make sure to provide a `layer_idx` "
                "when creating this class."
            )

        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.head_dim or self.hidden_size // self.num_heads
        partial_rotary_factor = config.partial_rotary_factor if hasattr(config, "partial_rotary_factor") else 1.0
        self.rope_dim = int(self.head_dim * partial_rotary_factor)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = False

        self.query_key_value = nn.Linear(
            self.hidden_size,
            (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim,
            bias=config.use_qkv_bias,
        )

        self.query_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.key_layernorm = LLaDA2MoeRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.dense = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.use_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            if self.layer_idx is None:
                raise ValueError(
                    f"The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} "
                    "for auto-regressive decoding with k/v caching, please make sure to initialize the attention class "
                    "with a layer index."
                )
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )
        # attention_mask = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# ---------------------------------------------------------------------------
# Segmented (per-pack-segment) attention -- opt-in via config.use_segmented_attention
# ---------------------------------------------------------------------------
# The packing mask built by the backbone is strictly block-diagonal (`same_seg` is a
# hard AND term), so attention can be computed segment by segment instead of over the
# whole S x S matrix: FLOPs drop from S^2 to sum(L_i^2). This is an equality in exact
# arithmetic; whether it holds in bf16 is exactly what tools/seg_attn_parity.py measures.
#
# One case is *not* an identity if handled naively: rows that are fully masked (the SP
# tail padding, at most sp_size-1 tokens). The mask uses torch.finfo(dtype).min rather
# than -inf, so such a row degenerates into a uniform average -- over all S keys in the
# dense path, but only over its own segment if the pad were folded into the last one.
# The tail is therefore attended against the full K/V, which reproduces dense exactly.

SEG_ATTN_STRICT = os.environ.get("SEG_ATTN_STRICT", "0") == "1"


def resolve_seg_bounds(cu_seqlens, seq_len):
    """cu_seqlens -> a python tuple of segment ends, validated (never silently clipped).

    Returns None when there is nothing to segment. Called once per backbone forward so
    that the per-layer path never touches a CUDA tensor (a `.tolist()` in the layer loop
    would be one device sync per layer, doubled again by gradient-checkpoint recompute).
    """
    if cu_seqlens is None:
        return None
    bounds = [int(x) for x in cu_seqlens.tolist()]
    if len(bounds) < 2:
        return None
    if bounds[0] != 0:
        raise ValueError(f"cu_seqlens must start at 0, got {bounds[0]}")
    for prev, cur in zip(bounds, bounds[1:]):
        if cur <= prev:
            raise ValueError(f"cu_seqlens must be strictly increasing, got {bounds}")
    if bounds[-1] > seq_len:
        raise ValueError(f"cu_seqlens ends at {bounds[-1]} beyond sequence length {seq_len}")
    return tuple(bounds)


def check_seg_tail_is_padding(attention_mask, seg_bounds):
    """Assert the positions past cu_seqlens[-1] are fully masked. Strict mode only:
    reading mask values costs a device sync, so production never runs this."""
    tail = seg_bounds[-1]
    if attention_mask is None or tail >= attention_mask.shape[-2]:
        return
    if bool((attention_mask[:, :, tail:, :].max() > torch.finfo(attention_mask.dtype).min).item()):
        raise RuntimeError(
            f"positions [{tail}, {attention_mask.shape[-2]}) lie outside cu_seqlens but are not "
            "fully masked; the tail of a packed sequence must be padding only"
        )


def _segmented_sdpa(query_states, key_states, value_states, attention_mask, seg_bounds, dropout_p):
    """Block-diagonal SDPA: one kernel call per packed segment, plus one for the tail pad."""
    seq_len = query_states.shape[2]
    outputs = []
    for start, end in zip(seg_bounds, seg_bounds[1:]):
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query_states[:, :, start:end, :].contiguous(),
                key_states[:, :, start:end, :].contiguous(),
                value_states[:, :, start:end, :].contiguous(),
                attn_mask=attention_mask[:, :, start:end, start:end].contiguous(),
                dropout_p=dropout_p,
                is_causal=False,
            )
        )

    tail = seg_bounds[-1]
    if tail < seq_len:
        # Fully-masked pad rows: match the dense path by attending against all of K/V.
        # That the tail really is padding is checked once per forward in the backbone
        # (under SEG_ATTN_STRICT); doing it here would cost a device sync per layer.
        outputs.append(
            torch.nn.functional.scaled_dot_product_attention(
                query_states[:, :, tail:, :].contiguous(),
                key_states,
                value_states,
                attn_mask=attention_mask[:, :, tail:, :].contiguous(),
                dropout_p=dropout_p,
                is_causal=False,
            )
        )

    return torch.cat(outputs, dim=2)


def use_segmented_attention(module, attention_mask, seg_bounds, q_len, use_cache, past_key_value):
    """Whether this attention call takes the segmented path, and why not if it doesn't.

    Under SEG_ATTN_STRICT a config that asks for segmented attention but cannot get it
    raises instead of silently running dense -- otherwise a parity run reporting
    max_diff=0 could just mean the segmented path never executed, and a checkpoint could
    record use_segmented_attention=true while having been trained dense.
    """
    if not getattr(module.config, "use_segmented_attention", False):
        return False

    reason = None
    if seg_bounds is None:
        reason = "no cu_seqlens in the batch"
    elif use_cache or past_key_value is not None:
        reason = "incremental decoding (use_cache / past_key_value)"
    elif attention_mask is None or attention_mask.dim() != 4:
        reason = f"attention_mask is not the 4D packing mask (got {None if attention_mask is None else attention_mask.dim()}D)"
    elif attention_mask.shape[-1] != q_len or attention_mask.shape[-2] != q_len:
        reason = f"mask {tuple(attention_mask.shape)} does not match q_len {q_len}"

    if reason is not None:
        if SEG_ATTN_STRICT:
            raise RuntimeError(f"use_segmented_attention=True but the segmented path is unavailable: {reason}")
        return False

    # A single segment is a perfectly normal batch (one sample filled the pack); the
    # segmented path is then just the dense one, so run it rather than complaining.
    module._seg_calls += 1
    return True


# Copied from transformers.models.llama.modeling_llama.LlamaSdpaAttention with Llama->LLaDA2Moe
class LLaDA2MoeSdpaAttention(LLaDA2MoeAttention):
    """
    LLaDA2Moe attention module using torch.nn.functional.scaled_dot_product_attention. This module inherits from
    `LLaDA2MoeAttention` as the weights of the module stays untouched. The only changes are on the forward pass to adapt to
    SDPA API.
    """

    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        seg_bounds: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:


        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )

        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention_mask = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
        #######
        # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
        # Reference: https://github.com/pytorch/pytorch/issues/112577.
        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()
        if use_segmented_attention(self, attention_mask, seg_bounds, q_len, use_cache, past_key_value):
            attn_output = _segmented_sdpa(
                query_states,
                key_states,
                value_states,
                attention_mask,
                seg_bounds,
                self.attention_dropout if self.training else 0.0,
            )
        else:
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                # The q_len > 1 is necessary to match with AttentionMaskConverter.to_causal_4d that does not create a causal mask in case q_len == 1.
                is_causal=self.is_causal and attention_mask is None and q_len > 1,
            )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value


class LLaDA2MoeFlexAttention(LLaDA2MoeAttention):
    # Adapted from LLaDA2MoeAttention.forward
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if output_attentions:
            # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
            logger.warning_once(
                "LLaDA2MoeModel is using LLaDA2MoeFlexAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, "
                'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                output_attentions=output_attentions,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
            )
        bsz, q_len, _ = hidden_states.size()

        qkv = self.query_key_value(hidden_states)
        qkv = qkv.view(bsz, q_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)

        query_states, key_states, value_states = qkv.split(
            [self.num_heads, self.num_key_value_heads, self.num_key_value_heads], dim=-2
        )
        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        query_states = self.query_layernorm(query_states)
        key_states = self.key_layernorm(key_states)

        kv_seq_len = key_states.shape[-2]
        if past_key_value is not None:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos}  # Specific to RoPE models
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        # attention_mask = None
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
        
        attn_output, attn_weights = ALL_ATTENTION_FUNCTIONS["flex_attention"](
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            **kwargs
        )

        # attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1).contiguous()

        attn_output = self.dense(attn_output)

        return attn_output, None, past_key_value


ATTENTION_CLASSES = {
    "eager": LLaDA2MoeAttention,
    "flex_attention": LLaDA2MoeFlexAttention,
    "sdpa": LLaDA2MoeSdpaAttention,
}


class LLaDA2MoeDecoderLayer(nn.Module):
    def __init__(self, config: LLaDA2MoeConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.attention = ATTENTION_CLASSES[config._attn_implementation](config=config, layer_idx=layer_idx)

        self.mlp = (
            LLaDA2MoeSparseMoeBlock(config, layer_idx=layer_idx)
            if (config.num_experts is not None and layer_idx >= config.first_k_dense_replace)
            else LLaDA2MoeMLP(config=config, intermediate_size=config.intermediate_size)
        )
        self.input_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        output_router_logits: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        seg_bounds: Optional[Tuple[int, ...]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*):
                attention mask of size `(batch_size, sequence_length)` if flash attention is used or `(batch_size, 1,
                query_sequence_length, key_sequence_length)` if default attention is used.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
                config.n_positions - 1]`.
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*):
                cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_router_logits (`bool`, *optional*):
                Whether or not to return the logits of all the routers. They are useful for computing the router loss,
                and should not be returned during inference.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
        """
        if "padding_mask" in kwargs:
            warnings.warn(
                "Passing `padding_mask` is deprecated and will be removed in v4.37. Please make sure use `attention_mask` instead.`"
            )
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        # Self Attention
        hidden_states, self_attn_weights, present_key_value = self.attention(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            use_cache=use_cache,
            seg_bounds=seg_bounds,
        )
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        if isinstance(hidden_states, tuple):
            hidden_states, router_logits = hidden_states
        else:
            router_logits = None
        hidden_states = residual + hidden_states.to(residual.device)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        if output_router_logits:
            outputs += (router_logits,)

        return outputs


LLADA2MOE_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)
    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.
    Parameters:
        config ([`LLaDA2MoeConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoePreTrainedModel(PreTrainedModel):
    config_class = LLaDA2MoeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LLaDA2MoeDecoderLayer"]
    _skip_keys_device_placement = "past_key_values"
    # This implementation has no FlashAttention-2 class.  Claiming support here
    # makes Transformers select ``flash_attention_2`` when it is installed, then
    # the decoder fails with a KeyError because ATTENTION_CLASSES has no such
    # entry.  Inference pins SDPA in the loader; keep the capability declaration
    # honest for generic Transformers callers.
    _supports_flash_attn_2 = False
    _supports_sdpa = True
    _supports_flex_attn = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
    
    def get_parallel_plan(self):
        from .parallel_plan import get_parallel_plan

        return get_parallel_plan()


LLADA2MOE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
            it.
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            [What are input IDs?](../glossary#input-ids)
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](../glossary#attention-mask)
            Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
            [`PreTrainedTokenizer.__call__`] for details.
            If `past_key_values` is used, optionally only the last `input_ids` have to be input (see
            `past_key_values`).
            If you want to change padding behavior, you should read [`modeling_opt._prepare_decoder_attention_mask`]
            and modify to your needs. See diagram 1 in [the paper](https://arxiv.org/abs/1910.13461) for more
            information on the default strategy.
            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.
        position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Indices of positions of each input sequence tokens in the position embeddings. Selected in the range `[0,
            config.n_positions - 1]`.
            [What are position IDs?](../glossary#position-ids)
        past_key_values (`Cache` or `tuple(tuple(torch.FloatTensor))`, *optional*):
            Pre-computed hidden-states (key and values in the self-attention blocks and in the cross-attention
            blocks) that can be used to speed up sequential decoding. This typically consists in the `past_key_values`
            returned by the model at a previous stage of decoding, when `use_cache=True` or `config.use_cache=True`.
            Two formats are allowed:
            - a [`~cache_utils.Cache`] instance;
            - Tuple of `tuple(torch.FloatTensor)` of length `config.n_layers`, with each tuple having 2 tensors of
            shape `(batch_size, num_heads, sequence_length, embed_size_per_head)`). This is also known as the legacy
            cache format.
            The model will output the same cache format that is fed as input. If no `past_key_values` are passed, the
            legacy cache format will be returned.
            If `past_key_values` are used, the user can optionally input only the last `input_ids` (those that don't
            have their past key value states given to this model) of shape `(batch_size, 1)` instead of all `input_ids`
            of shape `(batch_size, sequence_length)`.
        inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
            Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation. This
            is useful if you want more control over how to convert `input_ids` indices into associated vectors than the
            model's internal embedding lookup matrix.
        use_cache (`bool`, *optional*):
            If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding (see
            `past_key_values`).
        output_attentions (`bool`, *optional*):
            Whether or not to return the attentions tensors of all attention layers. See `attentions` under returned
            tensors for more detail.
        output_hidden_states (`bool`, *optional*):
            Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors for
            more detail.
        return_dict (`bool`, *optional*):
            Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
"""


@add_start_docstrings(
    "The bare LLaDA2Moe Model outputting raw hidden-states without any specific head on top.",
    LLADA2MOE_START_DOCSTRING,
)
class LLaDA2MoeModel(LLaDA2MoePreTrainedModel):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LLaDA2MoeDecoderLayer`]
    Args:
        config: LLaDA2MoeConfig
    """

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        
        self.num_codebooks = config.num_codebooks
        if self.num_codebooks is not None and self.num_codebooks > 1:
            self.img_embeddings = nn.ModuleList([
                nn.Embedding(5010, config.hidden_size, None)
                for _ in range(self.num_codebooks)
            ])
            self.word_embeddings = nn.Embedding(157184, config.hidden_size, self.padding_idx)
            self.img_embeddings_proj = nn.Linear(config.hidden_size * self.num_codebooks, config.hidden_size, bias=False)
            self.use_multi_codebook = True
        else:
            self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
            self.use_multi_codebook = False
        
        self.layers = nn.ModuleList(
            [LLaDA2MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self._use_sdpa = config._attn_implementation == "sdpa"
        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        self.norm = LLaDA2MoeRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LLaDA2MoeRotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.word_embeddings

    def set_input_embeddings(self, value):
        self.word_embeddings = value
    
    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        type_position_ids: Optional[torch.LongTensor] = None,
        type: Optional[str] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, MoeModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if input_ids is None:
            if inputs_embeds is None:
                raise ValueError("input_ids or inputs_embeds is required")
            if inputs_embeds.dim() != 3:
                raise ValueError(
                    "inputs_embeds must have shape [B, S, H], "
                    f"got {tuple(inputs_embeds.shape)}"
                )
            batch_size, seq_length = inputs_embeds.shape[:2]
            num_codebooks = 1
            embedding_device = inputs_embeds.device
        else:
            if input_ids.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                input_ids = input_ids.long()
            if input_ids.dim() == 2:
                # Hugging Face's conventional [batch, sequence] text input has one
                # token stream. Internal multimodal callers provide [batch, seq, C].
                input_ids = input_ids.unsqueeze(-1)
            elif input_ids.dim() != 3:
                raise ValueError(f"input_ids must have shape [B, S] or [B, S, C], got {tuple(input_ids.shape)}")
            batch_size, seq_length, num_codebooks = input_ids.shape
            embedding_device = input_ids.device

        if type_position_ids is None:
            type_position_ids = torch.ones(
                (batch_size, seq_length), dtype=torch.long, device=embedding_device
            )
        elif type_position_ids.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
            type_position_ids = type_position_ids.long()
        if type_position_ids.dim() == 1:
            type_position_ids = type_position_ids.unsqueeze(0)
        elif type_position_ids.dim() != 2:
            raise ValueError(
                "type_position_ids must have shape [B, S] or [S], "
                f"got {tuple(type_position_ids.shape)}"
            )
        if type_position_ids.shape != (batch_size, seq_length):
            raise ValueError(
                "type_position_ids shape must match the batch/sequence dimensions: "
                f"expected {(batch_size, seq_length)}, got {tuple(type_position_ids.shape)}"
            )
        type_position_ids = type_position_ids.to(embedding_device)
        if attention_mask is not None and attention_mask.dim() == 1:
            # Packing collator may provide a 1D attention_mask: [seq_length].
            attention_mask = attention_mask.unsqueeze(0)

        if input_ids is not None:
            input_ids = input_ids.long()
            image_mask = (type_position_ids == 2) | (type_position_ids == 4)
            flat_input = input_ids.reshape(-1, num_codebooks)
            flat_image_mask = image_mask.reshape(-1)
            flat_text_mask = ~flat_image_mask
            flat_embeds = torch.empty(
                batch_size * seq_length,
                self.config.hidden_size,
                device=embedding_device,
                dtype=self.word_embeddings.weight.dtype,
            )
            if flat_text_mask.any():
                flat_embeds[flat_text_mask] = self.word_embeddings(flat_input[flat_text_mask, 0])
            if flat_image_mask.any():
                img_tokens = flat_input[flat_image_mask]
                img_embeds = torch.stack(
                    [self.img_embeddings[i](img_tokens[:, i]) for i in range(num_codebooks)],
                    dim=1,
                )
                flat_embeds[flat_image_mask] = self.img_embeddings_proj(
                    img_embeds.reshape(img_embeds.size(0), -1)
                )
            inputs_embeds = flat_embeds.reshape(batch_size, seq_length, -1)




        # inputs_embeds = flat_embeds.view(batch_size, seq_length, -1)
        batch_size, seq_length = inputs_embeds.shape[:2]
        if attention_mask is not None and attention_mask.dim() == 3:
            attention_mask = attention_mask[:, :, 0]
        if self.gradient_checkpointing and self.training:
            if use_cache:
                logger.warning_once(
                    "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`transformers."
                )
                use_cache = False

        past_key_values_length = 0
        if use_cache:
            use_legacy_cache = not isinstance(past_key_values, Cache)
            if use_legacy_cache:
                past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            past_key_values_length = past_key_values.get_usable_length(seq_length)

        if position_ids is None:
            device = inputs_embeds.device
            # position_ids 始终是 (B, seq_len)，因为位置是空间位置，不是码本维度
            position_ids = torch.arange(
                past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
            )
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        else:
            # Packing collator concatenates position_ids into a 1D tensor [S]; rotary expects [B, S].
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
            if position_ids.size(-1) != seq_length:
                if position_ids.size(-1) < seq_length:
                    pad_len = seq_length - position_ids.size(-1)
                    pad = torch.arange(pad_len, dtype=position_ids.dtype, device=position_ids.device).unsqueeze(0).expand(position_ids.size(0), -1)
                    position_ids = torch.cat([position_ids, pad], dim=-1)
                else:
                    position_ids = position_ids[..., :seq_length]
        
                

        cu_seqlens = kwargs.get("cu_seqlens", None)
        if past_key_values_length == 0 and attention_mask is not None and cu_seqlens is not None and cu_seqlens.numel() > 1:
            # attention_mask: [B, S] (token_valid: 1 for valid, 0 for padding)
            if attention_mask.dim() == 1:
                attention_mask = attention_mask.unsqueeze(0)

            token_valid = attention_mask > 0  # [B, S]
            cu_seqlens = cu_seqlens.to(device=attention_mask.device)
            seg_lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.long)  # [num_seg]
            num_seg = seg_lens.numel()

            seg_ids = torch.repeat_interleave(
                torch.arange(num_seg, device=attention_mask.device, dtype=torch.long), seg_lens
            )  # [S]

            # Guard against possible shape mismatches (e.g. SP padding).
            if seg_ids.numel() != seq_length:
                if seg_ids.numel() < seq_length:
                    pad_len = seq_length - seg_ids.numel()
                    pad_val = num_seg - 1 if num_seg > 0 else 0
                    seg_ids = torch.cat(
                        [
                            seg_ids,
                            torch.full((pad_len,), pad_val, device=seg_ids.device, dtype=seg_ids.dtype),
                        ],
                        dim=0,
                    )
                else:
                    seg_ids = seg_ids[:seq_length]

            # same_seg: [S, S]
            same_seg = seg_ids.unsqueeze(-1) == seg_ids.unsqueeze(-2)
            # Expand to batch: [B, S, S]
            same_seg = same_seg.unsqueeze(0).expand(batch_size, -1, -1)

            # Also mask out padding tokens.
            same_seg = same_seg & token_valid.unsqueeze(-1) & token_valid.unsqueeze(-2)

            # Block-diffusion-style multi-turn mask using turn_ids + is_clean.
            #   SFT samples are packed as [clean prefix/user turns + noisy answer turns | clean answer memory].
            #     - noisy text q attends to same noisy block and clean previous turns / prior clean blocks
            #     - noisy visual q attends to same noisy visual turn and clean previous turns
            #     - clean q attends only to clean kv of previous turns or valid same-turn blocks
            #     - clean q never attends to noisy kv
            turn_ids = kwargs.get("turn_ids", None)
            is_clean = kwargs.get("is_clean", None)
            block_ids = kwargs.get("block_ids", None)

            def _align(t, S, dtype):
                if t is None:
                    return None
                if t.dim() == 1:
                    t = t.unsqueeze(0)
                if t.size(-1) != S:
                    if t.size(-1) < S:
                        pad = torch.zeros((t.size(0), S - t.size(-1)), device=t.device, dtype=t.dtype)
                        t = torch.cat([t, pad], dim=-1)
                    else:
                        t = t[..., :S]
                return t.to(dtype=dtype)

            turn_ids = _align(turn_ids, seq_length, torch.long)
            is_clean = _align(is_clean, seq_length, torch.long)
            block_ids = _align(block_ids, seq_length, torch.long)
            tp_mask = _align(type_position_ids, seq_length, torch.long)

            if turn_ids is not None or is_clean is not None:
                if turn_ids is None:
                    turn_ids = torch.zeros((batch_size, seq_length), device=attention_mask.device, dtype=torch.long)
                if is_clean is None:
                    is_clean = torch.zeros((batch_size, seq_length), device=attention_mask.device, dtype=torch.long)
                clean_q = is_clean.unsqueeze(-1).bool()      # [B, S, 1]
                clean_kv = is_clean.unsqueeze(-2).bool()     # [B, 1, S]
                turn_q = turn_ids.unsqueeze(-1)              # [B, S, 1]
                turn_kv = turn_ids.unsqueeze(-2)             # [B, 1, S]

                use_sft_block_mask = (
                    block_ids is not None
                    and tp_mask is not None
                    and (block_ids >= 0).any()
                )
                if use_sft_block_mask:
                    block_q = block_ids.unsqueeze(-1)
                    block_kv = block_ids.unsqueeze(-2)
                    type_q = tp_mask.unsqueeze(-1)
                    type_kv = tp_mask.unsqueeze(-2)

                    same_turn = turn_q == turn_kv
                    prev_turn = turn_kv < turn_q
                    is_text_q = type_q == 3
                    is_text_kv = type_kv == 3
                    is_visual_q = type_q == 4
                    is_special_kv = type_kv == 1
                    is_non_text_kv = type_kv != 3

                    # Text generation: same block is bidirectional on xt;
                    # x0 side is visible only for previous turns or prior text blocks.
                    m_text_nn = (~clean_q) & is_text_q & (~clean_kv) & same_turn & (
                        (is_text_kv & (block_q == block_kv))
                        | (is_special_kv & (block_kv <= block_q))
                    )
                    m_text_nc_prev = (~clean_q) & is_text_q & clean_kv & prev_turn
                    m_text_nc_same_prior = (~clean_q) & is_text_q & clean_kv & same_turn & (
                        is_text_kv & (block_kv < block_q)
                    )

                    # Image generation remains fully bidirectional inside the noisy image turn.
                    m_visual_nn = (~clean_q) & is_visual_q & (~clean_kv) & same_turn
                    m_visual_nc = (~clean_q) & is_visual_q & clean_kv & prev_turn

                    # Special/non-target noisy q positions have no labels. Keep them from
                    # reading same-turn text, otherwise their kv could leak clean/current text.
                    m_other_noisy = (~clean_q) & (~is_text_q) & (~is_visual_q) & (
                        ((~clean_kv) & same_turn & is_non_text_kv)
                        | (clean_kv & prev_turn)
                    )

                    m_clean_text = clean_q & is_text_q & clean_kv & (
                        prev_turn
                        | (same_turn & (
                            (is_text_kv & (block_kv <= block_q))
                            | (is_special_kv & (block_kv <= block_q))
                        ))
                    )
                    m_clean_other = clean_q & (~is_text_q) & clean_kv & (
                        prev_turn | (same_turn & is_non_text_kv)
                    )

                    bd_allowed = (
                        m_text_nn | m_text_nc_prev | m_text_nc_same_prior
                        | m_visual_nn | m_visual_nc
                        | m_other_noisy
                        | m_clean_text | m_clean_other
                    )
                else:
                    m_nn = (~clean_q) & (~clean_kv) & (turn_q == turn_kv)
                    m_nc = (~clean_q) & (clean_kv) & (turn_kv < turn_q)
                    m_cc = (clean_q) & (clean_kv) & (turn_kv <= turn_q)
                    bd_allowed = m_nn | m_nc | m_cc
                same_seg = same_seg & bd_allowed

            neg_inf = torch.finfo(inputs_embeds.dtype).min
            attention_mask_4d = torch.zeros(
                (batch_size, 1, seq_length, seq_length),
                device=inputs_embeds.device,
                dtype=inputs_embeds.dtype,
            )
            attention_mask_4d.masked_fill_(~same_seg.unsqueeze(1), neg_inf)
            attention_mask = attention_mask_4d

            # Optional: verify mask semantics on a couple sampled indices.
            # This is cheap (O(1)) and only runs when ENABLE explicitly.
            if os.environ.get("VERIFY_PACK_MASK", "0") == "1" and batch_size == 1:
                with torch.no_grad():
                    i = 0
                    seg0 = seg_ids[i].item()
                    other = (seg_ids != seg0).nonzero(as_tuple=False)
                    if other.numel() > 0:
                        j = other[0].item()
                        v_ij = attention_mask_4d[0, 0, i, j].item()
                    else:
                        v_ij = None
                    v_ii = attention_mask_4d[0, 0, i, i].item()
                    print(
                        f"[VERIFY_PACK_MASK] seg0={seg0} v(ii)={v_ii} v(i,other)={v_ij} "
                        f"neg_inf={float(neg_inf)}"
                    )
        if self._use_sdpa and not output_attentions:
            # output_attentions=True can not be supported when using SDPA, and we fall back on
            # the manual implementation that requires a 4D causal mask in all cases.
            if attention_mask is None or attention_mask.dim() != 4:
                attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
                    attention_mask,
                    (batch_size, seq_length),
                    inputs_embeds,
                    past_key_values_length,
                )
        else:
            # 4d mask is passed through the layers
            if attention_mask is None or attention_mask.dim() != 4:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
                )
        # embed positions
        hidden_states = inputs_embeds

        # Segment bounds for the optional per-segment attention path. Resolved once here
        # (a `.tolist()` inside the layer loop would sync the device on every layer) and
        # only when the flag is on, so the default path is untouched.
        seg_bounds = None
        if getattr(self.config, "use_segmented_attention", False):
            seg_bounds = resolve_seg_bounds(kwargs.get("cu_seqlens", None), seq_length)
            if seg_bounds is not None and SEG_ATTN_STRICT:
                check_seg_tail_is_padding(attention_mask, seg_bounds)

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_router_logits = () if output_router_logits else None
        next_decoder_cache = None
        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    attention_mask,
                    position_ids,
                    past_key_values,
                    output_attentions,
                    output_router_logits,
                    use_cache,
                    position_embeddings,
                    seg_bounds,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    output_router_logits=output_router_logits,
                    use_cache=use_cache,
                    seg_bounds=seg_bounds,
                    position_embeddings=position_embeddings,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache = layer_outputs[2 if output_attentions else 1]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

            if output_router_logits and layer_outputs[-1] is not None:
                all_router_logits += (layer_outputs[-1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = None
        if use_cache:
            next_cache = next_decoder_cache.to_legacy_cache() if use_legacy_cache else next_decoder_cache
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_router_logits]
                if v is not None
            )
        
        if output_router_logits:
            return hidden_states, inputs_embeds, all_router_logits
        return hidden_states, inputs_embeds
        # return MoeModelOutputWithPast(
        #     last_hidden_state=hidden_states,
        #     past_key_values=next_cache,
        #     hidden_states=all_hidden_states,
        #     attentions=all_self_attns,
        #     router_logits=all_router_logits,
        # )



class MultiCodebookARHead(nn.Module):
    def __init__(
        self,
        hidden_size,          # = config.hidden_size
        num_codebooks,        # C, e.g. 8
        channel_embed_dim=None,
        channel_depth=4,
        channel_heads=8,
        mlp_ratio=4.0
    ):
        super().__init__()
        if channel_embed_dim is None:
            channel_embed_dim = hidden_size

        self.num_codebooks = num_codebooks

        # 条件投影：hidden_states -> 通道空间
        self.condition_proj = nn.Linear(hidden_size, channel_embed_dim)

        # 通道位置 & timestep embedding（跟 TokenBridge 类似）
        self.channel_embed = nn.Parameter(
            torch.zeros(1, num_codebooks, channel_embed_dim)
        )
        self.timesteps_embeddings = nn.Parameter(
            torch.zeros(1, num_codebooks, channel_embed_dim)
        )


        # 通道级 transformer blocks（自回归掩码）
        channel_mask = torch.full(
            (num_codebooks, num_codebooks),
            float("-inf"),
            dtype=torch.bfloat16,
        )
        channel_mask.triu_(1)
        self.register_buffer("channel_mask", channel_mask)

        self.channel_blocks = nn.ModuleList([
            CausalBlock(
                dim=channel_embed_dim,
                num_heads=channel_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=True,
                qk_norm=True,
                proj_drop=0.0,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            for _ in range(channel_depth)
        ])

        self.channel_norm = nn.LayerNorm(channel_embed_dim, eps=1e-6)
        self.channel_final = FinalLayer(channel_embed_dim, norm_layer=partial(nn.LayerNorm, eps=1e-6))


        # 初始化（与 TokenBridge 一致风格，但都是重新学习）
        self._init_own_weights()

    def _init_own_weights(self):
        """Initialize weights for this module's own parameters."""
        nn.init.normal_(self.channel_embed, std=0.02)
        nn.init.normal_(self.timesteps_embeddings, std=0.02)
        # for emb in self.token_embeddings:
        #     nn.init.normal_(emb.weight, std=0.02)
        for block in self.channel_blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.channel_final.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.channel_final.adaLN_modulation[-1].bias, 0)

    def _init_weights(self, module):
        """
        Initialize weights for submodules when called via module.apply().
        This method is called by the weight loading system to initialize missing parameters.
        """
        # Initialize Linear layers
        if isinstance(module, nn.Linear):
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
            nn.init.normal_(module.weight, std=0.02)
        # Initialize Embedding layers
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        # Initialize LayerNorm layers
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0)
            nn.init.constant_(module.weight, 1.0)

    def forward(self, hidden_states, token_embs, temperature=1.0):
        """
        hidden_states: [num_valid, H]       (flatten 后有效位置的 hidden)
        token_embs:    [num_valid, C-1, H]  (对应有效位置的 codebook token embedding)
        """

        # cond: [num_valid, channel_embed_dim]
        cond = self.condition_proj(hidden_states)  # [num_valid, 2048]



        x = torch.cat([cond.unsqueeze(1), token_embs], dim=1)  # [num_valid, 8, D]

        # 加通道 & timestep embedding
        x = x + self.channel_embed
        cond_seq = cond.unsqueeze(1) + self.timesteps_embeddings[:, :cond.shape[1]]

        # 通道自回归 attention（用 channel_mask）
        for block in self.channel_blocks:
            self.channel_mask = self.channel_mask.to(x.dtype)
            if self.training:
                x = torch.utils.checkpoint.checkpoint(block,x,attn_mask=self.channel_mask, c=cond_seq, use_reentrant=False)
            else:
                x = block(x, attn_mask=self.channel_mask, c=cond_seq)



        x = self.channel_norm(x)
        x = self.channel_final(x, cond_seq)   # [num_valid, C, D]

        return x


class LLaDA2MoeModelLM(LLaDA2MoePreTrainedModel, GenerationMixin):
    # _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config: LLaDA2MoeConfig):
        super().__init__(config)
        self.model = LLaDA2MoeModel(config)
        self.vocab_size = config.vocab_size

        ## AR modules
        self.num_codebooks = getattr(config, "num_codebooks", 8)
        self.codebook_size = getattr(config, "codebook_size", 4096)
        self.lm_img_head = nn.ModuleList([
                nn.Linear(config.hidden_size, 4096, bias=False)
                for _ in range(self.num_codebooks)
            ])
        self.lm_head = nn.Linear(config.hidden_size, 157184, bias=False)
        # self.codebook_proj = nn.Linear(config.hidden_size, 8*config.hidden_size, bias=False)

        self.multi_codebook_head = MultiCodebookARHead(
            hidden_size=config.hidden_size,
            num_codebooks=self.num_codebooks,
            channel_embed_dim=config.hidden_size,
            channel_depth=6,
            channel_heads=8,
            mlp_ratio=4.0
        )
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.word_embeddings

    def set_input_embeddings(self, value):
        self.model.word_embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def _conditional_modules(self) -> List[nn.Module]:
        """Modules whose participation in the graph depends on what the batch holds.

        The image heads run only under `if label_mask.any()` and the text head only
        when the pack has supervised text, so a rank whose pack happens to carry
        none of one kind leaves those parameters out of the autograd graph. FSDP2
        then issues one fewer gradient collective on that rank than on its peers,
        the shard group desyncs, and the job hangs until the NCCL watchdog fires --
        with every other rank reporting the timeout and the culprit reporting
        nothing. Both forward paths must therefore pin the same set.
        """
        return (
            list(self.lm_img_head)                    # img answer head
            + [self.lm_head]                          # text answer head
            + [self.multi_codebook_head]              # img codebook head
            + list(self.model.img_embeddings)          # img input embeddings
            + [self.model.img_embeddings_proj]         # img input projection
            + [self.model.word_embeddings]             # text input embeddings
        )

    def _make_head_dummy(self) -> torch.Tensor:
        """A zero scalar that every conditional parameter contributes to."""
        return sum(
            p.float().sum() for m in self._conditional_modules() for p in m.parameters()
        ) * 0


    @add_start_docstrings_to_model_forward(LLADA2MOE_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=MoeCausalLMOutputWithPast, config_class=_CONFIG_FOR_DOC)
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        type_position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        type: Optional[str] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_router_logits: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        return_hidden_states_for_loss: Optional[bool] = False,
        infer_t2i: Optional[bool] = False,
        infer_mmu: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, MoeCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer
        >>> model = LLaDA2MoeModelLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
        >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)
        >>> # Use infer_1024_sft.sh or a task-specific infer_*_sft.py entrypoint
        >>> # for modality-aware block-diffusion generation.
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_router_logits = (
            output_router_logits if output_router_logits is not None else self.config.output_router_logits
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        # Public/Hugging Face text callers do not provide modality metadata.  Use
        # the ordinary text type for those calls so the custom heads and
        # ``generate`` helper do not dereference None.
        if type_position_ids is None:
            shape_source = input_ids if input_ids is not None else inputs_embeds
            if shape_source is None:
                raise ValueError("input_ids or inputs_embeds is required")
            type_position_ids = torch.ones(
                shape_source.shape[:2], dtype=torch.long, device=shape_source.device
            )
        elif type_position_ids.dim() == 1:
            type_position_ids = type_position_ids.unsqueeze(0)

        outputs = self.model(
            type=type,
            input_ids=input_ids,
            type_position_ids=type_position_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            output_router_logits=output_router_logits,
            return_dict=return_dict,
            **kwargs,
        )

        if output_router_logits:
            hidden_states, inputs_embeds, all_router_logits = outputs
        else:
            hidden_states, inputs_embeds = outputs

        loss = None
        aux_loss = None

        batch_size, seq_length, hidden_size = hidden_states.shape
        hidden_states_flat = hidden_states.reshape(-1, hidden_size)

        type_position_ids = type_position_ids.to(hidden_states.device)
        type_position_ids_flat = type_position_ids.reshape(-1)

        # Image answer branch (type_position_ids == 4)
        img_ans_mask = type_position_ids_flat == 4

        hidden_states_img = hidden_states_flat[img_ans_mask]

        if infer_t2i:
            return hidden_states_img.unsqueeze(0)

            # Default empty logits when no valid positions, so return never raises NameError
        logits_img = torch.empty(
            (0, self.num_codebooks, self.codebook_size),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        logits_text = torch.empty(
            (0, self.lm_head.out_features),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        if labels is not None:
            # FSDP2 mixed precision may cast kwargs to bf16; embedding / CE need integer labels.
            if labels.dtype not in (torch.long, torch.int, torch.int32, torch.int64):
                labels = labels.long()
            img_labels = labels.to(hidden_states.device)
            # Packing (multi-codebook) 时，labels 通常是 [seq_length, num_codebooks] 或 [B, seq_length, num_codebooks]。
            # `img_ans_mask` 是按“位置 token”展开后的 mask（长度为 B*seq_length），因此不能直接用它去索引展平到 num_codebooks 维度后的张量。
            if img_labels.dim() == 2:
                if img_labels.size(-1) == self.num_codebooks:
                    img_labels = img_labels.unsqueeze(0)  # -> [1, seq_length, num_codebooks]
                else:
                    img_labels = img_labels.unsqueeze(-1)  # legacy: [B, seq_length] -> [B, seq_length, 1]

            # Now img_labels should be [B, seq_length, C]
            img_labels_flat = img_labels.reshape(-1, img_labels.size(-1))  # [B*seq_length, C]
            masked_img_labels = img_labels_flat[img_ans_mask]  # [num_valid, C]
            label_mask = (masked_img_labels != -100).any(dim=-1)

            if label_mask.any():
                hidden_states_img = hidden_states_img[label_mask]
                valid_img_labels = masked_img_labels[label_mask]

                if (valid_img_labels < 0).any():
                    raise ValueError("valid image labels must be non-negative")
                if not torch.isfinite(hidden_states_img).all():
                    raise FloatingPointError("non-finite transformer states in image branch")

                img_emb = torch.stack(
                    [self.model.img_embeddings[i](valid_img_labels[:, i]) for i in range(self.num_codebooks)],
                    dim=1,
                )
                # drop the last codebook token as before

                if not torch.isfinite(img_emb).all():
                    raise FloatingPointError("non-finite image-token embeddings")
                img_emb = img_emb[:, :-1, :]

                mc_emb = self.multi_codebook_head(
                    hidden_states_img,   # [num_valid, hidden]
                    img_emb,             # [num_valid, 7, hidden]
                    temperature=1.0,
                )
                if not torch.isfinite(mc_emb).all():
                    raise FloatingPointError("non-finite multi-codebook hidden states")
                logits_img = torch.cat(
                    [self.lm_img_head[i](mc_emb[:, i]).unsqueeze(1) for i in range(self.num_codebooks)],
                    dim=1,
                )



        if not torch.isfinite(logits_img).all():
            raise FloatingPointError("non-finite image logits")

        # Text answer branch (type_position_ids == 3)
        text_ans_mask = type_position_ids_flat == 3
        if infer_mmu:
            infer_text_mask = text_ans_mask
            infer_is_clean = kwargs.get("is_clean", None)
            if infer_is_clean is not None:
                if infer_is_clean.dim() == 1:
                    infer_is_clean = infer_is_clean.unsqueeze(0)
                infer_is_clean_flat = infer_is_clean.to(hidden_states.device).reshape(-1)
                if infer_is_clean_flat.numel() == infer_text_mask.numel():
                    infer_text_mask = infer_text_mask & (infer_is_clean_flat == 0)
            hidden_states_text = hidden_states_flat[infer_text_mask]
            logits_text = self.lm_head(hidden_states_text)
            return logits_text

        # Keep the conventional Transformers call usable for text-only callers.
        # Multimodal decoding should use infer_t2i/infer_mmu, which return the
        # modality-specific logits expected by the public generators.
        if labels is None:
            logits = self.lm_head(hidden_states)
            if return_dict:
                return MoeCausalLMOutputWithPast(logits=logits)
            return (logits,)

        text_labels = labels.to(hidden_states.device)
        # Packing 多码本时：labels 常见形状为 [seq_length, num_codebooks]（或 [B, seq_length, num_codebooks]）
        # 这里文本分支最终只需要“每个位置一个 label”，因此固定取第 0 个码本的 token id。
        if text_labels.dim() == 2:
            if text_labels.size(-1) == self.num_codebooks:
                text_labels = text_labels.unsqueeze(0)  # -> [1, seq_length, num_codebooks]
                text_labels = text_labels[:, :, 0]  # -> [1, seq_length]
        elif text_labels.dim() == 3:
            text_labels = text_labels[:, :, 0]  # -> [B, seq_length]

        text_labels_flat = text_labels.reshape(-1)  # [B*seq_length]
        hidden_states_text = hidden_states_flat[text_ans_mask]
        text_labels_masked = text_labels_flat[text_ans_mask]


        label_mask_text = text_labels_masked != -100

        hidden_states_text = hidden_states_text[label_mask_text]
        logits_text = self.lm_head(hidden_states_text)


        # text_token = logits_text.argmax(dim=-1) ######

        if not return_dict:
            output = ((logits_img, logits_text),) + outputs[1:]
            if output_router_logits:
                output = (aux_loss,) + output
            return (loss,) + output if loss is not None else output

        if output_router_logits:
            if return_hidden_states_for_loss:
                # Widened training contract: `None` fills the hidden-states slot and the
                # head dummy pins every conditional head into the autograd graph so FSDP2
                # gradient collectives stay in lockstep across ranks (see _make_head_dummy).
                return logits_img, logits_text, all_router_logits, None, self._make_head_dummy()
            return logits_img, logits_text, all_router_logits
        return logits_img, logits_text

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, token_type_ids=None, **kwargs
    ):
        if past_key_values is not None:
            if isinstance(past_key_values, Cache):
                cache_length = past_key_values.get_seq_length()
                past_length = past_key_values.seen_tokens
                max_cache_length = (
                    past_key_values.get_max_length()
                    if hasattr(past_key_values, "get_max_length")
                    else past_key_values.get_max_cache_shape()
                )
            else:
                cache_length = past_length = past_key_values[0][0].shape[2]
                max_cache_length = None

            # Keep only the unprocessed tokens:
            # 1 - If the length of the attention_mask exceeds the length of input_ids, then we are in a setting where
            # some of the inputs are exclusivelly passed as part of the cache (e.g. when passing input_embeds as input)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                input_ids = input_ids[:, -(attention_mask.shape[1] - past_length) :]
            # 2 - If the past_length is smaller than input_ids', then input_ids holds all input tokens. We can discard
            # input_ids based on the past_length.
            elif past_length < input_ids.shape[1]:
                input_ids = input_ids[:, past_length:]
            # 3 - Otherwise (past_length >= input_ids.shape[1]), let's assume input_ids only has unprocessed tokens.

            # If we are about to go beyond the maximum cache length, we need to crop the input attention mask.
            if (
                max_cache_length is not None
                and attention_mask is not None
                and cache_length + input_ids.shape[1] > max_cache_length
            ):
                attention_mask = attention_mask[:, -max_cache_length:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs

    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx.to(past_state.device)) for past_state in layer_past),
            )
        return reordered_past

    @staticmethod
    def _top_k_logits(logits, k):
        if k is None or k <= 0:
            return logits
        else:
            values, _ = torch.topk(logits, k)
            min_values = values[..., -1, None]
            return torch.where(
                logits < min_values, torch.full_like(logits, float("-inf")), logits
            )

    @staticmethod
    def _top_p_logits(logits, p):
        if p is None or p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        mask_indices = torch.scatter(
            torch.full_like(logits, False, dtype=torch.bool),
            -1,
            sorted_indices,
            sorted_mask,
        )
        return logits.masked_fill(mask_indices, float("-inf"))

    def _sample_with_temperature_topk_topp(self, logits, temperature=1.0, top_k=0, top_p=1.0):
        orig_shape = logits.shape[:-1]
        vocab_size = logits.shape[-1]
        logits = logits.reshape(-1, vocab_size)
        if temperature == 0:
            token = logits.argmax(dim=-1, keepdim=True)
            probs = F.softmax(logits, dim=-1)
        else:
            logits = logits / temperature
            logits = self._top_k_logits(logits, top_k)
            logits = self._top_p_logits(logits, top_p)
            probs = F.softmax(logits, dim=-1)
            token = torch.multinomial(probs, num_samples=1)
        token_prob = torch.gather(probs, -1, token)
        return token.view(*orig_shape), token_prob.view(*orig_shape)

    @staticmethod
    def _get_num_transfer_tokens(block_length, steps):
        if steps == 0:
            return torch.tensor([], dtype=torch.int64)
        base = block_length // steps
        remainder = block_length % steps
        num_transfer_tokens = torch.full((steps,), base, dtype=torch.int64)
        num_transfer_tokens[:remainder] += 1
        return num_transfer_tokens

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        temperature: float = 0.0,
        block_length: int = 32,
        steps: int = 32,
        gen_length: int = 2048,
        top_p: Optional[int] = None,
        top_k: Optional[int] = None,
        eos_early_stop: bool = False,
        minimal_topk: int = 1,
        threshold: float = 0.95,
        eos_id: int = 157166,
        mask_id: int = 156895,
    ):
        r"""
        Generates tokens using a block-wise, iterative refinement strategy.
        This method operates differently from standard autoregressive generation. It first creates a template of the
        full desired length, filled with a special `mask_id`. It then processes this template in segments (`blocks`)
        and iteratively "denoises" or "refines" the `mask_id` tokens into actual tokens over a series of `steps` for
        each block. A custom block-diagonal causal attention mask ensures that generation within a block can attend to
        all previous blocks but not future ones.
        <Tip warning={true}>
        This is a specialized generation method. The quality and speed of the output are highly dependent on the interplay
        between `block_length`, `steps`, and `threshold`. It aims to achieve faster generation through parallel
        decoding within blocks, which is a departure from the token-by-token generation of standard `.generate()` methods.
        </Tip>
        Parameters:
            inputs (`torch.Tensor`):
                The token sequence used as a prompt for the generation.
            temperature (`float`, *optional*, defaults to 0.0):
                The value used to module the next token probabilities. A value of 0.0 corresponds to greedy decoding.
            block_length (`int`, *optional*, defaults to 32):
                The size of each generation block. The model generates text in parallel within these blocks. This is a
                key parameter for controlling the granularity of the generation process.
            steps (`int`, *optional*, defaults to 32):
                The number of iterative refinement (or "denoising") steps to perform for each block. Within each block,
                the model will try to replace `mask_id` tokens with real tokens for this many iterations.
            gen_length (`int`, *optional*, defaults to 2048):
                The maximum number of tokens to generate, excluding the prompt.
            top_p (`float`, *optional*):
                If set to a float value between 0 and 1, only the most probable tokens with probabilities that add up to
                `top_p` or higher are kept for generation (nucleus sampling).
            top_k (`int`, *optional*):
                The number of highest probability vocabulary tokens to keep for top-k-filtering.
            eos_early_stop (`bool`, *optional*, defaults to `False`):
                If `True`, generation will stop as soon as a valid End-Of-Sequence token is generated and confirmed,
                even if `gen_length` has not been reached.
            minimal_topk (`int`, *optional*, defaults to 1):
                A parameter used to dynamically adjust the number of refinement `steps`. The effective number of steps
                is capped at `gen_length // minimal_topk`.
            threshold (`float`, *optional*, defaults to 0.95):
                The confidence probability threshold for accepting a sampled token. During each refinement step, a
                sampled token is only kept if its probability is above this threshold. If not enough tokens meet the
                threshold, the ones with the highest confidence are chosen.
            eos_id (`int`, *optional*, defaults to 157166):
                The token ID for the end-of-sequence token. Used for `eos_early_stop`.
            mask_id (`int`, *optional*, defaults to 156895):
                The token ID used as a placeholder for tokens that are yet to be generated. This is central to the
                iterative refinement algorithm.
        Return:
            `torch.Tensor`: A string containing the generated token IDs, starting
            after the prompt and stopping at the first `eos_id` or `gen_length`.
        """
        steps = min(steps, gen_length // minimal_topk)
        input_ids = inputs.to(self.device)

        prompt_length = input_ids.shape[1]
        num_blocks = (prompt_length + gen_length + block_length - 1) // block_length
        total_length = num_blocks * block_length

        block_mask = torch.tril(torch.ones(num_blocks, num_blocks, device=self.device))
        block_diffusion_attention_mask = (
            block_mask.repeat_interleave(block_length, dim=0)
            .repeat_interleave(block_length, dim=1)
            .unsqueeze(0)
            .unsqueeze(0)
        ).bool()
        block_diffusion_attention_mask = torch.where(
            block_diffusion_attention_mask, 0.0, float("-inf")
        ).to(torch.bfloat16)

        position_ids = torch.arange(total_length, device=self.device).unsqueeze(0)
        x = torch.full((1, total_length), mask_id, dtype=torch.long, device=self.device)
        x[:, :prompt_length] = input_ids.clone()

        prompt_index_full = torch.zeros_like(x, dtype=torch.bool)
        prompt_index_full[:, :prompt_length] = True

        prefill_blocks = prompt_length // block_length

        denoising_steps_per_block = steps
        num_transfer_tokens_schedule = self._get_num_transfer_tokens(
            block_length, denoising_steps_per_block
        )
        for num_block in range(prefill_blocks, num_blocks):
            current_window_end = (num_block + 1) * block_length
            cur_x = x[:, :current_window_end]
            cur_attn_mask = block_diffusion_attention_mask[
                :, :, :current_window_end, :current_window_end
            ]
            cur_position_ids = position_ids[:, :current_window_end]

            for step in range(denoising_steps_per_block):
                active_block_mask = cur_x[:, -block_length:] == mask_id
                if active_block_mask.sum() == 0:
                    break

                logits = self.forward(
                    cur_x,
                    attention_mask=cur_attn_mask,
                    position_ids=cur_position_ids,
                ).logits

                active_logits = logits[:, -block_length:, :]
                x0, x0_p = self._sample_with_temperature_topk_topp(
                    active_logits, temperature=temperature, top_k=top_k, top_p=top_p
                )

                num_to_transfer = num_transfer_tokens_schedule[step].item()
                transfer_index = torch.zeros_like(x0, dtype=torch.bool)

                confidence = torch.where(active_block_mask, x0_p, -torch.inf)
                high_conf_mask = confidence[0] > threshold
                num_high_confidence = high_conf_mask.sum().item()

                if num_high_confidence >= num_to_transfer:
                    transfer_index[0] = high_conf_mask
                else:
                    _, idx = torch.topk(
                        confidence[0],
                        k=min(num_to_transfer, active_block_mask.sum().item()),
                    )
                    transfer_index[0, idx] = True

                if transfer_index.any():
                    cur_x[:, -block_length:][transfer_index] = x0[transfer_index]
                if eos_early_stop and (x0[transfer_index] == eos_id).any():
                    eos_pos_in_x = (cur_x[0] == eos_id).nonzero(as_tuple=True)
                    if len(eos_pos_in_x[0]) > 0:
                        eos_pos = eos_pos_in_x[0][0].item()
                        if (cur_x[0, prompt_length:eos_pos] != mask_id).all():
                            final_x = x[:, :total_length][:, : eos_pos + 1]
                            return final_x

            x[:, :current_window_end] = cur_x
            if (
                eos_id is not None
                and (x[0, prompt_length:current_window_end] == eos_id).any()
            ):
                break

        generated_answer = x[:, : prompt_length + gen_length]

        mask_positions = (generated_answer[0][input_ids.shape[1] :] == eos_id).nonzero(
            as_tuple=True
        )[0]
        if len(mask_positions) > 0:
            first_mask_position = mask_positions[0].item()
        else:
            first_mask_position = gen_length
        return generated_answer[:, input_ids.shape[1] : input_ids.shape[1] + first_mask_position + 1]


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb_llada2_moe(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_embed_rot, k_embed_rot = liger_rotary_pos_emb(q_rot, k_rot, cos, sin)

    q_embed = torch.cat([q_embed_rot, q_pass], dim=-1)
    k_embed = torch.cat([k_embed_rot, k_pass], dim=-1)
    
    return q_embed, k_embed


if is_liger_kernel_available():
    apply_rotary_pos_emb = apply_rotary_pos_emb_llada2_moe
    LLaDA2MoeRMSNorm = LigerRMSNorm
    logger.info_rank0("Apply liger kernel to LLaDA2Moe")


ModelClass = LLaDA2MoeModelLM

__all__ = [
    "LLaDA2MoeModelLM",
    "LLaDA2MoeModel",
    "LLaDA2MoePreTrainedModel"
]

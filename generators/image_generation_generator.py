# -*- coding: utf-8 -*-
"""
Image generation generator
"""
import torch
import torch.nn.functional as F
from typing import Callable, Optional
from utils.generation_utils import cosine_schedule, gumbel_max_sample, mask_by_random_topk
from .text_understanding_generator import build_t2i_attention_mask_4d
def generate_image_ar(
    model,
    input_ids: torch.LongTensor,
    uncon_input_ids: torch.LongTensor,
    type_position_ids: torch.LongTensor,
    uncon_type_position_ids: torch.LongTensor,
    *,
    timesteps: int = 64,
    mask_token_id: int = 5000,
    newline_id: int = 5001,
    temperature: float = 1.0,
    cfg_scale: float = 0.0,
    cfg_text: Optional[float] = None,
    cfg_img: Optional[float] = None,
    uncon_text_input_ids: Optional[torch.LongTensor] = None,
    uncon_image_input_ids: Optional[torch.LongTensor] = None,
    uncon_text_type_position_ids: Optional[torch.LongTensor] = None,
    uncon_image_type_position_ids: Optional[torch.LongTensor] = None,
    uncon_text_user_prefix_len: Optional[int] = None,
    uncon_image_user_prefix_len: Optional[int] = None,
    code_start: int,
    noise_schedule: Callable[[torch.Tensor], torch.Tensor] = cosine_schedule,
    generator: Optional[torch.Generator] = None,
    debug: bool = False,
    pad_len: Optional[int] = None,
    user_prefix_len: Optional[int] = None,
    media_region_start: Optional[int] = None,
    uncon_user_prefix_len: Optional[int] = None,
    uncon_media_region_start: Optional[int] = None,
    position_ids: Optional[torch.LongTensor] = None,
    uncon_position_ids: Optional[torch.LongTensor] = None,
    uncon_text_position_ids: Optional[torch.LongTensor] = None,
    uncon_image_position_ids: Optional[torch.LongTensor] = None,
) -> torch.LongTensor:
    """Iteratively generate all output-image VQ positions.

    Spatial positions use MaskGIT-style parallel refinement. Inside one
    spatial position, codebooks 0..7 are predicted sequentially, so "AR" in
    this function name refers to codebook channels rather than left-to-right
    image generation.

    Args:
        input_ids: Conditional sequence, shape [1, cond_S, 8].
        uncon_input_ids: CFG-negative sequence, shape [1, uncond_S, 8].
        type_position_ids: Cond role labels; type 4 identifies image answers.
        uncon_type_position_ids: Uncond branch role labels. Image-answer count and
            MASK pattern must match cond, but the surrounding sequence may be shorter.
        timesteps: Number of spatial MaskGIT refinement rounds.
        mask_token_id: Value marking an unknown image code.
        newline_id: Structural row separator removed from the final result.
        temperature: Gumbel sampling and confidence-remasking temperature.
        cfg_scale: Hidden-state CFG scale. The implemented formula is
            (1 + scale) * cond - scale * uncond.
        cfg_text: Edit-only text CFG scale. Must be paired with cfg_img.
        cfg_img: Edit-only image CFG scale. Must be paired with cfg_text.
        code_start: BOI sequence index; image tokens begin at code_start + 1.
        noise_schedule: Maps progress to the number of positions kept masked.
        generator: Seeded CUDA RNG used by sampling and remasking.
        user_prefix_len: End of conditional clean SFT turn 0.
        media_region_start: Conditional BOI / noisy image-turn start.
        uncon_user_prefix_len: End of unconditional clean turn 0. Together with
            uncon_media_region_start, opts into a branch-local asymmetric layout.
        uncon_media_region_start: BOI index in the uncond sequence. When omitted,
            it is derived from the conditional answer suffix.
        position_ids: Optional explicit cond RoPE positions.
            to add a virtual gap without inserting physical padding tokens.
        uncon_position_ids: Corresponding uncond RoPE positions.

        The number of generated positions is derived from type-4 IMAGE_MASK entries.

    Returns:
        Tensor [1, number_of_spatial_positions, 8], with NEW_LINE removed.
    """

    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    uncon_input_ids = uncon_input_ids.to(device)
    type_position_ids = type_position_ids.to(device)
    uncon_type_position_ids = uncon_type_position_ids.to(device)
    dual_cfg = cfg_text is not None or cfg_img is not None
    if (cfg_text is None) != (cfg_img is None):
        raise ValueError("cfg_text and cfg_img must be provided together")
    if dual_cfg and cfg_scale != 0:
        raise ValueError("non-zero cfg_scale cannot be combined with cfg_text/cfg_img")
    if dual_cfg:
        dual_inputs = (
            uncon_text_input_ids,
            uncon_image_input_ids,
            uncon_text_type_position_ids,
            uncon_image_type_position_ids,
        )
        if any(value is None for value in dual_inputs):
            raise ValueError("dual-CFG requires both negative input/type-id branches")
        uncon_text_input_ids = uncon_text_input_ids.to(device)
        uncon_image_input_ids = uncon_image_input_ids.to(device)
        uncon_text_type_position_ids = uncon_text_type_position_ids.to(device)
        uncon_image_type_position_ids = uncon_image_type_position_ids.to(device)
    if position_ids is not None:
        position_ids = position_ids.to(device)
    if uncon_position_ids is not None:
        uncon_position_ids = uncon_position_ids.to(device)
    if uncon_text_position_ids is not None:
        uncon_text_position_ids = uncon_text_position_ids.to(device)
    if uncon_image_position_ids is not None:
        uncon_image_position_ids = uncon_image_position_ids.to(device)
    if input_ids.dim() != 3:
        raise ValueError(f"img_prompt should be [B, seq_len, num_codebooks], got {input_ids.shape}")
    # This local seq_len is the complete model sequence length and overwrites
    # the legacy keyword argument. Actual image length is measured below.
    B, seq_len, num_codebooks = input_ids.shape
    assert B == 1, "batch>1 not supported – wrap in loop if needed"

    position_branches = (("cond", input_ids, type_position_ids, position_ids),)
    if dual_cfg:
        position_branches += (
            ("uncond_text", uncon_text_input_ids, uncon_text_type_position_ids, uncon_text_position_ids),
            ("uncond_image", uncon_image_input_ids, uncon_image_type_position_ids, uncon_image_position_ids),
        )
    else:
        position_branches += (("uncond", uncon_input_ids, uncon_type_position_ids, uncon_position_ids),)
    for branch_name, branch_input_ids, branch_type_ids, branch_position_ids in position_branches:
        if branch_input_ids.dim() != 3:
            raise ValueError(f"{branch_name} input_ids must be [B, S, C], got {branch_input_ids.shape}")
        if branch_type_ids.dim() != 2 or branch_type_ids.shape != branch_input_ids.shape[:2]:
            raise ValueError(
                f"{branch_name} type_position_ids shape {branch_type_ids.shape} "
                f"does not match input_ids batch/sequence shape {branch_input_ids.shape[:2]}"
            )
        if branch_position_ids is not None and branch_position_ids.shape[-1] != branch_input_ids.shape[1]:
            raise ValueError(
                f"{branch_name} position_ids length {branch_position_ids.shape[-1]} "
                f"does not match sequence length {branch_input_ids.shape[1]}"
            )

    cond_t2i_attn_mask = None
    uncond_t2i_attn_mask = None
    uncon_text_t2i_attn_mask = None
    uncon_image_t2i_attn_mask = None
    # Metadata-aware masking separates the clean user prefix from the noisy image turn.
    # A negative branch may use its own sequence length and media boundary when both
    # values are explicit; otherwise derive the boundary from the conditional suffix.
    if (user_prefix_len is None) != (media_region_start is None):
        raise ValueError("user_prefix_len and media_region_start must be provided together")
    if user_prefix_len is not None:
        model_dtype = next(model.parameters()).dtype

        def validate_attention_layout(
            branch_name,
            branch_input_ids,
            branch_prefix_len,
            branch_media_region_start,
        ):
            branch_seq_len = int(branch_input_ids.shape[1])
            branch_prefix_len = int(branch_prefix_len)
            branch_media_region_start = int(branch_media_region_start)
            if not 0 <= branch_prefix_len <= branch_media_region_start < branch_seq_len:
                raise ValueError(
                    f"{branch_name} boundaries must satisfy 0 <= user_prefix_len <= "
                    f"media_region_start < seq_len, got user_prefix_len={branch_prefix_len}, "
                    f"media_region_start={branch_media_region_start}, seq_len={branch_seq_len}"
                )
            return branch_prefix_len, branch_media_region_start

        user_prefix_len, media_region_start = validate_attention_layout(
            "cond", input_ids, user_prefix_len, media_region_start
        )
        cond_t2i_attn_mask = build_t2i_attention_mask_4d(
            type_position_ids,
            user_prefix_len,
            media_region_start,
            device,
            model_dtype,
        )
        cond_seq_len = int(input_ids.shape[1])
        answer_suffix_len = cond_seq_len - user_prefix_len
        answer_offset = media_region_start - user_prefix_len

        def build_negative_attention_mask(
            branch_name,
            branch_input_ids,
            branch_type_ids,
            branch_prefix_len,
            branch_media_region_start=None,
        ):
            branch_seq_len = int(branch_input_ids.shape[1])
            has_explicit_layout = (
                branch_prefix_len is not None and branch_media_region_start is not None
            )
            if has_explicit_layout:
                branch_prefix_len, branch_media_region_start = validate_attention_layout(
                    branch_name,
                    branch_input_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                )
            else:
                if branch_prefix_len is None:
                    branch_prefix_len = branch_seq_len - answer_suffix_len
                branch_prefix_len = int(branch_prefix_len)
                if branch_media_region_start is None:
                    branch_media_region_start = branch_prefix_len + answer_offset
                if branch_seq_len != branch_prefix_len + answer_suffix_len:
                    raise ValueError(
                        f"{branch_name} seq_len mismatch: got {branch_seq_len}, expected "
                        f"{branch_prefix_len + answer_suffix_len} "
                        f"(prefix_len={branch_prefix_len}, answer_suffix_len={answer_suffix_len}); "
                        f"provide both branch user/media boundaries for an asymmetric layout"
                    )
                branch_prefix_len, branch_media_region_start = validate_attention_layout(
                    branch_name,
                    branch_input_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                )
            return (
                build_t2i_attention_mask_4d(
                    branch_type_ids,
                    branch_prefix_len,
                    branch_media_region_start,
                    device,
                    model_dtype,
                ),
                branch_prefix_len,
                branch_media_region_start,
            )

        if dual_cfg:
            (
                uncon_text_t2i_attn_mask,
                uncon_text_user_prefix_len,
                _,
            ) = build_negative_attention_mask(
                "uncond_text",
                uncon_text_input_ids,
                uncon_text_type_position_ids,
                uncon_text_user_prefix_len,
            )
            (
                uncon_image_t2i_attn_mask,
                uncon_image_user_prefix_len,
                _,
            ) = build_negative_attention_mask(
                "uncond_image",
                uncon_image_input_ids,
                uncon_image_type_position_ids,
                uncon_image_user_prefix_len,
            )
        else:
            (
                uncond_t2i_attn_mask,
                uncon_user_prefix_len,
                uncon_media_region_start,
            ) = build_negative_attention_mask(
                "uncond",
                uncon_input_ids,
                uncon_type_position_ids,
                uncon_user_prefix_len,
                uncon_media_region_start,
            )

    same_positions = (
        (position_ids is None and uncon_position_ids is None)
        or (
            position_ids is not None
            and uncon_position_ids is not None
            and torch.equal(position_ids, uncon_position_ids)
        )
    )
    same_condition_branch = (
        not dual_cfg
        and torch.equal(input_ids, uncon_input_ids)
        and torch.equal(type_position_ids, uncon_type_position_ids)
        and same_positions
        and user_prefix_len == uncon_user_prefix_len
        and media_region_start == uncon_media_region_start
    )
    # CFG is numerically the conditional branch when scale is zero or both
    # branches are identical. Avoid an otherwise redundant backbone forward.
    single_branch_cfg = not dual_cfg and (cfg_scale == 0 or same_condition_branch)

    img_ans_mask = (type_position_ids == 4)  # [1, full_cond_sequence]
    img_ans_global_indices = torch.nonzero(img_ans_mask[0], as_tuple=False).squeeze(-1)
    if dual_cfg:
        uncon_text_img_ans_global_indices = torch.nonzero(
            (uncon_text_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        uncon_image_img_ans_global_indices = torch.nonzero(
            (uncon_image_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        branch_image_indices = [
            ("uncond_text", uncon_text_input_ids, uncon_text_img_ans_global_indices),
            ("uncond_image", uncon_image_input_ids, uncon_image_img_ans_global_indices),
        ]
    else:
        uncon_img_ans_global_indices = torch.nonzero(
            (uncon_type_position_ids == 4)[0], as_tuple=False
        ).squeeze(-1)
        branch_image_indices = [
            ("uncond", uncon_input_ids, uncon_img_ans_global_indices)
        ]
    cond_image_state = input_ids[0, img_ans_global_indices]
    for branch_name, branch_input_ids, branch_indices in branch_image_indices:
        if branch_indices.numel() != img_ans_global_indices.numel():
            raise ValueError(
                f"{branch_name} type-4 count {branch_indices.numel()} does not match "
                f"cond count {img_ans_global_indices.numel()}"
            )
        if not torch.equal(
            branch_input_ids[0, branch_indices] == mask_token_id,
            cond_image_state == mask_token_id,
        ):
            raise ValueError(f"{branch_name} initial image MASK pattern does not match cond")

    vq_mask = input_ids[img_ans_mask] == mask_token_id  # [image_turn_tokens, 8]
    # All eight codebooks at one spatial position are masked/committed together,
    # so codebook 0 is sufficient to decide whether the position is unknown.
    position_mask = vq_mask[..., 0]
    unknown_cnt = position_mask.sum(dim=0, keepdim=True)  # Dynamic MASK count.
    vq_len = unknown_cnt.clone()                         # Fixed initial count.

    debug_steps = [] if debug else None

    for step in range(timesteps):
        if unknown_cnt.item() == 0:
            break

        # keep_n is the target number of spatial positions that remain MASKed
        # after this round. The cosine schedule decreases it toward zero; the
        # final round commits every remaining position.
        if step < timesteps - 1:
            frac = noise_schedule(torch.tensor([(step + 1) / timesteps], device=device))
            keep_n = (vq_len.float() * frac).floor().clamp_min(1).long()
        else:
            keep_n = torch.zeros_like(unknown_cnt)
        with torch.inference_mode():
            # infer_t2i returns hidden states only for type-4 image-answer
            # positions. Explicit position_ids, when supplied by blkpad, are
            # consumed by model RoPE; they do not alter physical tensor length.
            cond_hidden_states = model(
                input_ids=input_ids,
                type_position_ids=type_position_ids,
                attention_mask=cond_t2i_attn_mask,
                position_ids=position_ids,
                infer_t2i=True,
                cat='cond',
                use_cache=False,
            )

            if dual_cfg:
                uncon_text_hidden_states = model(
                    input_ids=uncon_text_input_ids,
                    type_position_ids=uncon_text_type_position_ids,
                    attention_mask=uncon_text_t2i_attn_mask,
                    position_ids=uncon_text_position_ids,
                    infer_t2i=True,
                    cat='uncond_text',
                    use_cache=False,
                )
                uncon_image_hidden_states = model(
                    input_ids=uncon_image_input_ids,
                    type_position_ids=uncon_image_type_position_ids,
                    attention_mask=uncon_image_t2i_attn_mask,
                    position_ids=uncon_image_position_ids,
                    infer_t2i=True,
                    cat='uncond_image',
                    use_cache=False,
                )
                if (
                    uncon_text_hidden_states.shape != cond_hidden_states.shape
                    or uncon_image_hidden_states.shape != cond_hidden_states.shape
                ):
                    raise ValueError(
                        "dual-CFG hidden-state shapes must match: "
                        f"cond={cond_hidden_states.shape}, "
                        f"text={uncon_text_hidden_states.shape}, "
                        f"image={uncon_image_hidden_states.shape}"
                    )
                hidden_states = (
                    cond_hidden_states
                    + cfg_text * (cond_hidden_states - uncon_text_hidden_states)
                    + cfg_img * (cond_hidden_states - uncon_image_hidden_states)
                )
            elif not single_branch_cfg:
                uncond_hidden_states = model(
                    input_ids=uncon_input_ids,
                    type_position_ids=uncon_type_position_ids,
                    attention_mask=uncond_t2i_attn_mask,
                    position_ids=uncon_position_ids,
                    infer_t2i=True,
                    cat='uncond',
                    use_cache=False,
                )
            # Hidden-state CFG convention: cfg_scale=4 computes
            # 5*cond - 4*uncond. Identical branches reduce exactly to cond.
            if not dual_cfg:
                if single_branch_cfg:
                    hidden_states = cond_hidden_states
                elif cfg_scale >= 0:
                    hidden_states = (
                        (1 + cfg_scale) * cond_hidden_states
                        - cfg_scale * uncond_hidden_states
                    )
                else:
                    hidden_states = uncond_hidden_states
            
            img_ans_mask = (type_position_ids == 4)
            # type 4 includes both real VQ positions and fixed NEW_LINE tokens.
            # vq_mask is false for NEW_LINE, so position_mask selects only the
            # still-unknown real spatial positions in this round.
            if vq_mask.dim() == 3:
                img_answer_vq_mask = vq_mask[img_ans_mask]  # [num_img_answer_tokens, num_codebooks]
            else:
                img_answer_vq_mask = vq_mask
            position_mask = img_answer_vq_mask[:, 0]  # [num_img_answer_tokens]
            
            # Project the backbone state into the eight-codebook channel
            # decoder's conditioning space, then retain current MASKs.
            cond = model.multi_codebook_head.condition_proj(hidden_states)

            if cond.dim() == 3:
                cond = cond[0]


            cond = cond[position_mask]
            base_cond = cond
            cond = base_cond.unsqueeze(1) + model.multi_codebook_head.timesteps_embeddings
            masked_positions = position_mask.nonzero(as_tuple=False).squeeze(-1)  # [num_valid]
            num_valid = masked_positions.shape[0]
            
            channel_sequence = [base_cond.unsqueeze(1)]

            
            next_tokens = []
            all_conf_list = []
            
            # Spatial positions are batched in parallel. This loop is the AR
            # part: codebook c conditions on sampled embeddings from codebooks
            # 0..c-1 at the same spatial position.
            for c in range(num_codebooks):
                x = torch.cat(channel_sequence, dim=1)
                x = x + model.multi_codebook_head.channel_embed[:, :x.shape[1]]
                curr_len = x.size(1)
                attn_mask = model.multi_codebook_head.channel_mask[:curr_len, :curr_len]
                curr_cond = cond[:, :curr_len]
                
                for block in model.multi_codebook_head.channel_blocks:
                    x = block(x, attn_mask=attn_mask, c=curr_cond)
                x = model.multi_codebook_head.channel_norm(x)
                x = model.multi_codebook_head.channel_final(x, curr_cond)
                
                logits = model.lm_img_head[c](x[:, -1:]).view(num_valid, -1)
                token = gumbel_max_sample(logits, temperature, generator=generator)
                
                # Keep per-codebook confidence for the later remasking score.
                next_tokens.append(token)
                
                probs = F.softmax(logits / temperature, dim=-1) if temperature > 0 else F.softmax(logits, dim=-1)
                conf = probs.gather(-1, token.unsqueeze(-1)).squeeze(-1)
                all_conf_list.append(conf)
                
                if c < num_codebooks - 1:
                    token_emb = model.model.img_embeddings[c](token)
                    channel_sequence.append(token_emb.unsqueeze(1))

            next_tokens = torch.stack(next_tokens, dim=1)
            # Cond and uncond must contain identical sampled image states before
            # the next CFG round; only their user/text prefixes differ.
            input_ids[0, img_ans_global_indices[masked_positions]] = next_tokens
            for _, branch_input_ids, branch_indices in branch_image_indices:
                branch_input_ids[0, branch_indices[masked_positions]] = next_tokens
            
            all_conf = torch.stack(all_conf_list, dim=1)
            # Mean the eight sampled probabilities into one spatial confidence;
            # lower-confidence positions are selected for remasking.
            position_conf = all_conf.mean(dim=-1)
            if masked_positions.numel() == 0:
                break
            
            # Ensure keep_n is 1D (B,) for mask_by_random_topk
            if keep_n.dim() == 0:
                keep_n_1d = keep_n.unsqueeze(0)
            elif keep_n.dim() > 1:
                keep_n_1d = keep_n.squeeze(1)
            else:
                keep_n_1d = keep_n
            mask_sel = mask_by_random_topk(
                keep_n_1d, position_conf.unsqueeze(0), temperature=temperature, generator=generator
            )[0]
            # mask_sel is over positions sampled in this round; keep_n controls
            # how many remain unknown for the next round.
            
            re_mask_count = 0
            if mask_sel.any():
                re_mask_pos = masked_positions[mask_sel]
                re_mask_count = re_mask_pos.numel()
                input_ids[0, img_ans_global_indices[re_mask_pos]] = mask_token_id
                for _, branch_input_ids, branch_indices in branch_image_indices:
                    branch_input_ids[0, branch_indices[re_mask_pos]] = mask_token_id

            vq_mask = input_ids[img_ans_mask] == mask_token_id
            unknown_cnt = vq_mask[:, 0].sum().view(1, 1)

            if debug:
                keep_n_val = int(keep_n.item()) if keep_n.numel() == 1 else int(keep_n[0].item())
                frac_val = float(noise_schedule(torch.tensor([(step + 1) / timesteps], device=device)).item()) if step < timesteps - 1 else 0.0
                debug_steps.append({
                    "step": step,
                    "masked": num_valid,
                    "keep_n": keep_n_val,
                    "frac": frac_val,
                    "enter_re_mask_block": True,  # ori always runs re-mask when keep_n>0
                    "re_mask_count": re_mask_count,
                    "next_tokens_sample": next_tokens[0, :4].cpu().tolist() if num_valid > 0 else [],
                })

    # Slice between BOI and [EOI, EOA]. pad_len is a legacy physical-padding
    # path; current blkpad inference leaves it None and instead passes explicit
    # position_ids. Finally remove structural NEW_LINE entries.
    if pad_len is not None:
        img_only = input_ids[:, code_start+1:-(2+pad_len), :]
    else:
        img_only = input_ids[:, code_start+1:-2, :]
    valid_rows = (img_only[0, :, 0] != newline_id)
    vq_ids = img_only[:, valid_rows, :].contiguous()
    if debug:
        vq_len_val = int(vq_len.item()) if vq_len.numel() == 1 else int(vq_len[0].item())
        debug_info = {"name": "generate_image_ar", "total_to_generate": vq_len_val, "steps": debug_steps}
        return (vq_ids, debug_info)
    return vq_ids

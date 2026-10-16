#!/usr/bin/env bash
# Muon SFT launcher. Same objective/data path as train_sft.sh, but trains with
# the Per-Head Muon optimizer (matrix params -> Muon, embeddings/heads/norms/bias ->
# AdamW inside the same optimizer).
#
# Init modes (pick one):
#   MODEL_INIT_PATH=<pretrain hf_ckpt>  -> fresh Muon SFT from HF weights
#   RESUME_CKPT=<Muon SFT dcp step dir> -> exact resume of a Muon SFT run
#     (pass --train.muon_checkpoint_mode resume to also restore optimizer state;
#      the default model_only starts a fresh Muon and requires
#      --train.resume_reset_lr_schedule true)
#
# Any trailing CLI args override the values below.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TRAIN_ENTRY="${TRAIN_ENTRY:-tasks/train_sft_muon.py}"

exec bash "${SCRIPT_DIR}/train_sft.sh" \
  --train.optimizer muon \
  --train.beta1 0.9 \
  --train.beta2 0.999 \
  --train.lr 1.0e-5 \
  --train.text_embedding_head_lr 5.0e-6 \
  --train.lr_start 0.0 \
  --train.lr_warmup_ratio 0.01 \
  --train.lr_decay_style constant \
  --train.lr_decay_ratio 1.0 \
  --train.weight_decay 0.1 \
  --train.muon_checkpoint_mode model_only \
  --train.muon_momentum 0.95 \
  --train.muon_nesterov true \
  --train.muon_ns_steps 5 \
  --train.muon_rms_scale 0.2 \
  --train.muon_per_head true \
  --train.muon_per_head_o_proj false \
  --train.muon_gather_dtype bf16 \
  --train.muon_owner_alltoall false \
  "$@"

#!/usr/bin/env bash
# Multitask SFT launcher. Text is masked with 156895 (<|mask|>) and terminal
# answer turns are EOA-padded-and-supervised -- matching the BD-pretrain
# objective. Continue ONLY from a BD-pretrain / SFT checkpoint (mask
# 156895); an incompatible checkpoint would silently produce garbage.
#
# Init modes (pick one):
#   MODEL_INIT_PATH=<pretrain hf_ckpt>   -> fresh SFT: HF weights only, optimizer /
#                                              scheduler / RNG / data all start new
#                                              (the usual "start SFT from pretrain" case)
#   RESUME_CKPT=<SFT dcp step dir>          -> resume an SFT run (full DCP state)
# If both are set, RESUME_CKPT wins.
#
# Activate your Python environment before running this script (see README).
set -euo pipefail

cd "$(dirname "$0")"

TRAIN_ENTRY="${TRAIN_ENTRY:-tasks/train_sft.py}"
TRAIN_CONFIG="${TRAIN_CONFIG:-configs/sft_packing.yaml}"

# ---- torchrun topology (override via env for multi-node) ----
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"

if [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "ERROR: nproc_per_node=${NPROC_PER_NODE} (need >= 1 GPU)" >&2
  exit 1
fi
if [[ "${NNODES}" -gt 1 && ( "${MASTER_ADDR}" == "127.0.0.1" || "${MASTER_ADDR}" == "0.0.0.0" ) ]]; then
  echo "ERROR: multi-node job requires a real MASTER_ADDR, got: ${MASTER_ADDR}" >&2
  exit 1
fi

# ---- init source (pretrain HF weights, or an SFT DCP resume) ----
MODEL_INIT_PATH="${MODEL_INIT_PATH:-}"
RESUME_CKPT="${RESUME_CKPT:-}"

# ---- paths ----
XLLMX_DATA_CACHE_DIR="${XLLMX_DATA_CACHE_DIR:-./xllmx_data_cache}"
export XLLMX_DATA_CACHE_DIR
mkdir -p "${XLLMX_DATA_CACHE_DIR}"

exp_name="${EXP_NAME:-./output/sft_multitask}"
mkdir -p "${exp_name}"
tensorboard_log_dir="${exp_name}/checkpoints/tensorboard_logs"
mkdir -p "${tensorboard_log_dir}"
LOG_HOST="$(hostname -s 2>/dev/null || hostname)"
LOG_HOST="${LOG_HOST%%.*}"
TRAIN_LOG="${exp_name}/train_1024_sft_${LOG_HOST}.log"

export PYTHONPATH="$(pwd):$(pwd)/VeOmni${PYTHONPATH:+:$PYTHONPATH}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRITON_CACHE_DIR="/tmp/triton_cache_$$"
mkdir -p "$TRITON_CACHE_DIR"
export TOKENIZERS_PARALLELISM=false

# Optional: object-storage config for remote data (leave unset for local data).
if [[ -n "${PETREL_CONF_PATH:-}" ]]; then
  export PETREL_CONF_PATH
fi
export DATALOADER_MP_CONTEXT="${DATALOADER_MP_CONTEXT:-fork}"
export TORCH_DISTRIBUTED_DEFAULT_TIMEOUT="${TORCH_DISTRIBUTED_DEFAULT_TIMEOUT:-1800}"
export TORCHELASTIC_RENDEZVOUS_TIMEOUT="${TORCHELASTIC_RENDEZVOUS_TIMEOUT:-1800}"

# Init args: DCP resume takes precedence; else HF-weights-only init from pretrain.
INIT_ARGS=()
if [[ -n "${RESUME_CKPT}" ]]; then
  INIT_ARGS+=(--train.load_checkpoint_path "${RESUME_CKPT}")
  echo "init: DCP resume from ${RESUME_CKPT}"
else
  if [[ -z "${MODEL_INIT_PATH}" ]]; then
    echo "ERROR: set MODEL_INIT_PATH to a BD-pretrain HF checkpoint (or RESUME_CKPT to resume an SFT run)." >&2
    exit 1
  fi
  INIT_ARGS+=(--model.load_model_path "${MODEL_INIT_PATH}")
  echo "init: HF-weights-only from ${MODEL_INIT_PATH} (fresh optimizer/scheduler/data)"
fi

echo "python: $(which python)"
echo "torchrun: nproc_per_node=${NPROC_PER_NODE} nnodes=${NNODES} node_rank=${NODE_RANK} master=${MASTER_ADDR}:${MASTER_PORT}"
echo "output_dir=${exp_name}"
echo "log_file=${TRAIN_LOG}"
exec >> "${TRAIN_LOG}" 2>&1

torchrun \
  --nproc_per_node "${NPROC_PER_NODE}" \
  --nnodes "${NNODES}" \
  --node_rank "${NODE_RANK}" \
  --master_addr "${MASTER_ADDR}" \
  --master_port "${MASTER_PORT}" \
  "${TRAIN_ENTRY}" \
  "${TRAIN_CONFIG}" \
  --train.output_dir "$exp_name" \
  --train.micro_batch_size 1 \
  --train.global_batch_size 128 \
  --train.lr 2.0e-5 \
  --train.use_same_datasets false \
  --train.resume_use_new_lr true \
  --train.resume_reset_lr_schedule false \
  --train.tensorboard_log_dir "${tensorboard_log_dir}" \
  --train.enable_batch_dump false \
  --train.save_steps 250 \
  --train.text_loss_weight 1.0 \
  --train.image_loss_weight 1.0 \
  --train.max_grad_norm 1.0 \
  --data.train_size 150000000000 \
  --data.max_seq_len 16384 \
  --data.dyn_pack_max_seq_len 16384 \
  --data.max_text_length 4096 \
  --data.num_workers "${DATA_NUM_WORKERS:-8}" \
  --train.bsz_warmup_ratio 0 \
  --data.train_path "${SFT_TRAIN_PATH:-data/example_sft.yaml}" \
  "${INIT_ARGS[@]}" \
  "$@" \
  2>&1

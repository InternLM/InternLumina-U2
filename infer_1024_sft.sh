#!/usr/bin/env bash
# SFT inference driver. Runs the SFT entrypoints (text mask id 156895), so it
# must be pointed ONLY at a compatible Intern Lumina U2 SFT checkpoint. The
# Python entrypoints validate the tokenizer's special-token IDs before inference.
#
# Usage:
#   CHECKPOINT=/path/to/sft/checkpoints/e0-XXXX/hf_ckpt_split bash infer_1024_sft.sh
#   SECTION=t2i T2I_PROMPT="..." CHECKPOINT=... bash infer_1024_sft.sh
# Sections: t2i(default) text under under_ut edit video 3d
#
# The T2I / image sections additionally need the AToken tokenizer weights:
#   ATOKEN_MODEL_PATH=/path/to/atoken-sod.pt (default: atoken_inference/checkpoints/atoken-sod.pt)
set -euo pipefail

SECTION="${SECTION:-t2i}"
case "${SECTION}" in
  all|text|t2i|edit|under|under_ut|video|3d) ;;
  *)
    echo "ERROR: unsupported SECTION=${SECTION}; choose all, text, t2i, edit, under, under_ut, video, or 3d" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${LUMINA_ROOT:-}"
if [[ -z "${ROOT}" ]]; then ROOT="${SCRIPT_DIR}"; fi
if [[ ! -d "${ROOT}" ]]; then
  echo "ERROR: repository root not found: ${ROOT}" >&2
  exit 1
fi
ROOT="$(cd "${ROOT}" && pwd)"
cd "${ROOT}"
TORCHRUN="${TORCHRUN:-}"
if [[ -z "${TORCHRUN}" ]]; then TORCHRUN="$(command -v torchrun || true)"; fi
if [[ -z "${TORCHRUN}" ]]; then
  echo "ERROR: torchrun not found in PATH" >&2
  exit 1
fi
export PYTHONPATH="${ROOT}:${ROOT}/VeOmni${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1
INFER_NPROC_PER_NODE="${INFER_NPROC_PER_NODE:-1}"
if [[ "${INFER_NPROC_PER_NODE}" != "1" ]]; then
  echo "ERROR: inference is single-process; set INFER_NPROC_PER_NODE=1" >&2
  exit 2
fi
infer_torchrun() {
  "${TORCHRUN}" --standalone --nproc_per_node 1 "${@}"
}

if [[ -z "${CHECKPOINT:-}" ]]; then
  echo "ERROR: set CHECKPOINT to a compatible SFT split dir (…/hf_ckpt_split)." >&2
  exit 1
fi
SEED="${SEED:-65513}"
OUTPUT_BASE="${OUTPUT_BASE:-$(dirname "${CHECKPOINT%/}")/output_sft_infer}"
mkdir -p "${OUTPUT_BASE}"

DEFAULT_TEXT_SYSTEM_PROMPT="You are a helpful assistant."
TEXT_SYSTEM_PROMPT="${TEXT_SYSTEM_PROMPT:-${DEFAULT_TEXT_SYSTEM_PROMPT}}"
DEFAULT_UNDER_SYSTEM_PROMPT="You are a multimodal assistant. Answer directly using the provided image(s), text, and conversation context; if there is no question, describe the image(s) clearly and accurately."
UNDER_SYSTEM_PROMPT="${UNDER_SYSTEM_PROMPT:-${DEFAULT_UNDER_SYSTEM_PROMPT}}"
DEFAULT_VIDEO_SYSTEM_PROMPT="You are a multimodal model that can process text and video. Answer the following question based on the provided video frames or clips. Analyze temporal dynamics, motion, interactions, and scene changes across frames. Track relevant objects and events over time, and integrate information to produce a consistent and accurate answer."
VIDEO_SYSTEM_PROMPT="${VIDEO_SYSTEM_PROMPT:-${DEFAULT_VIDEO_SYSTEM_PROMPT}}"
DEFAULT_UNDER_3D_SYSTEM_PROMPT="You are a multimodal model that can process text and 3D content. Answer the following question based on the provided 3D representations. Carefully analyze the object's geometry, spatial structure, viewpoint consistency, and fine-grained details across views. Combine relevant information to produce a coherent and accurate answer."
UNDER_3D_SYSTEM_PROMPT="${UNDER_3D_SYSTEM_PROMPT:-${DEFAULT_UNDER_3D_SYSTEM_PROMPT}}"
DEFAULT_T2I_SYSTEM_PROMPT="Generate an image according to the text prompt."
T2I_SYSTEM_PROMPT="${T2I_SYSTEM_PROMPT:-${DEFAULT_T2I_SYSTEM_PROMPT}}"
DEFAULT_EDIT_SYSTEM_PROMPT="Generate an image applying the following editing instruction based on the original image."
EDIT_SYSTEM_PROMPT="${EDIT_SYSTEM_PROMPT:-${DEFAULT_EDIT_SYSTEM_PROMPT}}"

# Demo assets shipped with the repo.
IMG_NATURAL="${ROOT}/imgs/natural.jpg"
IMG_GRAPH="${ROOT}/imgs/graph.png"
UT_NATURAL="Could you provide a caption that neatly summarizes the image without losing focus on the important visuals?"
UT_GRAPH="Can you look at this picture and let me know if it's a chart or table? Describe the chart in Markdown or make a LaTeX version of the table, if it's there. Also, what can you make of it?"

run() { [[ "${SECTION}" == "all" || "${SECTION}" == "$1" ]]; }

# ── text: single-turn text generation. ──
if run text; then
  TEXT_PROMPT="${TEXT_PROMPT:-Explain why the sky appears blue in a few sentences.}"
  infer_torchrun "${ROOT}/infer_mmu_sft.py" \
    --checkpoint "${CHECKPOINT}" --task_type text \
    --system_prompt "${TEXT_SYSTEM_PROMPT}" --user_text "${TEXT_PROMPT}" \
    --gen_length "${TEXT_GEN_LEN:-1024}" --block_length "${TEXT_BLOCK_LEN:-32}" \
    --timesteps "${TEXT_TIMESTEPS:-32}" --temperature "${TEXT_TEMP:-0.0}" \
    --seed "${SEED}" --output_dir "${OUTPUT_BASE}/text"
fi

# ── t2i: one user prompt. Override T2I_PROMPT as needed; size/cfg/temperature via
#    T2I_SIZE (HxW), T2I_CFG, T2I_TEMP. ──
T2I_SIZE="${T2I_SIZE:-1024x1024}"
T2I_CFG="${T2I_CFG:-3.0}"
T2I_TEMP="${T2I_TEMP:-1.0}"

if run t2i; then
  T2I_RUN_PROMPT="${T2I_PROMPT:-A serene mountain landscape reflected in a crystal-clear alpine lake at sunrise.}"
  H="${T2I_SIZE%x*}"; W="${T2I_SIZE#*x}"
  echo "=== t2i ${H}x${W} cfg${T2I_CFG} temp${T2I_TEMP} seed${SEED} ==="
  infer_torchrun "${ROOT}/infer_t2i_sft.py" \
    --checkpoint "${CHECKPOINT}" --task_type t2i \
    --system_prompt "${T2I_SYSTEM_PROMPT}" \
    --user_text "${T2I_RUN_PROMPT}" \
    --image_height "${H}" --image_width "${W}" --timesteps "${T2I_TIMESTEPS:-96}" \
    --cfg_scale "${T2I_CFG}" --temperature "${T2I_TEMP}" --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/t2i_${H}x${W}_cfg${T2I_CFG}_temp${T2I_TEMP}"
fi

# ── edit: image editing on a user-supplied source image. EDIT_IMAGE is a raw
#    image file (encoded on the fly with AToken); EDIT_INSTRUCTION is the editing
#    instruction; EDIT_MAX_EDGE caps the encode resolution. Set EDIT_CFG for
#    single CFG, or set EDIT_CFG_TEXT and EDIT_CFG_IMG together for dual CFG. ──
EDIT_CFG="${EDIT_CFG:-1}"
EDIT_CFG_TEXT="${EDIT_CFG_TEXT:-}"
EDIT_CFG_IMG="${EDIT_CFG_IMG:-}"
EDIT_TIMESTEPS="${EDIT_TIMESTEPS:-64}"
EDIT_MAX_EDGE="${EDIT_MAX_EDGE:-1024}"

if run edit; then
  if [[ -n "${EDIT_CFG_TEXT}" || -n "${EDIT_CFG_IMG}" ]]; then
    if [[ -z "${EDIT_CFG_TEXT}" || -z "${EDIT_CFG_IMG}" ]]; then
      echo "ERROR: EDIT_CFG_TEXT and EDIT_CFG_IMG must be set together" >&2
      exit 2
    fi
    EDIT_CFG_ARGS=(--cfg_scale 0 --cfg_text "${EDIT_CFG_TEXT}" --cfg_img "${EDIT_CFG_IMG}")
    EDIT_CFG_LABEL="cfg_text${EDIT_CFG_TEXT}_cfg_img${EDIT_CFG_IMG}"
  else
    EDIT_CFG_ARGS=(--cfg_scale "${EDIT_CFG}")
    EDIT_CFG_LABEL="cfg${EDIT_CFG}"
  fi
  EDIT_IMAGE="${EDIT_IMAGE:-${ROOT}/imgs/edit_source.png}"
  EDIT_INSTRUCTION="${EDIT_INSTRUCTION:-Convert the scene to a charcoal sketch style.}"
  EDIT_TAG="${EDIT_TAG:-edit0}"
  echo "=== edit ${EDIT_TAG} ${EDIT_CFG_LABEL} edge<=${EDIT_MAX_EDGE} ==="
  infer_torchrun "${ROOT}/infer_t2i_sft.py" \
    --checkpoint "${CHECKPOINT}" --task_type single_edit \
    --system_prompt "${EDIT_SYSTEM_PROMPT}" \
    --user_text "${EDIT_INSTRUCTION}" \
    --input_image_paths "${EDIT_IMAGE}" \
    --input_max_edge "${EDIT_MAX_EDGE}" \
    --timesteps "${EDIT_TIMESTEPS}" "${EDIT_CFG_ARGS[@]}" \
    --temperature 1.0 --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/edit/${EDIT_TAG}"
fi

# ── under / caption: gen_length 4096 / timesteps 32 / block 32 / temperature 0.0.
#    Defaults run the shipped demo images; point UNDER_IMAGE (+ optional
#    UNDER_PROMPT) at your own image to ask your own question. UNDER_MAX_EDGE caps
#    the encode resolution (this SFT run's understanding data is 512 long-edge;
#    much larger inputs can make the model emit EOA at position 0). ──
UNDER_MAX_EDGE="${UNDER_MAX_EDGE:-512}"
CAP_IMAGES=(
  "${IMG_NATURAL}|natural|${UT_NATURAL}"
  "${IMG_GRAPH}|graph|${UT_GRAPH}"
)

_caption_sft() {  # _caption_sft <image> <name> <user_text> <out-subdir>
  infer_torchrun "${ROOT}/infer_mmu_sft.py" \
    --checkpoint "${CHECKPOINT}" --task_type under \
    --system_prompt "${UNDER_SYSTEM_PROMPT}" \
    --user_text "$3" \
    --image_path "$1" --input_max_edge "${UNDER_MAX_EDGE}" \
    --gen_length "${UNDER_GEN_LEN:-4096}" --block_length "${UNDER_BLOCK_LEN:-32}" \
    --timesteps "${UNDER_TIMESTEPS:-32}" \
    --temperature 0.0 --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/$4/$2"
}

if run under; then
  if [[ -n "${UNDER_IMAGE:-}" ]]; then
    _caption_sft "${UNDER_IMAGE}" "$(basename "${UNDER_IMAGE%.*}")" "${UNDER_PROMPT:-}" mmu
  else
    for entry in "${CAP_IMAGES[@]}"; do
      IFS='|' read -r img name ut <<< "${entry}"
      _caption_sft "${img}" "${name}" "" mmu
    done
  fi
fi

# ── under_ut: demo images with their demo questions ──
if run under_ut; then
  for entry in "${CAP_IMAGES[@]}"; do
    IFS='|' read -r img name ut <<< "${entry}"
    _caption_sft "${img}" "${name}" "${ut}" mmu_usertext
  done
fi


# ── video understanding (VIDEO_PATH: any local .mp4; VIDEO_PROMPT optional) ──
if run video; then
  VIDEO_PATH="${VIDEO_PATH:-${ROOT}/imgs/demo_video.mp4}"
  infer_torchrun "${ROOT}/infer_video_sft.py" \
    --checkpoint "${CHECKPOINT}" \
    --system_prompt "${VIDEO_SYSTEM_PROMPT}" \
    --user_text "${VIDEO_PROMPT:-Describe this video in detail from start to finish.}" \
    --video_path "${VIDEO_PATH}" --num_frames "${VIDEO_NUM_FRAMES:-64}" \
    --video_max_size "${VIDEO_MAX_SIZE:-448}" \
    --timesteps "${VIDEO_TIMESTEPS:-32}" --gen_length "${VIDEO_GEN_LEN:-1024}" \
    --block_length "${VIDEO_BLOCK_LEN:-32}" \
    --temperature 1.0 --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/video"
fi

# ── 3d understanding. GLB/GLTF requires Blender for training-consistent multiview
#    encoding (install once: bash tools/setup_blender.sh; or set BLENDER_BIN). ──
if run 3d; then
  ASSET_3D="${ASSET_3D:-${ROOT}/imgs/demo_3d.glb}"
  # Optional overrides are useful when Blender or AToken assets live outside the
  # repository; the bundled GLB encoder is used by default.
  EXTRA_ARGS_3D=()
  [[ -n "${ATOKEN_3D_ROOT:-}" ]] && EXTRA_ARGS_3D+=(--atoken_3d_root "${ATOKEN_3D_ROOT}")
  [[ -n "${ATOKEN_ROOT_3D:-}" ]] && EXTRA_ARGS_3D+=(--atoken_root "${ATOKEN_ROOT_3D}")
  [[ -n "${BLENDER_3D:-}" ]] && EXTRA_ARGS_3D+=(--blender_bin "${BLENDER_3D}")
  RENDER_DIR_ARGS_3D=()
  if [[ -n "${RENDER_DIR_3D:-}" ]]; then
    RENDER_DIR_ARGS_3D=(--render_dir "${RENDER_DIR_3D}")
  fi
  BLENDER_LIB_ARGS_3D=()
  if [[ -n "${BLENDER_LIBS_3D:-}" ]]; then
    BLENDER_LIB_ARGS_3D=(--blender_lib_dir "${BLENDER_LIBS_3D}")
  fi
  infer_torchrun "${ROOT}/infer_3d_sft.py" \
    --checkpoint "${CHECKPOINT}" \
    --system_prompt "${UNDER_3D_SYSTEM_PROMPT}" \
    --user_text "${PROMPT_3D:-Describe this 3D object in detail.}" \
    --asset_path "${ASSET_3D}" --voxel_resolution 48 \
    "${EXTRA_ARGS_3D[@]}" \
    "${BLENDER_LIB_ARGS_3D[@]}" \
    --render_num_views "${NUM_VIEWS_3D:-64}" --render_image_size 256 \
    --render_samples "${RENDER_SAMPLES_3D:-8}" \
    "${RENDER_DIR_ARGS_3D[@]}" \
    --mesh_samples "${MESH_SAMPLES_3D:-100000}" \
    --gen_length "${GEN_LEN_3D:-128}" --block_length "${BLOCK_LEN_3D:-32}" \
    --timesteps "${TIMESTEPS_3D:-32}" \
    --temperature 1.0 --seed "${SEED}" \
    --output_dir "${OUTPUT_BASE}/under_3d"
fi

echo "=== SFT inference done (section=${SECTION}) -> ${OUTPUT_BASE} ==="

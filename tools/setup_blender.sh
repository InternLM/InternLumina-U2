#!/usr/bin/env bash
# Download a headless-capable Blender into third_party/ for the 3D multiview
# encoding path. infer_3d_sft.py picks it up automatically (no env var needed).
#
# Usage: bash tools/setup_blender.sh
set -euo pipefail

BLENDER_VERSION="${BLENDER_VERSION:-4.5.13}"
BLENDER_SERIES="${BLENDER_VERSION%.*}"
NAME="blender-${BLENDER_VERSION}-linux-x64"
URL="https://download.blender.org/release/Blender${BLENDER_SERIES}/${NAME}.tar.xz"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/third_party"
mkdir -p "${DEST}"

if [[ -x "${DEST}/${NAME}/blender" ]]; then
  echo "Blender already installed: ${DEST}/${NAME}/blender"
  exit 0
fi

echo "Downloading ${URL} ..."
TARBALL="${DEST}/${NAME}.tar.xz"
if command -v wget >/dev/null 2>&1; then
  wget -q --show-progress -O "${TARBALL}" "${URL}"
else
  curl -L --progress-bar -o "${TARBALL}" "${URL}"
fi

echo "Extracting ..."
tar -xJf "${TARBALL}" -C "${DEST}"
rm -f "${TARBALL}"

BIN="${DEST}/${NAME}/blender"
if [[ ! -x "${BIN}" ]]; then
  echo "ERROR: extraction finished but ${BIN} is missing" >&2
  exit 1
fi

echo "Blender installed: ${BIN}"
echo "Verifying headless startup ..."
if "${BIN}" --background --factory-startup --python-expr "print('blender-ok')" 2>/dev/null | grep -q blender-ok; then
  echo "OK - infer_3d_sft.py will find it automatically."
else
  cat >&2 <<'EOF'
Blender downloaded, but headless startup failed - some system libraries are
probably missing. On Debian/Ubuntu install them with:

  sudo apt-get install -y libx11-6 libxi6 libxxf86vm1 libxfixes3 libxrender1 \
      libxkbcommon0 libgl1 libegl1 libsm6

If you cannot install system packages, place the missing .so files in a local
directory and point BLENDER_LIB_DIR at it (it is prepended to LD_LIBRARY_PATH).
EOF
  exit 1
fi

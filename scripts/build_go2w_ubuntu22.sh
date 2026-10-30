#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GUI=ON
INSTALL=0
for arg in "$@"; do
  case "$arg" in
    --headless) GUI=OFF ;;
    --install-deps) INSTALL=1 ;;
    *) echo "Usage: bash scripts/build_go2w_ubuntu22.sh [--headless] [--install-deps]"; exit 2 ;;
  esac
done
if [[ "$(uname -m)" != x86_64 ]]; then
  echo "This release is validated for Ubuntu 22.04 x86_64."; exit 1
fi
if [[ "$INSTALL" == 1 ]]; then
  SUDO=()
  if [[ "$EUID" != 0 ]]; then SUDO=(sudo); fi
  "${SUDO[@]}" apt-get update
  DEPS=(build-essential cmake python3 libboost-program-options-dev libyaml-cpp-dev libeigen3-dev libspdlog-dev libfmt-dev)
  if [[ "$GUI" == ON ]]; then DEPS+=(libglfw3-dev); fi
  "${SUDO[@]}" apt-get install -y "${DEPS[@]}"
fi
python3 "$ROOT/scripts/apply_go2w_sdk_patch.py"
python3 "$ROOT/scripts/repair_linux_links.py"
python3 "$ROOT/scripts/prepare_go2w_sim.py"
JOBS="${GO2W_BUILD_JOBS:-2}"
cmake -S "$ROOT/deploy/robots/go2w" -B "$ROOT/deploy/robots/go2w/build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$ROOT/deploy/robots/go2w/build" -j "$JOBS"
cmake -S "$ROOT/simulate" -B "$ROOT/simulate/build" -DCMAKE_BUILD_TYPE=Release -DUNITREE_BUILD_GUI="$GUI"
cmake --build "$ROOT/simulate/build" -j "$JOBS"
ctest --test-dir "$ROOT/deploy/robots/go2w/build" --output-on-failure
GO2W_CRC_TEST="$ROOT/deploy/robots/go2w/build/go2w_crc_test" python3 "$ROOT/tests/test_go2w_crc.py"
echo "Build complete. Run: python3 scripts/check_go2w_ubuntu22.py"

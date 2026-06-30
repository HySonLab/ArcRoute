#!/usr/bin/env bash
# Setup script for HDCARP using uv (https://github.com/astral-sh/uv)
#
# Usage:
#   ./setup.sh
#
# This installs uv if missing, creates a virtual environment in .venv,
# and installs all dependencies pinned in pyproject.toml.
set -euo pipefail

cd "$(dirname "$0")"

# 1. Install uv if it is not already available.
if ! command -v uv >/dev/null 2>&1; then
    echo ">> uv not found, installing..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Make uv available in the current shell session.
    export PATH="$HOME/.local/bin:$PATH"
fi

echo ">> uv version: $(uv --version)"

# 2. Create the virtual environment and install dependencies.
echo ">> Syncing dependencies into .venv ..."
uv sync

echo
echo ">> Done. Activate the environment with:"
echo "     source .venv/bin/activate"
echo
echo ">> Or run commands directly via uv, e.g.:"
echo "     uv run python src/train.py --help"
echo "     uv run python -m solvers.ils --data_path \"data/instances/30/61_20.npz\""

#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

HERMES_DIR="${HOME}/hermes-agent"

# Clone hermes-agent if not already present
if [ ! -d "${HERMES_DIR}/.git" ]; then
  echo "Cloning hermes-agent..."
  git clone --depth=1 https://github.com/NousResearch/hermes-agent.git "${HERMES_DIR}"
fi

# Install dependencies with uv
echo "Installing hermes-agent dependencies..."
cd "${HERMES_DIR}"
uv sync

# Expose hermes-agent's virtualenv to the session
VENV_BIN="${HERMES_DIR}/.venv/bin"
if [ -d "${VENV_BIN}" ]; then
  echo "export PATH=\"${VENV_BIN}:\$PATH\"" >> "${CLAUDE_ENV_FILE}"
  echo "export PYTHONPATH=\"${HERMES_DIR}:\${PYTHONPATH:-}\"" >> "${CLAUDE_ENV_FILE}"
fi

echo "hermes-agent setup complete."

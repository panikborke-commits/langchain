#!/bin/bash
set -euo pipefail

# Only run in remote (Claude Code on the web) environments
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  exit 0
fi

RAGFLOW_DIR="${HOME}/ragflow"

# Clone RAGFlow if not already present
if [ ! -d "${RAGFLOW_DIR}/.git" ]; then
  echo "Cloning RAGFlow..."
  git clone --depth=1 https://github.com/infiniflow/ragflow.git "${RAGFLOW_DIR}"
fi

# Patch out the Aliyun PyPI mirror and redirect Gitee git deps to GitHub
python3 -c "
import re, pathlib, sys
p = pathlib.Path(sys.argv[1])
txt = p.read_text()
# Remove custom Aliyun PyPI mirror index
txt = re.sub(r'\[\[tool\.uv\.index\]\]\s*\nurl\s*=\s*\"https://mirrors\.aliyun[^\"]*\"\s*\n', '', txt)
# Redirect Gitee git sources to GitHub
txt = txt.replace('https://gitee.com/infiniflow/', 'https://github.com/infiniflow/')
p.write_text(txt)
" "${RAGFLOW_DIR}/pyproject.toml"

# Remove the lock file so uv re-resolves packages against public PyPI
rm -f "${RAGFLOW_DIR}/uv.lock"

# Install dependencies with uv (resolves fresh from PyPI after patch)
echo "Installing RAGFlow dependencies..."
cd "${RAGFLOW_DIR}"
uv sync --python 3.12

# Expose RAGFlow's virtualenv to the session
VENV_BIN="${RAGFLOW_DIR}/.venv/bin"
if [ -d "${VENV_BIN}" ]; then
  echo "export PATH=\"${VENV_BIN}:\$PATH\"" >> "${CLAUDE_ENV_FILE}"
  echo "export PYTHONPATH=\"${RAGFLOW_DIR}:\${PYTHONPATH:-}\"" >> "${CLAUDE_ENV_FILE}"
fi

echo "RAGFlow setup complete."

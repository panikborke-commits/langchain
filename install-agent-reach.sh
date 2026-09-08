#!/bin/bash

# Global installation script for Agent-Reach in LangChain monorepo
# Installs Agent-Reach globally and makes it available to all LangChain packages

set -e

echo "Installing Agent-Reach globally for LangChain monorepo..."

# Install agent-reach from GitHub
if pip show agent-reach > /dev/null 2>&1; then
    echo "Agent-Reach is already installed. Updating..."
    pip install --upgrade "git+https://github.com/Panniantong/Agent-Reach.git@main#egg=agent-reach" || pip install --upgrade agent-reach
else
    echo "Installing Agent-Reach..."
    pip install -q "git+https://github.com/Panniantong/Agent-Reach.git@main#egg=agent-reach" || pip install -q agent-reach
fi

# Set up uv dependencies if uv is available
if command -v uv &> /dev/null; then
    echo "Syncing uv dependencies..."
    uv sync --all-groups 2>/dev/null || uv sync 2>/dev/null || true
fi

echo "Verifying Agent-Reach installation..."
if agent-reach --version 2>/dev/null || python -m agent_reach.cli --version 2>/dev/null; then
    echo "✓ Agent-Reach successfully installed"
else
    echo "⚠ Warning: Agent-Reach CLI not immediately available"
fi

echo "✓ Installation complete. Agent-Reach is now globally available."
echo "  Type: agent-reach --help"
echo "  Or:   python -m agent_reach.cli --help"

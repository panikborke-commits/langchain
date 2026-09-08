# Agent-Reach Global Integration for LangChain

This document describes the global Agent-Reach integration in the LangChain monorepo.

## Installation

### Automatic Setup

```bash
./install-agent-reach.sh
```

### Manual Installation

```bash
# Install directly from GitHub
pip install git+https://github.com/Panniantong/Agent-Reach.git@main

# Or via PyPI
pip install agent-reach

# Verify
agent-reach --version
```

## Project Configuration

Agent-Reach is declared in the root `pyproject.toml`:

```toml
[project]
dependencies = [
    "agent-reach>=1.5.0",
]
```

This makes it available globally across the entire monorepo.

## Usage in LangChain Packages

Any package in the monorepo can now use Agent-Reach:

```python
from agent_reach import AgentReach
from agent_reach.channels import TwitterChannel, RedditChannel

# Create agent
agent = AgentReach()

# Search
results = agent.search("AI agents", platform="twitter")

# Read content
content = agent.read("https://example.com")
```

## CLI Access

```bash
# Global access
agent-reach --help
agent-reach doctor
agent-reach search "query"

# Or via Python module
python -m agent_reach.cli --help
```

## Supported Platforms

- Twitter/X
- Reddit  
- YouTube
- Bilibili
- Xiaohongshu (XHS)
- Google Search
- Hacker News
- Medium
- Dev.to
- RSS Feeds
- Generic HTTP/HTTPS

## Integration with LangChain Tools

Agent-Reach provides:

1. **Web Reading** – Read full content from URLs across 13+ platforms
2. **Search** – Unified search interface across multiple platforms
3. **No API Keys Required** – Works with public content without authentication
4. **MCP Server** – Can be exposed as an MCP tool for Claude and other agents

### Using with LangChain Agents

```python
from langchain.agents import initialize_agent
from agent_reach import get_agent_reach_tools

# Get Agent-Reach tools for LangChain
tools = get_agent_reach_tools()

# Use in agent
agent = initialize_agent(
    tools,
    llm,
    agent="zero-shot-react-description",
)
```

## Troubleshooting

```bash
# Check installation
pip show agent-reach
python -m agent_reach.cli --version

# Reinstall if needed
pip install --upgrade --force-reinstall agent-reach

# Run diagnostics
agent-reach doctor
```

## References

- [Agent-Reach GitHub](https://github.com/Panniantong/Agent-Reach)
- [LangChain Documentation](https://docs.langchain.com/)

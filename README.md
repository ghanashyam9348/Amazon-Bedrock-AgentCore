# AgentCore Crash Course (GitHub Upload Version)

This folder is a clean, upload-ready version of the project.
It was prepared in a separate directory so the original project files remain unchanged.

## What is in this version

- Uses a new custom dataset: `data/custom_faq.csv`
- Does not include the original provided dataset
- Includes scripts for local LangGraph, AgentCore runtime, and AgentCore memory
- Includes a build script for FAISS index generation

## Project file guide

### `00_langgraph_agent.py`
Implements a local tool-using FAQ assistant using LangChain + LangGraph-style agent tooling:

- Loads FAQ rows from `data/custom_faq.csv`
- Builds embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- Creates a FAISS vector store in memory
- Defines three tools:
  - `search_faq`
  - `search_detailed_faq`
  - `reformulate_query`
- Runs the agent with Groq model `openai/gpt-oss-20b`

Use this file to understand the baseline agent behavior before deployment.

### `build_index.py`
Pre-computes a local FAISS index from `data/custom_faq.csv`:

- Reads question/answer records
- Splits content into chunks
- Uses `FastEmbedEmbeddings` (`BAAI/bge-small-en-v1.5`)
- Saves index in `faiss_index/`

Run this before executing runtime scripts that load `faiss_index`.

### `01_agentcore_runtime.py`
Implements AgentCore runtime integration without conversation memory:

- Loads pre-built `faiss_index/`
- Uses the same three FAQ tools
- Creates `BedrockAgentCoreApp` entrypoint
- Accepts payload prompt and returns final agent response

Use this file for stateless runtime deployment.

### `02_agentcore_memory.py`
Implements AgentCore runtime with short-term memory support:

- Loads pre-built `faiss_index/`
- Configures `AgentCoreMemorySaver`
- Uses payload fields `actor_id` and `thread_id`
- Preserves thread-level context across invocations

Use this file for multi-turn conversational behavior.

### `data/custom_faq.csv`
Custom FAQ dataset created for this upload package.

### `faiss_index/`
Directory used to store generated FAISS index files.

### `pyproject.toml`
Dependency and Python version configuration (`>=3.13`).

### `.sample_env`
Template for environment variables.

### `.gitignore`
Ignores local/secret/runtime artifact files.

## Setup

1. Create virtual environment and install dependencies:

```bash
uv sync
```

2. Create `.env` from `.sample_env` and add your API key:

```env
GROQ_API_KEY=your_real_key
```

3. Build vector index:

```bash
uv run python build_index.py
```

## Run

### Local baseline

```bash
uv run python 00_langgraph_agent.py
```

### AgentCore runtime (stateless)

```bash
agentcore configure -e 01_agentcore_runtime.py
agentcore launch --env GROQ_API_KEY=your_real_key
```

### AgentCore runtime (with memory)

```bash
agentcore configure -e 02_agentcore_memory.py
agentcore launch --env GROQ_API_KEY=your_real_key
```

## Notes

- Keep using the same `thread_id` in memory mode to preserve context.
- Use a new `thread_id` to start a fresh conversation.
- If `faiss_index/` is missing, run `build_index.py` again.

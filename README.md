# Amazon Bedrock AgentCore FAQ Assistant

## Project Description

This project is a practical reference for building a Retrieval-Augmented Generation (RAG) style FAQ assistant with Amazon Bedrock AgentCore.
It demonstrates how to move from a local prototype to a deployable runtime service, and then to a memory-enabled conversational agent.

The assistant uses semantic search over a custom FAQ dataset, calls tools to retrieve relevant answers, and returns grounded responses using an LLM.

## What This Project Does

- Ingests FAQ question-answer pairs from [data/custom_faq.csv](data/custom_faq.csv)
- Builds vector embeddings and stores them in a FAISS index for fast semantic retrieval
- Exposes retrieval tools the agent can call during reasoning
- Runs in three modes: local script, AgentCore runtime, and AgentCore runtime with short-term memory
- Supports multi-turn conversations by preserving context with actor_id and thread_id in memory mode

This project shows how to build and run a tool-using FAQ agent in three stages:

1. Local agent execution
2. AgentCore runtime deployment
3. AgentCore runtime with short-term conversation memory

This repository uses a custom FAQ dataset in [data/custom_faq.csv](data/custom_faq.csv) and does not include the original course dataset.

## Features

- Semantic FAQ retrieval using FAISS
- Tool-based agent behavior (search + query reformulation)
- Local script and AgentCore runtime entrypoints
- Memory-aware runtime flow using thread and actor identifiers

## Repository Structure

- [00_langgraph_agent.py](00_langgraph_agent.py): Local baseline agent that builds FAISS from CSV in memory and answers a sample question.
- [build_index.py](build_index.py): One-time utility to generate [faiss_index/](faiss_index) from [data/custom_faq.csv](data/custom_faq.csv).
- [01_agentcore_runtime.py](01_agentcore_runtime.py): AgentCore runtime app with stateless request handling.
- [02_agentcore_memory.py](02_agentcore_memory.py): AgentCore runtime app with short-term memory via AgentCoreMemorySaver.
- [data/custom_faq.csv](data/custom_faq.csv): Custom public FAQ content used for retrieval.
- [faiss_index/](faiss_index): Generated FAISS index artifacts used by runtime scripts.
- [pyproject.toml](pyproject.toml): Python and dependency configuration.
- [.sample_env](.sample_env): Environment variable template.

## Requirements

- Python 3.13+
- uv package manager
- AWS credentials configured for AgentCore workflows
- GROQ_API_KEY for model access

## Quick Start

1. Install dependencies.

```bash
uv sync
```

2. Create a local environment file.

```bash
cp .sample_env .env
```

3. Set your key in .env.

```env
GROQ_API_KEY=your_groq_api_key
```

4. Build vector index from the custom dataset.

```bash
uv run python build_index.py
```

## Run Locally

Execute the local baseline agent:

```bash
uv run python 00_langgraph_agent.py
```

## Run with AgentCore (Stateless)

```bash
agentcore configure -e 01_agentcore_runtime.py
agentcore launch --env GROQ_API_KEY=your_groq_api_key
```

Example invoke:

```bash
agentcore invoke '{"prompt":"Explain roaming activation"}'
```

## Run with AgentCore Memory

```bash
agentcore configure -e 02_agentcore_memory.py
agentcore launch --env GROQ_API_KEY=your_groq_api_key
```

Example multi-turn invoke:

```bash
agentcore invoke '{"prompt":"My name is Alex","actor_id":"user1","thread_id":"chat1"}'
agentcore invoke '{"prompt":"What is my name?","actor_id":"user1","thread_id":"chat1"}'
```

Use the same thread_id to preserve short-term context.

## Important Notes

- Rebuild [faiss_index/](faiss_index) whenever [data/custom_faq.csv](data/custom_faq.csv) changes.
- [02_agentcore_memory.py](02_agentcore_memory.py) currently includes a hardcoded MEMORY_ID and region; update them for your AWS environment before production use.
- Never commit secrets such as .env files.

## Troubleshooting

- If faiss_index is missing, run: uv run python build_index.py
- If model calls fail, verify GROQ_API_KEY in .env
- If AgentCore commands fail, verify AWS credentials and region access

## License

Add your preferred license in this repository before broad public distribution.

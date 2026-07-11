# AI / ML Engineering Portfolio

> Eight end-to-end projects covering the modern AI engineering stack — RAG, multi-agent systems, MCP servers, production ML pipelines, forecasting, MLOps, agent evaluation, and LLM fine-tuning. Every project ships real, tested, runnable code: **309 passing tests**, offline-first demos, Dockerfiles, and CI.

[![CI](https://github.com/subratsushant05/project_ai_ml/actions/workflows/ci.yml/badge.svg)](https://github.com/subratsushant05/project_ai_ml/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hi, I'm **Subrat Sushant** — a Masters student in Information Technology & Analytics (AI/ML focus). I built this portfolio the way production systems are built: typed Python, pluggable interfaces, deterministic tests, honest metrics, and documentation that explains *why*, not just *what*.

📧 [subrat.sushant@gmail.com](mailto:subrat.sushant@gmail.com) · 💼 [LinkedIn](https://www.linkedin.com/in/subrat-sushant/) · 🐙 [GitHub](https://github.com/subratsushant05)

---

## Projects

| # | Project | What it demonstrates | Tests |
|---|---------|----------------------|-------|
| 01 | [RAG Knowledge Base](./01-rag-knowledge-base/) | Hybrid retrieval (BM25 + dense with reciprocal rank fusion), semantic chunking, citation-grounded answers, FastAPI service | 45 |
| 02 | [Multi-Agent Researcher](./02-multi-agent-researcher/) | LangGraph StateGraph with Planner → Researcher → Writer → Critic, bounded revision loops, human-in-the-loop `interrupt()`, checkpoint/resume | 32 |
| 03 | [MCP Analytics Server](./03-mcp-analytics-server/) | Custom Model Context Protocol server (FastMCP, stdio) with a three-layer read-only SQL guard — usable from Claude Desktop/Cursor | 56 |
| 04 | [Churn Prediction Pipeline](./04-churn-prediction-pipeline/) | sklearn/LightGBM pipeline, Optuna tuning, profit-maximizing decision thresholds, TreeSHAP explanations, model serving | 28 |
| 05 | [Time-Series Forecasting](./05-time-series-forecasting/) | Four models behind one interface, rolling-origin backtesting, MASE-based selection, split-conformal prediction intervals | 33 |
| 06 | [MLOps Pipeline](./06-mlops-pipeline/) | MLflow tracking, registry promotion policy with rollback + audit log, PSI/KS drift detection, Prometheus-instrumented serving | 33 |
| 07 | [Agent Eval Framework](./07-agent-eval-framework/) | Trajectory-level agent evaluation: tool-selection F1, call-order edit distance, efficiency, LLM-as-judge, CI gating | 26 |
| 08 | [QLoRA Fine-Tuning Toolkit](./08-llm-finetuning-qlora/) | Instruction-dataset engineering, chat templates from scratch, analytic LoRA/VRAM planning, TRL SFTTrainer training layer | 56 |

Every project is self-contained: its own README (architecture diagram, design decisions, real output), `requirements.txt`, `Dockerfile`, `Makefile`, and test suite. All demos run **fully offline** — no API keys needed — while OpenAI/Anthropic/Tavily backends plug in via environment variables.

---

## Engineering principles behind this repo

**Offline-first, provider-pluggable.** Every LLM, embedding, and search dependency sits behind a small interface with a deterministic offline implementation. Demos and tests run anywhere; swapping in GPT-4o or Claude is a config change, not a rewrite.

**Tested like production code.** 309 pytest tests across the repo — metric math verified against hand-computed cases, leakage checks in backtesting, SQL-injection guards exercised against every mutation class, promotion policies proven to reject worse models. CI runs lint + the full matrix on every push.

**Honest results.** Datasets are synthetic and documented as such; every number in every README comes from a real run of the committed code. No borrowed benchmarks, no fabricated metrics.

**Small, readable modules.** Typed Python 3.11, Google-style docstrings, pydantic v2 configs, files under ~300 lines, `ruff`-clean.

---

## Where to look, by role

| If you're hiring for | Start with |
|----------------------|------------|
| **LLM / AI Engineer** | 01 (RAG), 02 (agents), 03 (MCP), 08 (fine-tuning) |
| **ML Engineer** | 04 (churn pipeline), 05 (forecasting), 06 (MLOps) |
| **MLOps / Platform** | 06 (registry + drift), 03 (serving + guardrails), CI setup |
| **Data Scientist** | 05 (backtesting + conformal intervals), 04 (SHAP + business thresholds) |
| **AI Quality / Evals** | 07 (agent evals), 04 (model validation), 06 (drift detection) |

---

## Tech stack

```
Core:            Python 3.11, pydantic v2, FastAPI, pytest, ruff
LLM / Agents:    LangGraph, LangChain Core, MCP (Model Context Protocol),
                 OpenAI / Anthropic APIs (optional adapters), QLoRA (peft, TRL, bitsandbytes)
ML:              scikit-learn, LightGBM, Optuna, SHAP, statsmodels
MLOps:           MLflow, Prometheus, Docker, GitHub Actions, drift detection (PSI/KS)
Retrieval:       BM25 (rank_bm25), dense embeddings, reciprocal rank fusion, ChromaDB-ready
Serving:         FastAPI, uvicorn, Docker (non-root, slim images), docker-compose
```

## Running anything

```bash
git clone https://github.com/subratsushant05/project_ai_ml.git
cd project_ai_ml/01-rag-knowledge-base   # or any project
pip install -r requirements.txt
make test    # run the test suite
make demo    # end-to-end offline demo
```

---

## Contact

I'm actively looking for AI/ML engineering roles and always happy to talk through any design decision in this repo.

**Subrat Sushant** · [subrat.sushant@gmail.com](mailto:subrat.sushant@gmail.com) · [linkedin.com/in/subrat-sushant](https://www.linkedin.com/in/subrat-sushant/)

*Last updated: July 2026*

# Multi-Agent Research Assistant

A LangGraph pipeline where four specialized agents — **Planner**, **Researcher**,
**Writer**, and **Critic** — collaborate to turn a research question into a
structured, citation-backed markdown report. The Critic gates quality through a
bounded revision loop, a human approval gate can pause the graph mid-run, and
every provider has a deterministic offline default, so the whole system runs
end to end with **no API keys and no network access**.

## Architecture

```
                 +----------+      +------------+      +---------------+
  question ----> | Planner  | ---> | Researcher | ---> | Approval Gate |
                 +----------+      +------------+      +---------------+
                 sub-questions     findings with        interrupt() when
                                   [n] citations        approval required
                                                          |          |
                                                      rejected    approved
                                                          |          |
                                                          v          v
                                                        (END)   +--------+
                                                                | Writer |<--+
                                                                +--------+   |
                                                                     |       | revise
                                                                     v       | (bounded by
                                                                +--------+   |  max_revisions)
                                                                | Critic |---+
                                                                +--------+
                                                                     | score >= threshold
                                                                     | or budget exhausted
                                                                     v
                                                              report.md (END)
```

State flowing through the graph (`agent_researcher/state.py`):
`question`, `plan`, `findings` (additive reducer), `draft`, `critique`,
`revision_count`, `approved`.

## Features

- **Four-agent StateGraph** with conditional edges: an approval branch and a
  critic-driven revision loop that is provably bounded by `max_revisions`.
- **Offline-first providers.** `OfflineChatModel` is a deterministic rule-based
  model that builds plans, answers, drafts, and critiques from templates plus
  the retrieved evidence; `OfflineSearchTool` ranks a bundled 16-document JSON
  corpus with keyword scoring. Identical inputs always produce identical
  reports.
- **Pluggable hosted providers.** A `ChatModel` protocol plus an env-driven
  factory create `ChatOpenAI` / `ChatAnthropic` (and a Tavily search adapter)
  lazily, so none of those packages are required for the offline path.
- **Human-in-the-loop.** An optional approval gate implemented with LangGraph's
  `interrupt()`: the graph checkpoints, pauses before the Writer, and resumes
  on the same `thread_id` with `Command(resume=True|False)`.
- **Checkpointing** via `MemorySaver`; the demo shows a real pause/resume cycle.
- **Global citation registry.** Sources get report-wide `[n]` numbers that stay
  consistent between inline markers and the references section.

## Quickstart

Requires Python 3.11+.

```bash
pip install -r requirements.txt

# Run the full pipeline offline (no keys needed) and write report.md
python -m agent_researcher.demo "How do transformer models work?"

# Skip the human approval gate
python -m agent_researcher.demo "What limits electric vehicle adoption?" --no-approval

# Tests and lint
pip install pytest ruff
pytest
ruff check .

# Container
docker build -t agent-researcher .
docker run --rm agent-researcher
```

To use hosted providers instead, install the matching extra and set the
environment (see `.env.example`), e.g.
`pip install langchain-openai && export AGENT_MODEL_PROVIDER=openai`.

## Example output

Demo run (offline, deterministic):

```
$ python -m agent_researcher.demo "How do transformer models work?"
Question : How do transformer models work?
Thread   : demo
Providers: model=offline, search=offline

[planner] planned 3 sub-questions
    1. What are the core principles behind transformer models?
    2. What are the main real-world applications of transformer models?
    3. What are the key challenges and limitations of transformer models?
[researcher] gathered 3 findings from 9 sources
[approval_gate] interrupted -- awaiting human approval (checkpoint saved)
[approval_gate] demo auto-approves; resuming thread 'demo'
[approval_gate] approved
[writer] draft v1 written (2784 chars)
[critic] scored 7/10 (1 issue(s))
[writer] draft v2 written (3139 chars)
[critic] scored 10/10 (0 issue(s))

Final quality score : 10/10
Revisions performed : 1
Report written to   : report.md (3139 chars)
```

Excerpt from the generated `report.md`:

```markdown
# Research Report: How do transformer models work?

## Findings

### What are the core principles behind transformer models?

The core principle of transformer models is self-attention, which lets every
token weigh its relevance to every other token in a sequence [1]. ...

## References

[1] Attention Mechanisms in Transformer Models - https://example.org/ml/attention-mechanisms
```

## Configuration

All settings are environment variables with the `AGENT_` prefix
(`.env` files are supported; see `.env.example`).

| Variable                  | Default                   | Description                                      |
| ------------------------- | ------------------------- | ------------------------------------------------ |
| `AGENT_MODEL_PROVIDER`    | `offline`                 | Chat model: `offline`, `openai`, `anthropic`     |
| `AGENT_MODEL_NAME`        | provider default          | Model name override for hosted providers         |
| `AGENT_SEARCH_PROVIDER`   | `offline`                 | Search backend: `offline`, `tavily`              |
| `AGENT_CORPUS_PATH`       | `sample_data/corpus.json` | Corpus used by offline search                    |
| `AGENT_SEARCH_TOP_K`      | `3`                       | Sources retrieved per sub-question               |
| `AGENT_NUM_SUB_QUESTIONS` | `3`                       | Sub-questions the Planner produces               |
| `AGENT_QUALITY_THRESHOLD` | `8.0`                     | Minimum critic score (0-10) to accept a draft    |
| `AGENT_MAX_REVISIONS`     | `1`                       | Upper bound on Writer revisions                  |
| `AGENT_REQUIRE_APPROVAL`  | `false`                   | Pause at the approval gate before writing        |

## Project structure

```
agent_researcher/
  config.py       # pydantic-settings configuration (AGENT_* env vars)
  state.py        # typed graph state + Citation/Finding/Critique models
  prompts.py      # prompt templates shared by all providers
  llm.py          # ChatModel protocol, LangChain adapter, provider factory
  offline_llm.py  # deterministic rule-based model (default provider)
  search.py       # SearchTool protocol, offline corpus search, Tavily adapter
  nodes.py        # agent nodes, routing functions, LLM-output parsers
  graph.py        # StateGraph assembly and compilation
  demo.py         # CLI entry point (python -m agent_researcher.demo)
sample_data/      # 16-document offline search corpus
tests/            # 32 offline, deterministic pytest tests
```

## Design decisions

- **Why LangGraph.** The pipeline is a graph, not a chain: the approval gate
  branches, and the critic can send control *backwards* to the writer. LangGraph
  models this directly with conditional edges, gives checkpointing and
  `interrupt()` for free, and keeps every step a plain function of typed state,
  which makes nodes unit-testable in isolation.
- **Why a bounded revision loop.** Self-critique measurably improves drafts,
  but an unconstrained critic/writer cycle can oscillate forever and burn
  tokens. Routing on `score >= threshold or revision_count >= max_revisions`
  makes termination a property of the graph itself rather than of model
  behavior — the tests assert that a persistently bad writer is cut off after
  exactly one revision.
- **Why provider abstraction.** Agents depend on two tiny protocols
  (`ChatModel.invoke(system, user)`, `SearchTool.search(query, top_k)`) rather
  than on SDKs. The offline implementations make the demo and CI deterministic
  and free, hosted SDKs stay optional imports behind a factory, and swapping
  providers is a config change, not a refactor.
- **Why a global citation registry.** Citation numbers are assigned once per
  source URL across all sub-questions, so `[n]` markers remain stable when the
  writer reorders content and the references section can be verified against
  the findings — one of the test invariants.

## Testing

32 tests run offline in well under a second (`pytest -q`). Coverage includes:

- search ranking, top-k limits, determinism, and missing-corpus errors
- planner/critic output parsing (mixed bullet styles, score clamping, garbage)
- each node in isolation: citation numbering, revision counting, gate behavior
- graph end-to-end runs, the interrupt/resume cycle on a checkpointed thread,
  rejection routing, exactly-one-revision behavior, and reference/citation
  consistency in the final report

## Roadmap

- Parallel sub-question research with LangGraph `Send` fan-out
- Persistent checkpointing (SQLite/Postgres saver) for cross-process resume
- Retrieval upgrades for the offline tool (BM25, embeddings) behind the same protocol
- A small FastAPI service exposing runs, checkpoints, and approvals over HTTP
- Structured-output providers (JSON mode) to harden plan/critique parsing

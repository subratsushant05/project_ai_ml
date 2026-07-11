# agent-evals

A lightweight framework for evaluating LLM agent **trajectories** — not just final
answers. It scores what an agent *did* (which tools it called, in what order, at what
cost) alongside what it *said*, aggregates the results into JSON / Markdown / HTML
reports, and plugs into pytest so agent regressions fail CI.

Answer-only evals miss the most common agent failure modes: redundant tool loops,
wrong tool choices that happen to land near the right answer, and silent cost blowups.
This framework treats the full trajectory as the unit of evaluation.

## Architecture

```
                        +---------------------+
 datasets/basic.jsonl   |  Agent under test   |   any callable
 (TestCase: input,      |  (str -> Trajectory)|   input -> Trajectory
  expected_tools,       +----------+----------+
  expected_answer,                 |
  reference_trajectory)            v
        |               +---------------------+
        |               |     Trajectory      |  Steps: llm_call | tool_call
        |               |  (steps + tokens +  |         tool_result | final_answer
        |               |   timestamps)       |  each with tokens & timestamps
        |               +----------+----------+
        |                          |
        +----------->  +-----------------------+
                       |        Runner         |
                       +-----------+-----------+
                                   |  applies Metric protocol
        +----------------+--------+---------+----------------+----------------+
        v                v                  v                v                v
  ToolSelection     ToolCallOrder    TrajectoryEff.    AnswerCorrect.    CostLatency
  (set P/R/F1)      (edit distance)  (steps vs ref)    (exact/fuzzy      (price table,
        |                |                  |            + Judge)  <---+   percentiles)
        |                |                  |                |         |
        |                |                  |                |    Judge interface
        |                |                  |                |    (OfflineJudge default;
        +----------------+---------+--------+----------------+     OpenAI/Anthropic opt-in)
                                   v
                       +-----------------------+
                       |      EvalResult       |
                       |  per-case + aggregate |
                       +-----------+-----------+
                                   v
                 results.json | report.md | report.html
                 assert_agent_passes(...) for CI gating
```

## Features

- **Trajectory-level data model** (pydantic v2): typed `Step` events with timestamps,
  token counts, and model names — enough to reconstruct cost and latency after the fact.
- **Five composable metrics** behind a single `Metric` protocol; add your own by
  implementing `score(case, trajectory) -> MetricResult`.
- **LLM-as-judge behind an interface**: the default `OfflineJudge` is a deterministic
  rubric scorer (keyword coverage with a written rationale) so CI never needs an API
  key; OpenAI/Anthropic judges are lazy-imported and selected via `AGENT_EVALS_JUDGE`.
- **Two bundled reference agents** (`GoodAgent`, `SloppyAgent`) over a deterministic
  fake toolset (calculator, weather fixtures, fixture-backed web search) — useful as a
  known-good/known-bad pair for testing the harness itself.
- **Reports**: `results.json` (full per-case detail), `report.md` (paste into a PR),
  and a self-contained `report.html` with an inline-SVG comparison chart (light and
  dark mode, no external assets).
- **CI gating**: `assert_agent_passes(agent, dataset, thresholds)` turns any eval into
  a pytest assertion with a readable failure message.
- **Configurable pricing/budgets** via environment variables (`pydantic-settings`).

## Quickstart

```bash
git clone <your-fork-url> && cd 07-agent-eval-framework
pip install -r requirements.txt

# Evaluate both bundled agents on the bundled dataset, write reports/ :
python -m agent_evals.demo

# Evaluate your own agent(s) on your own dataset:
python -m agent_evals.run --dataset datasets/basic.jsonl \
    --agent agent_evals.agents:good_agent \
    --agent agent_evals.agents:sloppy_agent \
    --output reports

# Or in Docker:
docker build -t agent-evals . && docker run --rm agent-evals
```

An agent is any callable `str -> Trajectory`; `--agent` takes a `module:factory` spec
where `factory()` returns that callable.

```python
from agent_evals import Dataset, Runner

runner = Runner()
result = runner.evaluate(my_agent, Dataset.from_jsonl("datasets/basic.jsonl"))
print(result.mean_scores, result.total_cost_usd)
```

## Metric reference

All scores are normalized to `[0, 1]`, higher is better. `E` is the expected/reference
tool sequence, `A` the actual one.

| Metric | Question it answers | Formula |
|---|---|---|
| `tool_selection` | Right tools called? | F1 of set(A) vs set(E): `2PR / (P + R)` with `P = |E∩A| / |A|`, `R = |E∩A| / |E|` |
| `tool_order` | Right order? | `1 − levenshtein(E, A) / max(|E|, |A|)` |
| `efficiency` | No wandering? | `min(1, |E| / |A|)`; exact duplicate calls flagged in details |
| `answer_correctness` | Right answer? | 1.0 on exact match; else fuzzy ratio, blended 50/50 with the judge score when a judge is set |
| `cost_latency` | Cheap and fast? | mean of `exp(−cost/budget_c)` and `exp(−latency/budget_t)`; budgets configurable |

The `cost_latency` details carry the raw numbers (USD, tokens, wall seconds); the
aggregate report adds p50/p95 latency and total cost per agent.

## Example output

Actual output of `python -m agent_evals.demo` on the bundled 10-case dataset:

| Metric | GoodAgent | SloppyAgent |
|---|---|---|
| Tool selection (F1) | 1.000 | 0.793 |
| Tool call order | 1.000 | 0.427 |
| Trajectory efficiency | 1.000 | 0.427 |
| Answer correctness | 1.000 | 0.507 |
| Cost / latency | 0.937 | 0.391 |
| **Overall score** | **0.987** | **0.509** |
| Total cost (USD) | 0.0012 | 0.0825 |
| Total tokens | 5000 | 20100 |
| Latency p50 (s) | 0.50 | 2.65 |
| Latency p95 (s) | 0.72 | 3.08 |

`SloppyAgent` reaches roughly the right answers, but the trajectory metrics expose
*how*: a redundant `web_search` before every task, duplicated tool calls, a pricier
model, and hedged answers. That gap is invisible to answer-only evaluation.

## Using in CI

Gate your agent on minimum mean scores; the assertion message lists every failing
metric. From `tests/test_runner_and_agents.py`:

```python
from agent_evals import Dataset, assert_agent_passes
from agent_evals.agents import GoodAgent, SloppyAgent

THRESHOLDS = {
    "tool_selection": 0.95,
    "tool_order": 0.95,
    "efficiency": 0.95,
    "answer_correctness": 0.9,
    "cost_latency": 0.85,
}

def test_agent_meets_quality_bar():
    dataset = Dataset.from_jsonl("datasets/basic.jsonl")
    assert_agent_passes(GoodAgent(), dataset, THRESHOLDS)   # passes

def test_regression_is_caught():
    dataset = Dataset.from_jsonl("datasets/basic.jsonl")
    with pytest.raises(AssertionError):
        assert_agent_passes(SloppyAgent(), dataset, THRESHOLDS)
```

Because the default judge is offline and the bundled agents are deterministic, the
whole suite runs in well under a second with no network and no API keys.

## Project structure

```
agent_evals/
  models.py        # Step, Trajectory, TestCase, Dataset, EvalResult (pydantic v2)
  metrics.py       # Metric protocol + the five bundled metrics
  judge.py         # Judge interface, OfflineJudge, lazy OpenAI/Anthropic judges
  tools.py         # deterministic fake toolset + fixtures
  agents.py        # GoodAgent / SloppyAgent reference agents
  runner.py        # Runner.evaluate / Runner.compare, percentile aggregation
  reporting.py     # results.json + Markdown report
  html_report.py   # self-contained HTML report with inline-SVG chart
  testing.py       # assert_agent_passes for CI
  settings.py      # env-configurable pricing, budgets, judge selection
  demo.py, run.py  # CLI entry points
datasets/basic.jsonl   # 10 bundled test cases
tests/                 # 26 fast, offline, deterministic tests
```

## Design decisions

- **Trajectory-level evals, not answer-level.** Two agents with identical answers can
  differ 10x in cost and reliability. Scoring the tool sequence, step count, and spend
  catches regressions (loops, redundant calls, model drift) that answer checks cannot,
  and the per-metric breakdown tells you *which* behavior regressed.
- **Offline judge by default.** LLM judges are non-deterministic, cost money, and make
  CI flaky. The default judge is a transparent keyword-coverage rubric that returns a
  rationale you can read in the report; remote judges are one env var away
  (`AGENT_EVALS_JUDGE=openai|anthropic`) and are never imported unless selected.
- **Agent = `Callable[[str], Trajectory]`.** No base class to inherit, no framework
  lock-in: wrap a LangChain/LlamaIndex/handwritten agent by recording its events into
  `Step` objects. The bundled agents double as fixtures for testing the harness.
- **Metrics as small classes behind a protocol.** Each metric is independently
  unit-tested against hand-computed cases and returns raw details next to its score,
  so a surprising number is always explainable.
- **Smooth cost/latency scoring** (`exp(-x/budget)`) instead of a hard pass/fail:
  cheaper is *always* better, ties are impossible, and thresholds stay meaningful as
  budgets change.

## Testing

```bash
pip install pytest ruff
pytest -q          # 26 tests, < 1s, no network
ruff check .       # lint (config in ruff.toml)
```

Tests cover each metric against hand-computed toy cases (edit distances, P/R/F1,
efficiency ratios, pricing math), JSONL round-trips, offline-judge determinism,
runner aggregation, report generation, both CLIs end-to-end, and the CI helper's
pass/fail behavior on the bundled agent pair.

## Roadmap

- LangSmith / LangFuse trace export and import
- OpenTelemetry (GenAI semantic conventions) trajectory ingestion
- Pairwise LLM-judge comparisons (A/B verdicts instead of absolute scores)
- Statistical significance over repeated stochastic runs (bootstrap CIs)
- Tool-argument correctness (schema-aware comparison, not just tool names)
- pytest plugin with `--agent-eval` flags and JUnit-style report attachment

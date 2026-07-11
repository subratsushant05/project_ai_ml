# Agent evaluation report

Dataset: `basic` | Agents: GoodAgent, SloppyAgent | Cases per agent: 10

## Side-by-side comparison

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

## Per-case results: GoodAgent

| Case | Tools called | Tool selection (F1) | Tool call order | Trajectory efficiency | Answer correctness | Cost / latency |
|---|---|---|---|---|---|---|
| c01 | calculator | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c02 | calculator | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c03 | weather | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c04 | weather | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c05 | weather | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c06 | web_search | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c07 | web_search | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c08 | web_search | 1.000 | 1.000 | 1.000 | 1.000 | 0.941 |
| c09 | weather, calculator | 1.000 | 1.000 | 1.000 | 1.000 | 0.928 |
| c10 | weather, weather, calculator | 1.000 | 1.000 | 1.000 | 1.000 | 0.916 |

## Per-case results: SloppyAgent

| Case | Tools called | Tool selection (F1) | Tool call order | Trajectory efficiency | Answer correctness | Cost / latency |
|---|---|---|---|---|---|---|
| c01 | web_search, calculator, calculator | 0.667 | 0.333 | 0.333 | 0.458 | 0.390 |
| c02 | web_search, calculator, calculator | 0.667 | 0.333 | 0.333 | 0.412 | 0.390 |
| c03 | web_search, weather, weather | 0.667 | 0.333 | 0.333 | 0.498 | 0.390 |
| c04 | web_search, weather, weather | 0.667 | 0.333 | 0.333 | 0.371 | 0.390 |
| c05 | web_search, weather, weather | 0.667 | 0.333 | 0.333 | 0.372 | 0.390 |
| c06 | web_search, web_search | 1.000 | 0.500 | 0.500 | 0.624 | 0.407 |
| c07 | web_search, web_search | 1.000 | 0.500 | 0.500 | 0.581 | 0.407 |
| c08 | web_search, web_search | 1.000 | 0.500 | 0.500 | 0.481 | 0.407 |
| c09 | web_search, weather, weather, calculator | 0.800 | 0.500 | 0.500 | 0.674 | 0.374 |
| c10 | web_search, weather, weather, weather, calculator | 0.800 | 0.600 | 0.600 | 0.600 | 0.359 |

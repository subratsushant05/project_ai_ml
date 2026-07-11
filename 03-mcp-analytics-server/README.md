# MCP Analytics Server

A custom [Model Context Protocol](https://modelcontextprotocol.io) server that gives any MCP client — Claude Desktop, Cursor, or your own agent — safe, read-only SQL analytics over a bundled SQLite database. The server ships with a deterministic synthetic e-commerce dataset and exposes schema inspection, guarded querying, column statistics, and chart rendering as MCP tools.

The headline feature is the **layered read-only guard**: every query passes a static SQL validator, a deny-by-default SQLite authorizer callback, and a connection physically opened in read-only mode — so an LLM (or a prompt-injected one) cannot mutate the database no matter what SQL it produces.

## Architecture

```
+----------------------+        JSON-RPC over stdio        +----------------------------+
|      MCP client      | <-------------------------------> |    mcp_analytics server    |
|  (Claude Desktop,    |    initialize / tools / calls     |         (FastMCP)          |
|   Cursor, demo.py)   |                                   |                            |
+----------------------+                                   |  tools:                    |
                                                           |   list_tables              |
                                                           |   describe_table           |
                                                           |   run_query   ---+         |
                                                           |   table_stats    |         |
                                                           |   plot_data      |         |
                                                           |  resource:       |         |
                                                           |   schema://database        |
                                                           +------------------|---------+
                                                                              v
                                                           +----------------------------+
                                                           |  guards.validate_select    |  layer 1: static SQL validation
                                                           |  sqlite authorizer (deny)  |  layer 2: engine-level action filter
                                                           |  file:...?mode=ro          |  layer 3: read-only connection
                                                           +------------------|---------+
                                                                              v
                                                           +----------------------------+
                                                           |   SQLite: ecommerce.db     |
                                                           |   customers / products /   |
                                                           |   orders / order_items     |
                                                           +----------------------------+
```

## Features

- **Five analytics tools** — schema overview, table description, guarded SQL, column statistics, matplotlib charts (PNG, headless Agg backend).
- **MCP resource** — the full database DDL exposed at `schema://database`.
- **Three-layer write protection** — static validator + SQLite authorizer + `mode=ro` connection (see [Security](#security)).
- **Bounded execution** — configurable row cap (default 200) and wall-clock query timeout (default 5 s) enforced via a SQLite progress handler.
- **Deterministic sample data** — a seeded generator produces the same ~590-row e-commerce dataset on every machine; the server auto-seeds on first run.
- **Proven end-to-end** — `python -m mcp_analytics.demo` spawns the server over stdio with the official MCP client library and calls every tool.
- **56 fast, offline tests** and a clean `ruff` bill of health.

## Quickstart

Requires Python 3.11+.

```bash
git clone <your-fork-url>
cd 03-mcp-analytics-server
pip install -r requirements.txt

# Optional -- the server seeds automatically on first run:
python -m mcp_analytics.seed --force

# Prove it works end-to-end (spawns the server over stdio):
python -m mcp_analytics.demo

# Or run the server directly (for an MCP client to connect to):
python -m mcp_analytics.server
```

### Claude Desktop

Add this to `claude_desktop_config.json` (Settings → Developer → Edit Config), then restart Claude Desktop:

```json
{
  "mcpServers": {
    "analytics": {
      "command": "python",
      "args": ["-m", "mcp_analytics.server"],
      "cwd": "/absolute/path/to/03-mcp-analytics-server"
    }
  }
}
```

Then ask Claude things like *"Which product category generates the most revenue? Plot it."* — it will chain `list_tables` → `run_query` → `plot_data` on its own.

### Cursor / other MCP clients

Any stdio-capable MCP client works with the same command. For Cursor, add the equivalent entry to `.cursor/mcp.json`.

### Docker

```bash
make docker-build
make docker-run     # runs with -i so the stdio transport stays open
```

## Tool reference

| Tool | Arguments | Returns |
| --- | --- | --- |
| `list_tables()` | — | Markdown table of every table with column and row counts |
| `describe_table(table)` | table name | Column names/types/constraints plus 5 sample rows |
| `run_query(sql)` | one `SELECT` statement | Result set as a markdown table (row-capped, time-boxed) |
| `table_stats(table, column)` | table + column | count, nulls, distinct, min/max, mean/std (numeric), top-5 values |
| `plot_data(sql, chart_type, x, y)` | query + `bar`/`line`/`scatter` + axis columns | Path of a rendered PNG chart |

Resource: `schema://database` — full `CREATE TABLE` / `CREATE INDEX` DDL as SQL text.

Configuration (environment variables): `MCP_ANALYTICS_DB` (database path), `MCP_ANALYTICS_CHART_DIR` (chart output dir), `MCP_ANALYTICS_MAX_ROWS` (row cap), `MCP_ANALYTICS_TIMEOUT_SECONDS` (query budget).

## Security

LLM-generated SQL is untrusted input. The server defends in depth — each layer is sufficient on its own, and all three are always active for user SQL:

1. **Static validation** (`guards.py`). The statement is scanned character-by-character; string literals, quoted identifiers, and comments are blanked out first, so `SELECT 'drop table'` passes while `DROP TABLE` hidden behind a comment does not. The validator then requires a single statement (no `;` chaining) beginning with `SELECT`/`WITH`/`VALUES` and rejects any mutating or administrative keyword (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, `PRAGMA`, `ATTACH`, `VACUUM`, transaction control, ...).
2. **SQLite authorizer** (`db.py`). A deny-by-default authorizer callback permits only `SQLITE_SELECT`, `SQLITE_READ`, `SQLITE_FUNCTION`, and `SQLITE_RECURSIVE` actions, so even a statement that slipped past layer 1 is refused by the engine itself.
3. **Read-only connection**. User SQL runs on a connection opened with `file:...?mode=ro`, so the OS-level handle cannot write to the file at all.

Resource bounds: results are capped at `MAX_ROWS` (fetching cap+1 to detect truncation honestly), and a progress handler aborts any statement that exceeds the wall-clock budget — including runaway recursive CTEs. Tools that interpolate identifiers (`describe_table`, `table_stats`) validate names against the live schema first, so identifier injection is impossible.

## Example output

Excerpt from `python -m mcp_analytics.demo`:

```
>> run_query(revenue by country)
------------------------------------------------------------------------
| country | orders | revenue |
| --- | --- | --- |
| India | 31 | 32,496.18 |
| Japan | 21 | 25,736.76 |
| UK | 24 | 25,035.56 |
...

>> run_query(DROP TABLE ...) -- expected to be BLOCKED
------------------------------------------------------------------------
Error executing tool run_query: Only read-only SELECT statements are allowed
(must start with SELECT, WITH, or VALUES).

>> table_stats("products", "price")
------------------------------------------------------------------------
### Statistics for `products.price`
- rows: 40  |  non-null: 40  |  nulls: 0  |  distinct: 40
- min: 27.57  |  max: 384.5
- mean: 202.3852  |  std (population): 118.7298

>> plot_data(avg price by category)
------------------------------------------------------------------------
Chart saved: /tmp/mcp_analytics_charts/chart_bar_bc2f859f63.png (bar, 5 points)
```

## Project structure

```
03-mcp-analytics-server/
├── mcp_analytics/
│   ├── server.py      # FastMCP app: 5 tools + schema resource + stdio entrypoint
│   ├── db.py          # read-only access layer: authorizer, row cap, timeout
│   ├── guards.py      # SELECT-only static SQL validator
│   ├── stats.py       # column statistics computed inside SQLite
│   ├── plotting.py    # matplotlib (Agg) chart rendering
│   ├── render.py      # markdown table formatting
│   ├── seed.py        # deterministic synthetic dataset generator (CLI)
│   ├── demo.py        # stdio MCP client exercising every tool (CLI)
│   └── config.py      # pydantic settings with environment overrides
├── tests/             # 56 tests: guards, db layer, seed, tools, stdio integration
├── requirements.txt   # mcp, matplotlib, pydantic
├── Dockerfile         # python:3.11-slim, non-root user
├── Makefile
└── README.md
```

## Design decisions

- **FastMCP over the low-level SDK** — tool schemas are derived from type hints and docstrings, keeping the server declarative and the tool contract in one place.
- **Deny-list *and* allow-list** — the validator allow-lists the leading keyword and deny-lists dangerous keywords anywhere; the authorizer is a pure allow-list of read actions. Disagreements between layers fail closed.
- **Markdown tool output** — LLMs consume markdown tables far more reliably than JSON blobs, and results render nicely in Claude Desktop.
- **Charts as file paths, not base64** — keeps tool responses small; clients that can display images can read the PNG, and the path is stable for follow-up use.
- **Uncached settings** — configuration is re-read from the environment per call, which makes the test suite trivially able to repoint the server at temporary databases.
- **Trusted vs. guarded connections** — internal schema introspection needs `PRAGMA table_info`, which our own authorizer would deny; introspection therefore uses a separate connection that skips the authorizer but is still `mode=ro`. User SQL never touches that path.
- **Deterministic seed** — a fixed RNG seed makes tests exact and lets the README show real, reproducible numbers.

## Testing

```bash
pip install pytest ruff
make test    # 56 tests, ~3 s, fully offline
make lint    # ruff check .
```

Coverage highlights: seed determinism (byte-identical dumps), referential integrity, every guard rejection class (mutations, multi-statement, comment smuggling, case tricks), row cap and timeout enforcement, authorizer behavior with the validator bypassed, statistics checked against `statistics.pstdev`, PNG magic-byte verification, and one full stdio integration test that runs the real demo client against the real server.

## Roadmap

- `EXPLAIN QUERY PLAN` tool for query optimization advice
- Optional streamable-HTTP transport for remote deployment
- Support pointing the server at any user-supplied SQLite file (same guard stack)
- Correlation / group-by comparison statistics tool
- MCP prompts with canned analysis workflows (e.g. "revenue deep-dive")

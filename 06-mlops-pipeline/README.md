# MLOps Pipeline: Gated Promotion, Drift Detection, Monitored Serving

An end-to-end MLOps system built around a deliberately simple ML task (loan-default
prediction on synthetic tabular data) so the operational machinery is the star:
schema-validated ingestion, MLflow experiment tracking, a model registry with
policy-gated promotion and rollback, statistical drift detection that triggers
retraining, and a FastAPI serving layer instrumented with Prometheus metrics.

The whole loop is reproducible locally with two commands and no external services.

## Architecture

```
                        ┌──────────────────────────────────────────────────┐
                        │        run_pipeline (orchestrator)               │
                        │                                                  │
  synthetic data ──────►│  ┌──────────┐   ┌─────────┐   ┌──────────────┐   │
  (deterministic)       │  │ validate │──►│  train   │──►│  evaluate    │   │
                        │  │ (schema  │   │ (MLflow  │   │  incumbent   │   │
        halt on fail ◄──┤  │  gate)   │   │ tracking)│   │ on same split│   │
                        │  └──────────┘   └─────────┘   └──────┬───────┘   │
                        └───────────────────────────────────────┼──────────┘
                                                                ▼
                     ┌─────────────────┐  promote if AUC beats prod + margin
                     │  Model Registry │◄───── promotion policy ────────────
                     │ (MLflow aliases │       + audit log (JSONL)
                     │  + rollback)    │
                     └────────┬────────┘
                              │ models:/loan-default-classifier@production
                              ▼
   clients ──POST /predict──► ┌──────────────────┐ ──► /metrics (Prometheus)
                              │  FastAPI serving │ ──► /health, /model/info
                              └────────┬─────────┘
                                       │ incoming feature windows
                                       ▼
                              ┌──────────────────┐   should_retrain()
                              │  drift detection │ ────────────────────┐
                              │   (PSI + KS)     │                     │
                              └──────────────────┘                     │
                                       ▲                               │
                                       └──────── retraining loop ◄─────┘
```

## Features

- **Data validation gate** - pydantic-backed schema checks (dtype families, value
  ranges, null-fraction thresholds, category domains) producing a JSON report;
  the pipeline halts with a non-zero exit code on failure.
- **Experiment tracking** - every training round logs params, metrics, the fitted
  sklearn pipeline, and the data-schema hash to MLflow (SQLite backend, so
  registry aliases work without a server).
- **Gated promotion** - a candidate takes the `production` alias only if its
  holdout AUC beats the incumbent *evaluated on the same holdout* by a
  configurable margin, and it clears absolute guardrails (AUC floor, Brier cap).
- **Rollback + audit** - one-call rollback to the previous production version;
  every promote/reject/rollback lands in `artifacts/registry_audit.jsonl`.
- **Drift detection** - PSI (quantile-binned, epsilon-smoothed) and two-sample
  KS tests per feature, hand-implemented and unit-tested against hand-computed
  values, with warn/alert severities and a `should_retrain()` decision.
- **Monitored serving** - FastAPI app that loads the production model from the
  registry at startup; Prometheus request counts, latency histograms, and the
  served prediction distribution on `/metrics`.
- **Retraining simulation** - one command that shows the full loop: stable
  window (no action), shifted window (alerts fire), retrain, and promotion only
  because the new model genuinely wins.

## Quickstart

```bash
cd 06-mlops-pipeline
pip install -r requirements.txt

# 1. Train, compare candidates, and promote the first production model
python -m mlops_pipeline.run_pipeline

# 2. Inspect runs and the registry in the MLflow UI (http://localhost:5000)
mlflow ui --backend-store-uri sqlite:///mlruns.db --port 5000

# 3. Serve the production model (http://localhost:8000/docs)
uvicorn mlops_pipeline.serving.app:app --port 8000

# 4. Score an application
curl -s -X POST localhost:8000/predict -H 'Content-Type: application/json' -d '{
  "age": 34, "income": 55000, "loan_amount": 12000, "credit_score": 700,
  "debt_to_income": 0.25, "num_prior_defaults": 0,
  "employment_status": "employed", "loan_term_months": 36}'
# {"default_probability":0.259902,"prediction":"no_default","model_version":2}

# 5. Watch drift trigger a retrain (see walkthrough below)
python -m mlops_pipeline.simulate_drift
```

Or with Docker: `docker compose up --build` (add `--profile mlflow` for the UI
on port 5000).

## Drift simulation walkthrough

`python -m mlops_pipeline.simulate_drift` scores two incoming windows against
the training reference. The first is drawn from the same distribution; the
second simulates a credit-cycle downturn (incomes sag, credit scores slip and
decouple from risk, leverage becomes more predictive). Output from a real run:

```
production model: logistic_regression v1 (training auc=0.7936)

--- Drift check: stable window (1000 rows) ---
feature                   PSI   KS stat   KS crit  severity
age                    0.0193    0.0213    0.0480  NONE
income                 0.0109    0.0420    0.0480  NONE
loan_amount            0.0142    0.0343    0.0480  NONE
credit_score           0.0134    0.0418    0.0480  NONE
debt_to_income         0.0114    0.0330    0.0480  NONE
num_prior_defaults     0.0002    0.0103    0.0480  NONE
employment_status      0.0004         -         -  NONE
loan_term_months       0.0027         -         -  NONE
alerts: 0/8 features (retrain threshold: 2)
should_retrain() -> False

--- Drift check: shifted window (1000 rows) ---
feature                   PSI   KS stat   KS crit  severity
age                    0.0129    0.0195    0.0480  NONE
income                 0.1985    0.1890    0.0480  WARN
loan_amount            0.0154    0.0387    0.0480  NONE
credit_score           0.5016    0.2830    0.0480  ALERT
debt_to_income         0.2671    0.1673    0.0480  ALERT
num_prior_defaults     0.0054    0.0338    0.0480  NONE
employment_status      0.0039         -         -  NONE
loan_term_months       0.0046         -         -  NONE
alerts: 2/8 features (retrain threshold: 2)
should_retrain() -> True

--- Retraining triggered: fitting on recent labelled traffic ---
retrained best : logistic_regression
candidate auc  : 0.7891 (new holdout)
incumbent auc  : 0.7596 (same holdout)
outcome        : PROMOTED as version 2
reasons        : candidate auc 0.7891 beats production 0.7596 + margin 0.005
```

Re-running the training pipeline afterwards demonstrates idempotency: an
equivalent candidate is **rejected** because it cannot beat production by the
margin, and the rejection is written to the audit log.

## Promotion policy

A candidate replaces the production model only when *all* of these hold:

| Rule | Default | Setting |
|---|---|---|
| Holdout AUC beats incumbent AUC + margin | margin = 0.005 | `MLOPS_PROMOTION_MARGIN` |
| Absolute AUC floor | 0.70 | `MLOPS_MIN_AUC` |
| Calibration guardrail (Brier score cap) | 0.20 | `MLOPS_MAX_BRIER` |

Two design details worth calling out:

1. **Same-holdout comparison.** The incumbent is re-evaluated on the *current*
   holdout split before the decision, so the comparison is apples-to-apples
   even after the data distribution moved (training-time metrics of the old
   model would be stale and misleading).
2. **First promotion still has guardrails.** With no incumbent, the margin rule
   is waived but the AUC floor and Brier cap still apply - an empty registry is
   never an excuse to ship a bad model.

Every decision (promote / reject / rollback) is appended to
`artifacts/registry_audit.jsonl` with timestamp, versions, metrics, and reason.

## Project structure

```
06-mlops-pipeline/
├── mlops_pipeline/
│   ├── config.py            # pydantic-settings config (MLOPS_* env overrides)
│   ├── data.py              # deterministic synthetic generator + drift knobs
│   ├── validation.py        # schema gate, validation report, schema hash
│   ├── training.py          # candidate comparison + MLflow tracking
│   ├── registry.py          # promotion policy, alias registry, rollback, audit
│   ├── drift.py             # PSI / KS math, severities, should_retrain()
│   ├── run_pipeline.py      # orchestrator: validate -> train -> gate
│   ├── simulate_drift.py    # drift-to-retrain walkthrough CLI
│   └── serving/app.py       # FastAPI + Prometheus serving layer
├── tests/                   # 33 tests: math, gates, registry, API, e2e
├── .github-workflow-example.yml  # how CI would gate promotion (reference)
├── Dockerfile               # python:3.11-slim, non-root
├── docker-compose.yml       # api + optional mlflow-ui service
├── Makefile
└── requirements.txt
```

## Design decisions

- **SQLite MLflow backend instead of `./mlruns` files.** MLflow's model
  registry (and therefore aliases) requires a database-backed store; SQLite
  keeps the project fully local and serverless while exercising the same
  registry APIs a Postgres-backed deployment would use.
- **Aliases over deprecated registry stages.** `@production` aliases are
  MLflow's current recommendation; promotion is one atomic alias move, and
  rollback is the same move in reverse.
- **Hand-rolled schema validation on pydantic.** A small, explicit validator
  keeps the dependency surface lean and makes the validation report format a
  first-class, versioned contract (the schema hash is logged with every run).
- **Hand-implemented PSI/KS.** ~60 lines of numpy that can be unit-tested
  against hand-computed values, instead of importing a monitoring platform for
  two statistics.
- **Policy as a pure function.** `promotion_decision()` takes metrics and a
  policy and returns a decision with reasons - trivially testable, and the
  registry class just executes what it says.
- **Concept + covariate drift in the simulator.** A pure covariate shift often
  leaves a well-specified model's ranking intact (retraining would be rightly
  rejected); the simulated downturn also changes the feature-outcome
  relationship, so the retrain wins on merit rather than by construction.

## Testing

```bash
pip install pytest
pytest tests/ -v        # 33 tests, ~20s
ruff check .            # lint-clean
```

Coverage highlights: schema pass/fail per check type, PSI and KS verified
against hand-computed values, retrain-threshold behaviour, promotion margin and
guardrail edge cases, registry promote/rollback/audit against a real MLflow
store, all API endpoints via `TestClient`, and the pipeline end-to-end in a
temp directory.

## Roadmap

- Shadow deployment: score with candidate and production side by side and
  compare live before promotion.
- Prediction logging with delayed-label joins to monitor realized AUC, not just
  input drift.
- Postgres-backed MLflow server + S3 artifact store deployment profile.
- Grafana dashboard JSON for the exported Prometheus metrics.
- Champion/challenger A/B routing in the serving layer.

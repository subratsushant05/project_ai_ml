# qlora_tune

A QLoRA fine-tuning toolkit for instruction-tuning small LLMs on domain data,
demonstrated on IT-helpdesk tickets. The repo is built around a strict
**light/heavy split**: the data pipeline, config validation, dry-run planning
and evaluation all run on a CPU with five small dependencies, while the actual
training code (torch / transformers / peft / bitsandbytes / trl) is real,
typed, lazily-imported code that activates only on a GPU machine.

**Honesty note:** this repository demonstrates the *engineering* of a QLoRA
pipeline — dataset construction, config discipline, analytic resource
planning, and evaluation tooling. No 8B model was trained to produce the
numbers in this README. The bundled `sample_data/sample_predictions.jsonl` is
an illustrative fixture that exists to demonstrate the evaluation harness and
report format, not a claim of measured model quality.

## Architecture

```
            LIGHT LAYER (CPU, 5 deps)                 HEAVY LAYER (GPU)
 ┌──────────────────────────────────────────┐   ┌─────────────────────────┐
 │  data/generator ──> data/cleaning        │   │  train.py               │
 │   (300 synthetic     (PII scrub, dedupe, │   │   BitsAndBytesConfig    │
 │    tickets)           length filter)     │   │   LoraConfig            │
 │        │                   │             │   │   SFTTrainer (TRL)      │
 │        v                   v             │   │        │                │
 │  data/splitting ──> data/templates ──────┼──>│        v                │
 │   (stratified        (Llama-3 / ChatML,  │   │  merge.py               │
 │    80/10/10)          hand-written)      │   │   merge_and_unload      │
 │                                          │   │   -> GGUF (llama.cpp)   │
 │  config.py (pydantic v2 + YAML)          │   └─────────────────────────┘
 │        │                                 │      imports are lazy: the
 │        v                                 │      light layer never loads
 │  planning.py ── analytic LoRA params     │      torch/transformers
 │        │        + VRAM estimate          │
 │        v                                 │
 │  evaluation/ ── ROUGE-L (from scratch),  │
 │                 exact match, keyword     │
 │                 coverage, base-vs-tuned  │
 │                 report                   │
 └──────────────────────────────────────────┘
```

## Features

- **Deterministic synthetic dataset**: 300 helpdesk ticket -> resolution pairs
  across 5 categories (password reset, VPN, hardware, software install,
  access requests), fully reproducible from a seed.
- **Cleaning pipeline**: regex PII scrubbing (emails, phones), normalized
  deduplication, length filtering — with a structured `CleaningReport`.
- **Stratified splitting**: per-category 80/10/10 train/val/test.
- **Hand-written chat templates**: Llama-3 and ChatML formats implemented
  explicitly and tested against exact expected strings — no transformers
  dependency just to format text.
- **Validated configs**: pydantic v2 schema with real rules (LoRA rank
  whitelist, known target modules, `alpha = 2*r` convention warning,
  `extra="forbid"` against typos) and YAML round-tripping.
- **Dry-run planning without a model**: LoRA trainable-parameter counts are
  computed analytically from a table of published architecture dimensions,
  plus a documented first-principles VRAM estimate. Runs without torch.
- **From-scratch evaluation**: LCS-based ROUGE-L, normalized exact match, and
  keyword coverage; a harness that scores base vs fine-tuned predictions from
  a JSONL file and renders a delta report.
- **Real training code**: `train.py` builds the 4-bit quantized model with
  `BitsAndBytesConfig`, attaches `LoraConfig` adapters, and runs TRL's
  `SFTTrainer`; `merge.py` merges adapters in 16-bit for export.

## Quickstart

### Light path (any machine, no GPU)

```bash
pip install -r requirements.txt
python -m qlora_tune.demo                 # full pipeline + both reports
python -m qlora_tune.train --config configs/llama-3.1-8b.yaml --dry-run
pip install pytest && python -m pytest    # 56 tests, ~2 s, offline
```

### Real training (requires an NVIDIA GPU)

```bash
pip install -r requirements.txt -r requirements-train.txt
python -m qlora_tune.demo                 # writes demo_output/{train,val,test}.jsonl

# smoke test first (0.5B model), then the 8B run
python -m qlora_tune.train --config configs/qwen2.5-0.5b-smoke-test.yaml \
    --train-file demo_output/train.jsonl --val-file demo_output/val.jsonl
python -m qlora_tune.train --config configs/llama-3.1-8b.yaml \
    --train-file demo_output/train.jsonl --val-file demo_output/val.jsonl

# merge the adapter into a plain checkpoint
python -m qlora_tune.merge --base meta-llama/Llama-3.1-8B-Instruct \
    --adapter outputs/llama31-8b-helpdesk --out outputs/merged
```

Gated models (Llama) additionally need `HF_TOKEN` with accepted license
terms. `Dockerfile` packages the light demo; `Dockerfile.train` documents a
CUDA 12.4 training image.

### GGUF export (llama.cpp)

After merging, convert with llama.cpp's converter:

```bash
git clone https://github.com/ggerganov/llama.cpp
pip install -r llama.cpp/requirements/requirements-convert_hf_to_gguf.txt
python llama.cpp/convert_hf_to_gguf.py outputs/merged --outfile helpdesk-8b.gguf
llama.cpp/build/bin/llama-quantize helpdesk-8b.gguf helpdesk-8b-q4_k_m.gguf q4_k_m
```

## Hardware requirements

Estimates come from the dry-run math in `qlora_tune/planning.py`
(QLoRA 4-bit NF4, r=16, all seven projection modules, seq 2048, micro-batch 1,
gradient checkpointing; treat as +/-20%):

| Base model                  | Trainable params | Est. VRAM | Fits on          |
|-----------------------------|-----------------:|----------:|------------------|
| Qwen2.5-0.5B-Instruct       |            8.8 M |  3.3 GiB  | any 6 GB GPU     |
| Llama-3.2-1B-Instruct       |           11.3 M |  3.5 GiB  | any 6 GB GPU     |
| Llama-3.2-3B-Instruct       |           24.3 M |  5.0 GiB  | 8 GB (3070/T4)   |
| Mistral-7B-Instruct-v0.3    |           41.9 M |  6.5 GiB  | 12 GB (3060)     |
| Qwen2.5-7B-Instruct         |           40.4 M |  7.8 GiB  | 12 GB (3060)     |
| Llama-3.1-8B-Instruct       |           41.9 M |  8.0 GiB  | 12 GB (3060)     |

The light layer (demo, tests, planning, evaluation) needs no GPU at all.

## Chat templates

Templates are written out explicitly in `qlora_tune/data/templates.py` so the
exact training string is visible and unit-tested — a subtle template mismatch
between training and inference is one of the most common ways fine-tunes
silently underperform.

Llama-3 format:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant}<|eot_id|>
```

ChatML (Qwen-style) format:

```
<|im_start|>system
{system}<|im_end|>
<|im_start|>user
{user}<|im_end|>
<|im_start|>assistant
{assistant}<|im_end|>
```

## Example dry-run output

Actual output of
`python -m qlora_tune.train --config configs/llama-3.1-8b.yaml --dry-run`
on a machine with only the light requirements installed:

```
==============================================================
QLoRA DRY-RUN TRAINING PLAN
==============================================================
Model                  meta-llama/Llama-3.1-8B-Instruct
  total params         8,030,261,248
  layers / hidden      32 / 4096
Quantization           4-bit nf4 (double_quant=True, compute=bfloat16)
LoRA                   r=16 alpha=32 dropout=0.05
  target modules       q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
  trainable params     41,943,040 (0.5223% of base)
Sequence length        2048 (packing=True)
Batch                  micro=1 x accum=16 -> effective=16
Optimizer              paged_adamw_8bit lr=0.0002 (cosine, warmup=0.03)
Epochs                 2.0
--------------------------------------------------------------
Estimated VRAM (single GPU, +/-20%):
  base weights           3.89 GiB
  adapter + optimizer    0.62 GiB
  activations            2.47 GiB
  framework overhead     1.00 GiB
  TOTAL                  7.98 GiB
==============================================================
```

The 41,943,040 figure is verified in tests against a hand-computed expansion
(per layer: `r * (in + out)` for each adapted projection, times 32 layers).

## Example eval report

Actual output of the evaluation step of `python -m qlora_tune.demo`, scoring
the bundled `sample_data/sample_predictions.jsonl` (an illustrative fixture —
see the honesty note above):

```
==============================================================
EVALUATION REPORT (10 examples)
==============================================================
Metric                    Base  Fine-tuned     Delta
--------------------------------------------------------------
ROUGE-L F1               0.114       0.957    +0.843
Exact match              0.000       0.000    +0.000
Keyword coverage         0.079       0.980    +0.901
--------------------------------------------------------------
Per-category ROUGE-L F1 (base -> fine-tuned):
  access_request        0.113 -> 0.960
  hardware              0.104 -> 0.955
  password_reset        0.120 -> 0.954
  software_install      0.113 -> 0.958
  vpn                   0.117 -> 0.954
==============================================================
```

To evaluate a real run, generate outputs from both the base and fine-tuned
models into the same JSONL schema (`id`, `category`, `instruction`,
`reference`, `base_output`, `finetuned_output`) and point the demo's
`--predictions` flag (or `qlora_tune.evaluation.harness`) at it.

## Project structure

```
qlora_tune/
├── config.py              # pydantic v2 TrainingConfig + YAML load/save
├── planning.py            # analytic LoRA param counts + VRAM estimates
├── demo.py                # end-to-end light-path CLI
├── train.py               # QLoRA SFT (heavy deps lazy; --dry-run is light)
├── merge.py               # adapter merge/export (heavy deps lazy)
├── data/
│   ├── records.py         # Example dataclass
│   ├── generator.py       # deterministic synthetic helpdesk dataset
│   ├── cleaning.py        # PII scrub, dedupe, length filter
│   ├── splitting.py       # stratified train/val/test
│   ├── loaders.py         # JSONL/CSV read/write
│   └── templates.py       # hand-written Llama-3 + ChatML templates
└── evaluation/
    ├── metrics.py         # ROUGE-L (LCS), exact match, keyword coverage
    └── harness.py         # base-vs-finetuned report from predictions JSONL
configs/                   # llama-3.1-8b.yaml, qwen2.5-0.5b-smoke-test.yaml
sample_data/               # bundled sample predictions for the eval demo
tests/                     # 56 fast, offline tests
```

## Design decisions

- **Why QLoRA.** Full fine-tuning of an 8B model needs >100 GB of GPU memory
  across weights, gradients and optimizer states. QLoRA freezes the base
  model in 4-bit NF4 and trains ~0.5% of the parameters as low-rank adapters,
  bringing the same task inside a single consumer GPU at a small quality cost.
- **Why the light/heavy split.** CI and code review should not require a GPU
  or a 10 GB dependency tree. Everything that can be correct-by-construction
  before training — data quality, config validity, template strings, resource
  fit, evaluation math — lives in the light layer and is fully tested there.
  `train.py` imports heavy libraries inside functions, so `--dry-run` and the
  test suite work on a bare CPU box, verified by a test that asserts torch is
  never imported eagerly.
- **Why analytic planning instead of `print_trainable_parameters()`.** Loading
  a model to count its adapter parameters costs minutes and gigabytes; the
  count is pure arithmetic over published architecture dimensions. Doing the
  math explicitly also makes the VRAM budget auditable line by line.
- **Why hand-written templates and metrics.** Both are small, load-bearing,
  and frequently gotten wrong when hidden behind libraries. Implementing
  ROUGE-L and the chat formats from scratch makes them unit-testable against
  hand-computed cases and keeps the light layer dependency-free.
- **Why synthetic data.** Real helpdesk tickets are confidential. A
  deterministic generator gives a realistic-shaped corpus (with deliberately
  injected fake PII to exercise the scrubber) that any reviewer can reproduce
  bit-for-bit.

## Testing

```bash
python -m pytest        # 56 tests, ~2 s, no network, no GPU
ruff check .            # lint clean
```

Coverage highlights: generator determinism and balance, PII scrubbing
(including false-positive checks on version strings and error codes), dedupe
of PII-only variants, stratified split proportions, both chat templates
against exact expected strings, config validation good/bad cases, LoRA
parameter math against two hand-computed examples, ROUGE-L against
hand-computed LCS cases, the eval harness on the bundled predictions, and
subprocess tests that `--dry-run` and the demo run without heavy deps.

## Roadmap

- DPO/ORPO preference-tuning stage on top of the SFT adapter (TRL `DPOTrainer`).
- Evaluation on public benchmarks (MT-Bench single-turn, IFEval) instead of
  only domain metrics.
- Real inference driver to populate predictions JSONL from merged checkpoints
  (vLLM batch generation).
- Weights & Biases logging and a resume-from-checkpoint story.
- Extend the architecture table (Gemma, Phi) and support `modules_to_save`
  for embedding/LM-head tuning when adding new special tokens.

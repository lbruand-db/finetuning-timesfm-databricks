# Notes for Claude Code

Project-specific conventions that aren't obvious from the file tree.

## Python invocation

Use **`uv run python …`** or **`uvx <tool>`** for any local Python
work. `python`/`python3` are unresolved on this machine (pyenv with no
global default).

```bash
uv run --no-project python -m py_compile <file.py>      # quick syntax check
uv sync --group dev                                     # install test/lint deps
uv run pytest                                           # run the test suite
uv run ruff check . && uv run ruff format --check .     # lint + format check
uv run ty check                                         # type check (src/ only)
```

CI runs all of these on every push to `main` and on every PR — see
`.github/workflows/ci.yml`. The `notebooks/` directory is excluded from
ruff via `pyproject.toml` because the Databricks magic comments aren't
valid Python syntax that ruff should reason about.

## Tests

`tests/` covers `src/` only (the notebooks aren't unit-testable —
they're notebook-source-format and run on Databricks). Lightweight by
design: no transformers/peft/TimesFM checkpoint required. Tests for
`src/model.py` only exercise the input-validation paths in `predict`,
not the actual model forward pass — load the heavy model on Databricks
instead.

## Notebook format

Files in `notebooks/` are **Databricks Python source notebooks**, not
plain `.py` modules:

- First line: `# Databricks notebook source`
- Cells separated by `# COMMAND ----------`
- Markdown cells: `# MAGIC %md` then `# MAGIC <content>` per line
- Pip install: `# MAGIC %pip install …`

Don't lint these with a Python linter unconditionally — the magic
comments are meaningful to the Databricks runtime.

## Importing `src/` from notebooks

Each notebook adds the repo root to `sys.path` via `Path.cwd().parents[0]`.
The notebooks live in `notebooks/`, so the parent is the repo root
where `src/` lives. Don't move the notebooks without updating this.

## Config

`conf/default.yaml` is the single source of truth for UC paths, model
id, and hyperparameters. Every notebook reads it at the top. If you
add a new tunable, put it there — don't hardcode in a notebook.

## Databricks Asset Bundle workflow

Default workspace target is `e2-demo-field-eng` (`targets.dev`).
Always pass `--profile DEFAULT` (or set `DATABRICKS_CONFIG_PROFILE=DEFAULT`).

```bash
databricks bundle validate --target dev
databricks bundle deploy   --target dev          # uploads files + (re)creates the job
databricks bundle run timesfm_inverter_pipeline --target dev --no-wait
```

After editing notebooks/src, **re-deploy** before running — the bundle
ships a snapshot of the files, not a live link.

### Repair a failed run

Two gotchas:

1. After the *first* repair, subsequent repairs require
   `--latest-repair-id <id>`. Pull it from the run's `repair_history`:

   ```bash
   databricks api get /api/2.1/jobs/runs/get \
     --json '{"run_id": <run_id>, "include_history": true}' \
     | uv run --no-project python -c 'import json,sys;[print(h.get("id"), h.get("type")) for h in json.load(sys.stdin).get("repair_history",[])]'
   ```

2. `--rerun-all-failed-tasks --rerun-dependent-tasks` is the usual
   incantation.

### Inspecting a failed task

`databricks jobs get-run <run_id>` lists every task attempt and its
`run_id`. Use that attempt-level `run_id` with
`databricks jobs get-run-output <attempt_run_id>` to pull the
traceback (the `error` and `error_trace` fields).

## Data: Kaggle CSVs

The spec deliberately skips Kaggle auth. The 4 CSVs are mirrored at
<https://github.com/kaivalpanchal/Solar-Panel-Power-Generation> — pull
from there with `curl`, then `databricks fs cp` into the volume.
Don't add `kagglehub` as a dependency in v1.

## MLflow + UC registry

Always set the registry URI before logging models:

```python
mlflow.set_registry_uri("databricks-uc")
```

The pyfunc wrapper deliberately **does not** bake the 200 MB base
model into the artefact. It logs the base model id as a small text
file and lets `from_pretrained` fetch from HF Hub at load time. If you
change this, the registered artefact will balloon to ~400 MB.

## Don't

- Don't externally normalise the AC power. TimesFM 2.5 has RevIN
  inside; double-normalising breaks the forecast.
- Don't change `context_len` to a value that isn't a multiple of 32.
- Don't write to the `raw` volume from a notebook — it's the
  read-only ingest input, hand-uploaded.

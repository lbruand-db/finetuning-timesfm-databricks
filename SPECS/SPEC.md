# SPEC — Fine-tuning TimesFM 2.5 on Databricks for Datacenter Power-Inverter Forecasting

Status: Draft v0.1
Owner: lucas.bruand@databricks.com
Last updated: 2026-05-11

---

## 1. Goal

Adapt the upstream TimesFM 2.5 LoRA fine-tuning example
(`google-research/timesfm/timesfm-forecasting/examples/finetuning`) to a
**Databricks-native workflow** and apply it to a **datacenter power-inverter**
forecasting task.

Concretely we want to:

1. Fine-tune `google/timesfm-2.5-200m-transformers` with a LoRA adapter on
   per-device inverter telemetry (AC output power).
2. Run training on a **single Serverless GPU** notebook (no driver/worker
   cluster setup).
3. Track everything in **MLflow** (params, metrics, the LoRA adapter as an
   artifact) and register the resulting model in **Unity Catalog**.
4. Store raw and curated data in **UC Volumes / Delta** under a dedicated
   schema.
5. Produce a head-to-head **zero-shot vs LoRA** evaluation per device,
   reported as MAE / sMAPE / WAPE and logged to MLflow.

## 2. Use case framing

A datacenter UPS / static-transfer inverter converts a DC input (battery
string or rectifier bus) to a regulated AC output that feeds the IT load.
For operations we'd like to forecast, per inverter unit:

- **Short horizon (next 1 h)** — anomaly anticipation, load-balancing
  between parallel UPS modules.
- **Medium horizon (next 24 h)** — runtime / battery sizing, maintenance
  windowing.

The forecasting target is the AC output power time series per inverter,
sampled at a few-minute cadence, with multiple parallel units acting as
independent series (transfer learning across units is the whole point of a
foundation model).

## 3. Dataset

### Primary dataset

**[Solar Power Generation Data — anikannal / Kaggle](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)**

Why this one:

- Synchronized **inverter-level** AC and DC power readings.
- Two grid-connected PV plants in India, ~22 inverters each.
- 34 days at 15-min cadence → ~3,260 obs / inverter.
- Schema is exactly the shape we want: `(DATE_TIME, PLANT_ID, SOURCE_KEY, DC_POWER, AC_POWER, DAILY_YIELD, TOTAL_YIELD)`.
- CC0 / public on Kaggle; downloadable via `kagglehub` or direct CSV.

Why this is a **legitimate proxy** for a datacenter UPS inverter:

- A solar inverter and a UPS inverter share the core function — DC → AC
  conversion with closed-loop control on the AC side — and produce the
  same family of telemetry (DC bus power, AC output power, efficiency).
- High-quality **datacenter UPS** telemetry is not openly available
  (Vertiv / Schneider / Eaton ship telemetry behind customer contracts).
  Solar PV is the closest open analog with multi-device fleets and per
  device output power.
- We'll document this proxy explicitly in the notebook so the demo isn't
  misleading.

**Forecasting target:** `AC_POWER` per inverter (`SOURCE_KEY`).
**Exogenous features kept out of scope for v1** (we want a pure univariate
fine-tune, matching the upstream example).

### Backup / scale-up dataset (optional)

[`EDS-lab/pv-generation`](https://huggingface.co/datasets/EDS-lab/pv-generation)
on Hugging Face — multiple harmonised PV datasets, useful if we want to
push the fine-tune to a wider fleet.

### Data layout on Databricks

```
Catalog:   lucasbruand_catalog                   (existing or to create)
Schema:    lucasbruand_catalog.timesfm_inverter
Volume:    lucasbruand_catalog.timesfm_inverter.raw     (Kaggle CSVs as-is)
Tables:    lucasbruand_catalog.timesfm_inverter.inverter_readings_silver
           lucasbruand_catalog.timesfm_inverter.inverter_readings_train
           lucasbruand_catalog.timesfm_inverter.inverter_readings_test
```

`inverter_readings_silver` schema:

| column       | type        | notes                              |
|--------------|-------------|------------------------------------|
| ts           | timestamp   | regular 15-min grid               |
| plant_id     | string      | `4135001` / `4136001`             |
| inverter_id  | string      | `SOURCE_KEY`                      |
| ac_power_kw  | double      | target                            |
| dc_power_kw  | double      | reference / efficiency check      |
| daily_yield  | double      | kept for sanity                   |

Train/test split = last 5 days of each inverter held out, mirroring the
"last-window" validation pattern in `finetune_lora.py`.

## 4. Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  Databricks Workspace                                            │
│                                                                  │
│  Volume (UC) ──► Bronze CSV ──► Silver Delta ──► Window Sampler  │
│                                                          │       │
│                                                          ▼       │
│                                       ┌──────────────────────┐   │
│                                       │ Serverless GPU NB    │   │
│  HF Hub  ──► base TimesFM 2.5 bf16 ──►│ + PEFT LoRA training │   │
│                                       │ + MLflow autolog     │   │
│                                       └──────────┬───────────┘   │
│                                                  ▼               │
│                                       MLflow run (UC-backed)     │
│                                       └─► LoRA adapter artifact  │
│                                       └─► metrics, eval table    │
│                                                  │               │
│                                                  ▼               │
│                                   UC Model Registry              │
│                                   `lucasbruand_catalog.timesfm_inverter.lora_v1` │
│                                                  │               │
│                                                  ▼               │
│                              (optional) Model Serving endpoint   │
└──────────────────────────────────────────────────────────────────┘
```

### Compute

- **Serverless GPU** (notebook attached to a GPU-enabled serverless
  environment). Single A10G or H100 is plenty — base model is 200M params
  in bf16, LoRA adds ~1–4M trainable, batch fits comfortably.
- No DLT, no Spark cluster for training. We load Delta → pandas →
  `torch.utils.data.Dataset` in-process.

### Why serverless GPU (and not classic GPU cluster)

- Fast startup, no driver/worker selection.
- The training set is small enough (a few thousand samples) that we don't
  need distributed training. Single device, single process.
- Auto-suspends when the notebook closes — keeps the demo cheap.

## 5. Repository layout

```
finetuning-timesfm-databricks/
├── SPECS/
│   └── SPEC.md                       ← this doc
├── notebooks/
│   ├── 00_setup.py                   ← install deps, create catalog/schema/volume
│   ├── 01_ingest_inverter_data.py    ← Kaggle → Volume → Silver Delta
│   ├── 02_eda_and_split.py           ← quick plots, train/test split
│   ├── 03_finetune_lora.py           ← the main training notebook
│   ├── 04_evaluate.py                ← zero-shot vs LoRA, log metrics
│   └── 05_register_and_serve.py      ← register pyfunc to UC, optional serving
├── src/
│   ├── data.py                       ← RandomWindow / LastWindow datasets
│   ├── model.py                      ← pyfunc wrapper (base + LoRA)
│   └── metrics.py                    ← MAE / sMAPE / WAPE helpers
├── conf/
│   └── default.yaml                  ← hyperparameters, UC paths, model id
└── databricks.yml                    ← DAB bundle (jobs definition)
```

We keep `src/` thin and notebook-driven for demo readability — but
factored enough that the DAB job in `databricks.yml` can run the same
code headless.

## 6. TimesFM-specific choices

Sticking close to the upstream example, with a few changes for the
higher-frequency / longer-history regime:

| param          | upstream | this spec | reason                                 |
|----------------|----------|-----------|----------------------------------------|
| `context_len`  | 64       | **480**   | 5 days × 96 obs/day; multiple of 32     |
| `horizon_len`  | 13       | **96**    | next 24 h at 15-min cadence            |
| `lora_r`       | 4        | **8**     | more data per series, more capacity OK |
| `lora_alpha`   | 8        | 16        | keep alpha = 2·r                       |
| `epochs`       | 10       | 10        | unchanged                              |
| `batch_size`   | 32       | 32        | unchanged                              |
| `num_samples`  | 5000     | 10000     | larger fleet                           |
| dtype          | bf16     | bf16      | unchanged                              |
| optimizer      | AdamW    | AdamW     | unchanged                              |
| scheduler      | cosine   | cosine    | unchanged                              |

**No external normalisation** — RevIN inside TimesFM stays in charge.
Loss is computed by the model when `future_values=` is passed.

## 7. MLflow integration

We use **MLflow autologging** for PyTorch where it helps, plus explicit
calls for the things autolog misses with PEFT:

- `mlflow.set_experiment("/Users/lucas.bruand@databricks.com/timesfm-inverter")`
- One **run per training job**, with tags `{dataset, plant_id, lora_r, context_len}`.
- Logged params: full `argparse` namespace.
- Per-epoch logged metrics: `train_loss`, `val_loss`, `lr`.
- Per-inverter evaluation logged as an MLflow table artifact:
  `eval_table.json` with columns `inverter_id, mae_zeroshot, mae_lora, improvement_pct`.
- LoRA adapter saved via `model.save_pretrained(local_dir)` and logged
  with `mlflow.log_artifact(local_dir, artifact_path="lora_adapter")`.
- Final model registered with `mlflow.pyfunc.log_model(..., registered_model_name="lucasbruand_catalog.timesfm_inverter.lora_v1")`.

### pyfunc wrapper (`src/model.py`)

```python
class TimesFMInverterModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        from transformers import TimesFm2_5ModelForPrediction
        from peft import PeftModel
        base = TimesFm2_5ModelForPrediction.from_pretrained(
            context.artifacts["base_model_id"], torch_dtype=torch.bfloat16
        )
        self.model = PeftModel.from_pretrained(base, context.artifacts["lora_dir"])
        self.model.eval()

    def predict(self, context, model_input):
        # model_input: pandas DataFrame with one row per series, column "context"
        # holding a list[float] of length context_len.
        # returns: DataFrame with column "forecast" (list[float] of length horizon_len)
        ...
```

The wrapper bundles the **base model id as an artifact reference** rather
than baking the 200M weights into the registered model — they're fetched
from HF Hub at load time. This keeps the registered artifact small (the
LoRA is ~5 MB).

## 8. Evaluation

Mirrors `finetune_lora.py::evaluate`, but logged through MLflow:

- For each inverter, take the last `context_len` of train → predict next
  `horizon_len`, compare against held-out test.
- Metrics: MAE, sMAPE, WAPE (WAPE is the cleanest summary for AC power
  since values are positive and span 0 → rated kW).
- Save per-inverter forecast vs ground-truth plot grid as an MLflow
  artifact (matplotlib).

### Acceptance criteria for the demo

| check                                                  | target            |
|--------------------------------------------------------|-------------------|
| Notebook 03 completes end-to-end on serverless GPU    | < 15 min          |
| LoRA WAPE improvement over zero-shot, fleet-averaged  | ≥ 10% relative    |
| LoRA adapter artifact size                             | < 20 MB           |
| Registered UC model loads + predicts in notebook 05   | green             |

If the LoRA does not beat zero-shot by ≥ 10% WAPE, that's a result worth
documenting — the upstream example shows the gain is real on retail, but
we should be honest about the proxy data and report whatever we find.

## 9. Non-goals (v1)

- Multi-GPU / FSDP training. Not needed at this scale.
- Streaming / online updates of the adapter.
- Multivariate / exogenous features (weather, irradiance). Univariate
  first.
- Model serving load testing. We confirm the endpoint deploys but don't
  benchmark it.
- Probabilistic / quantile output. We report point forecasts only in v1.

## 10. Open questions

1. **Compute SKU** — which serverless GPU tier do we target by default?
   Need to confirm A10 availability vs H100 in the user's workspace
   region.
2. **Kaggle auth on Databricks** — `kagglehub` needs `KAGGLE_USERNAME` /
   `KAGGLE_KEY`. Where do we store them? Databricks secret scope
   `kaggle` with `username` / `key` is the proposed answer.
3. **Base model caching** — first run downloads ~400 MB from HF. Cache
   to a Volume so re-runs are instant?

## 11. Next steps

1. Confirm UC catalog name and serverless GPU availability.
2. Scaffold the repo per §5 (notebooks + `src/` + DAB).
3. Implement notebook 00 → 03 against the Kaggle solar dataset.
4. Run training, capture WAPE numbers.
5. Iterate on `lora_r` / `context_len` if §8 acceptance not met.

---

### References

- Upstream example: <https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning>
- HF notebook by @kashif: <https://github.com/huggingface/notebooks/blob/main/examples/timesfm2_5.ipynb>
- TimesFM 2.5 model card: <https://huggingface.co/google/timesfm-2.5-200m-transformers>
- Primary dataset: <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data>
- Backup dataset: <https://huggingface.co/datasets/EDS-lab/pv-generation>

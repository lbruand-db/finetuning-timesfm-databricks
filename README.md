# finetuning-timesfm-databricks

Fine-tune **TimesFM 2.5** with a **LoRA** adapter on Databricks, in
the framing of a datacenter **power-inverter** forecasting task. The
fine-tune itself is per-device time-series forecasting: given the last
~5 days of AC output power at 15-min cadence, predict the next 24 h
per inverter.

The use case is datacenter UPS / inverter telemetry; high-quality UPS
fleets are not openly available, so the demo uses a **solar PV
inverter** dataset (same DC→AC conversion, same telemetry shape) as
an open analogue.

See [`SPECS/SPEC.md`](SPECS/SPEC.md) for the design doc — dataset
choice, hyperparameters, MLflow & UC integration, acceptance criteria.

## What lives here

```
.
├── SPECS/SPEC.md                       design doc (read this first)
├── conf/default.yaml                   single source of truth: UC paths, hyperparameters
├── src/
│   ├── data.py                         random/last-window torch Datasets
│   ├── metrics.py                      MAE / sMAPE / WAPE
│   └── model.py                        TimesFMInverterModel pyfunc wrapper
├── notebooks/
│   ├── 00_setup.py                     pip install + create UC schema/volumes
│   ├── 01_ingest_inverter_data.py      CSVs → silver Delta
│   ├── 02_eda_and_split.py             train/test split on the last N days/inverter
│   ├── 03_finetune_lora.py             LoRA fine-tune, MLflow run, adapter artifact
│   ├── 04_evaluate.py                  zero-shot vs LoRA MAE/sMAPE/WAPE
│   └── 05_register_and_serve.py        register pyfunc to Unity Catalog
└── databricks.yml                      Asset Bundle: 6-task sequential job on serverless GPU
```

## Quickstart

Prereqs: Databricks CLI ≥ v0.298, a workspace with serverless GPU
enabled, a Unity Catalog you can write to (default: `lucasbruand_catalog`).

```bash
# 1. Auth
databricks auth login --profile DEFAULT          # or your profile

# 2. Validate + deploy
databricks bundle validate --target dev
databricks bundle deploy   --target dev

# 3. First run — creates the schema + raw volume + hf_cache volume,
#    then fails at `ingest` (no data yet).
databricks bundle run timesfm_inverter_pipeline --target dev

# 4. Upload the 4 Kaggle CSVs into the raw volume. The dataset is also
#    mirrored on GitHub for credential-free download:
mkdir -p /tmp/kaggle_solar && cd /tmp/kaggle_solar
for f in Plant_1_Generation_Data.csv Plant_1_Weather_Sensor_Data.csv \
         Plant_2_Generation_Data.csv Plant_2_Weather_Sensor_Data.csv; do
  curl -sSLfO "https://raw.githubusercontent.com/kaivalpanchal/Solar-Panel-Power-Generation/main/$f"
done
VOL="dbfs:/Volumes/lucasbruand_catalog/timesfm_inverter/raw"
for f in *.csv; do databricks fs cp --overwrite "$f" "$VOL/$f"; done

# 5. Repair the failed run from `ingest` onward (need the latest repair_id
#    from the run's repair_history if there's been one already).
databricks jobs repair-run <run_id> --rerun-all-failed-tasks --rerun-dependent-tasks
```

Outputs land in:
- **Silver Delta**: `lucasbruand_catalog.timesfm_inverter.inverter_readings_silver`
- **MLflow experiment**: `/Users/<you>/timesfm-inverter`
- **Registered UC model**: `lucasbruand_catalog.timesfm_inverter.lora_v1`

## Configuration

Everything tunable lives in [`conf/default.yaml`](conf/default.yaml).
The notebooks read it at the top — change once, propagates everywhere.

Notable knobs:

| key                    | default        | meaning                                    |
|------------------------|----------------|--------------------------------------------|
| `training.context_len` | 480            | 5 days × 96 obs/day (must be a multiple of 32) |
| `training.horizon_len` | 96             | next 24 h                                  |
| `training.lora_r`      | 8              | LoRA rank                                  |
| `training.epochs`      | 10             |                                            |
| `training.test_days`   | 5              | tail days held out per inverter            |
| `unity_catalog.catalog`| `lucasbruand_catalog` | swap for your own catalog            |

## References

- Upstream TimesFM fine-tune example: <https://github.com/google-research/timesfm/tree/master/timesfm-forecasting/examples/finetuning>
- HF notebook by @kashif: <https://github.com/huggingface/notebooks/blob/main/examples/timesfm2_5.ipynb>
- TimesFM 2.5 model card: <https://huggingface.co/google/timesfm-2.5-200m-transformers>
- Dataset (Kaggle): <https://www.kaggle.com/datasets/anikannal/solar-power-generation-data>
- Dataset (GitHub mirror, no auth required): <https://github.com/kaivalpanchal/Solar-Panel-Power-Generation>

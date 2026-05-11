# Databricks notebook source
# MAGIC %md
# MAGIC # 04 — Evaluate: zero-shot vs LoRA per inverter
# MAGIC
# MAGIC For each inverter:
# MAGIC 1. Take the last `context_len` of the train table as the context.
# MAGIC 2. Predict next `horizon_len` with the base model and the LoRA model.
# MAGIC 3. Compare against the test table.
# MAGIC
# MAGIC Metrics logged to MLflow: MAE, sMAPE, WAPE — zero-shot, LoRA, and
# MAGIC the relative improvement.

# COMMAND ----------

# MAGIC %pip install -q "transformers>=4.46" "peft>=0.13" "accelerate>=1.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

cfg = yaml.safe_load((REPO_ROOT / "conf" / "default.yaml").read_text())
uc, tr, md, dc = cfg["unity_catalog"], cfg["training"], cfg["model"], cfg["data"]

hf_cache = f"/Volumes/{uc['catalog']}/{uc['schema']}/{uc['hf_cache_volume']}"
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache

train_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['train_table']}"
test_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['test_table']}"

# COMMAND ----------

# Run id can be passed in by the DAB job; fall back to the most recent
# successful run in the experiment when running interactively.
import mlflow

run_id = None
try:
    run_id = dbutils.jobs.taskValues.get(
        taskKey="finetune", key="train_run_id", default=None, debugValue=None
    )
except Exception:
    run_id = None

if run_id is None:
    exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_path"])
    if exp is None:
        raise RuntimeError("No MLflow experiment found — run notebook 03 first.")
    runs = mlflow.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if runs.empty:
        raise RuntimeError("No finished runs in experiment.")
    run_id = runs.iloc[0]["run_id"]

print(f"Using run_id = {run_id}")

# COMMAND ----------

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import TimesFm2_5ModelForPrediction

from src.metrics import mae, smape, wape

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

dtype = getattr(torch, md["dtype"])
base = TimesFm2_5ModelForPrediction.from_pretrained(
    md["base_id"], torch_dtype=dtype, device_map=device
)
base.eval()
context_len = min(tr["context_len"], base.config.context_length)
horizon_len = tr["horizon_len"]

# Pull the LoRA adapter from the MLflow run.
local_adapter = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="lora_adapter"
)
ft_model = PeftModel.from_pretrained(base, local_adapter)
ft_model.eval()
print(f"LoRA adapter loaded from {local_adapter}")

# COMMAND ----------

train_pdf = (
    spark.table(train_fqn)
    .select(dc["series_id_col"], "ts", dc["target_col"])
    .toPandas()
    .sort_values(["inverter_id", "ts"])
)
test_pdf = (
    spark.table(test_fqn)
    .select(dc["series_id_col"], "ts", dc["target_col"])
    .toPandas()
    .sort_values(["inverter_id", "ts"])
)

inverter_ids = sorted(set(train_pdf["inverter_id"]) & set(test_pdf["inverter_id"]))
print(f"Evaluating {len(inverter_ids)} inverters")

# COMMAND ----------

rows: list[dict] = []
forecasts_per_inverter: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

with torch.no_grad():
    for sid in inverter_ids:
        train_s = train_pdf.loc[
            train_pdf["inverter_id"] == sid, dc["target_col"]
        ].to_numpy(dtype=np.float32)
        test_s = test_pdf.loc[
            test_pdf["inverter_id"] == sid, dc["target_col"]
        ].to_numpy(dtype=np.float32)

        if len(train_s) < context_len or len(test_s) < horizon_len:
            continue

        ctx = torch.tensor(
            train_s[-context_len:], dtype=torch.float32, device=device
        ).unsqueeze(0)
        gt = test_s[:horizon_len]

        zs = base(past_values=ctx).mean_predictions[0, :horizon_len].float().cpu().numpy()
        ft = (
            ft_model(past_values=ctx).mean_predictions[0, :horizon_len]
            .float().cpu().numpy()
        )

        forecasts_per_inverter[sid] = (gt, zs, ft)
        rows.append({
            "inverter_id": sid,
            "mae_zeroshot": mae(gt, zs),
            "mae_lora": mae(gt, ft),
            "smape_zeroshot": smape(gt, zs),
            "smape_lora": smape(gt, ft),
            "wape_zeroshot": wape(gt, zs),
            "wape_lora": wape(gt, ft),
        })

eval_df = pd.DataFrame(rows)
display(eval_df)

# COMMAND ----------

# Fleet-averaged summary and per-metric improvement
def _improvement(zs_col: str, ft_col: str) -> float:
    return float((eval_df[zs_col].mean() - eval_df[ft_col].mean())
                 / eval_df[zs_col].mean() * 100.0)

summary = {
    "mae_zeroshot_mean": float(eval_df["mae_zeroshot"].mean()),
    "mae_lora_mean": float(eval_df["mae_lora"].mean()),
    "mae_improvement_pct": _improvement("mae_zeroshot", "mae_lora"),
    "wape_zeroshot_mean": float(eval_df["wape_zeroshot"].mean()),
    "wape_lora_mean": float(eval_df["wape_lora"].mean()),
    "wape_improvement_pct": _improvement("wape_zeroshot", "wape_lora"),
    "smape_zeroshot_mean": float(eval_df["smape_zeroshot"].mean()),
    "smape_lora_mean": float(eval_df["smape_lora"].mean()),
    "smape_improvement_pct": _improvement("smape_zeroshot", "smape_lora"),
}
for k, v in summary.items():
    print(f"  {k:32s}  {v:8.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log metrics + per-inverter table back to the training run

# COMMAND ----------

with mlflow.start_run(run_id=run_id):
    mlflow.log_metrics(summary)
    mlflow.log_table(data=eval_df, artifact_file="eval_table.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Forecast plot grid (first 8 inverters)

# COMMAND ----------

import matplotlib.pyplot as plt

sample = list(forecasts_per_inverter.items())[:8]
n = len(sample)
ncols = 2
nrows = (n + 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 2.6 * nrows), sharex=False)
axes = axes.flatten()

for ax, (sid, (gt, zs, ft)) in zip(axes, sample):
    x = np.arange(horizon_len)
    ax.plot(x, gt, label="ground truth", lw=1.2, color="black")
    ax.plot(x, zs, label="zero-shot", lw=1.0, alpha=0.7)
    ax.plot(x, ft, label="LoRA", lw=1.0, alpha=0.9)
    ax.set_title(f"inverter {sid}")
    ax.legend(fontsize=7)

for ax in axes[n:]:
    ax.set_visible(False)
plt.tight_layout()
out_png = "/tmp/forecast_grid.png"
plt.savefig(out_png, dpi=120)
plt.show()

with mlflow.start_run(run_id=run_id):
    mlflow.log_artifact(out_png, artifact_path="plots")

print("done")

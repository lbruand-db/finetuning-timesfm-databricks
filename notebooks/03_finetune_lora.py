# Databricks notebook source
# MAGIC %md
# MAGIC # 03 — Fine-tune TimesFM 2.5 with LoRA
# MAGIC
# MAGIC Reads the train Delta table, builds random/last-window datasets,
# MAGIC attaches a LoRA adapter to `google/timesfm-2.5-200m-transformers`,
# MAGIC and runs a single-GPU training loop. Everything is logged to MLflow.
# MAGIC
# MAGIC Run on **serverless GPU (A10)**.

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC   "transformers>=4.46" \
# MAGIC   "peft>=0.13" \
# MAGIC   "accelerate>=1.0"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import os
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

cfg = yaml.safe_load((REPO_ROOT / "conf" / "default.yaml").read_text())
uc = cfg["unity_catalog"]
tr = cfg["training"]
md = cfg["model"]
dc = cfg["data"]

hf_cache = f"/Volumes/{uc['catalog']}/{uc['schema']}/{uc['hf_cache_volume']}"
os.environ["HF_HOME"] = hf_cache
os.environ["TRANSFORMERS_CACHE"] = hf_cache
print(f"HF_HOME = {hf_cache}")

train_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['train_table']}"

# COMMAND ----------

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import TimesFm2_5ModelForPrediction

from src.data import (
    TimeSeriesLastWindowDataset,
    TimeSeriesRandomWindowDataset,
    series_from_silver,
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
torch.manual_seed(tr["seed"])
np.random.seed(tr["seed"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build datasets

# COMMAND ----------

train_pdf = (
    spark.table(train_fqn)
    .select(dc["series_id_col"], "ts", dc["target_col"])
    .toPandas()
)
ids, series = series_from_silver(
    train_pdf,
    series_id_col=dc["series_id_col"],
    target_col=dc["target_col"],
)
print(f"Inverters: {len(ids)} | shortest series: {min(len(s) for s in series)} obs")

train_ds = TimeSeriesRandomWindowDataset(
    series,
    context_len=tr["context_len"],
    horizon_len=tr["horizon_len"],
    num_samples=tr["num_samples"],
    seed=tr["seed"],
)
val_ds = TimeSeriesLastWindowDataset(
    series, context_len=tr["context_len"], horizon_len=tr["horizon_len"]
)

train_loader = DataLoader(
    train_ds, batch_size=tr["batch_size"], shuffle=True, drop_last=True
)
val_loader = DataLoader(val_ds, batch_size=tr["batch_size"])

print(
    f"Train samples: {len(train_ds)} ({len(train_loader)} batches) | "
    f"Val series: {len(val_ds)}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load base + attach LoRA

# COMMAND ----------

dtype = getattr(torch, md["dtype"])
base = TimesFm2_5ModelForPrediction.from_pretrained(
    md["base_id"], torch_dtype=dtype, device_map=device
)
context_len = min(tr["context_len"], base.config.context_length)
if context_len != tr["context_len"]:
    print(
        f"⚠ Capping context_len to model max: {context_len} "
        f"(config {base.config.context_length})"
    )

lora_cfg = LoraConfig(
    r=tr["lora_r"],
    lora_alpha=tr["lora_alpha"],
    target_modules="all-linear",
    lora_dropout=tr["lora_dropout"],
    bias="none",
)
model = get_peft_model(base, lora_cfg)
model.print_trainable_parameters()

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow run

# COMMAND ----------

import mlflow

# Serverless compute doesn't expose `spark.mlflow.modelRegistryUri`, which
# `MlflowClient()` tries to read by default. Pin both URIs explicitly so
# the implicit lookup never runs.
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(cfg["mlflow"]["experiment_path"])

optimizer = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=0.01)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=tr["epochs"] * len(train_loader)
)

local_adapter_dir = "/tmp/timesfm_inverter_lora"

with mlflow.start_run(run_name="timesfm25-inverter-lora") as run:
    mlflow.log_params({
        "base_model": md["base_id"],
        **{k: v for k, v in tr.items()},
        "n_inverters": len(ids),
        "context_len_effective": context_len,
    })
    mlflow.set_tags({
        "dataset": "kaggle/anikannal/solar-power-generation-data",
        "framework": "transformers+peft",
        "target": dc["target_col"],
    })

    best_val_loss = float("inf")
    global_step = 0
    for epoch in range(1, tr["epochs"] + 1):
        model.train()
        epoch_loss = 0.0
        n = 0
        for ctx_batch, tgt_batch in train_loader:
            ctx_batch = ctx_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            out = model(
                past_values=ctx_batch,
                future_values=tgt_batch,
                forecast_context_len=context_len,
            )
            loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

            epoch_loss += float(loss.item())
            n += 1
            global_step += 1

        avg_train = epoch_loss / max(n, 1)

        model.eval()
        val_loss = 0.0
        m = 0
        with torch.no_grad():
            for ctx_batch, tgt_batch in val_loader:
                ctx_batch = ctx_batch.to(device)
                tgt_batch = tgt_batch.to(device)
                out = model(
                    past_values=ctx_batch,
                    future_values=tgt_batch,
                    forecast_context_len=context_len,
                )
                val_loss += float(out.loss.item())
                m += 1
        avg_val = val_loss / max(m, 1)

        mlflow.log_metrics(
            {"train_loss": avg_train, "val_loss": avg_val,
             "lr": scheduler.get_last_lr()[0]},
            step=epoch,
        )
        print(f"epoch {epoch:02d}/{tr['epochs']} | train {avg_train:.4f} | val {avg_val:.4f}")

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            model.save_pretrained(local_adapter_dir)
            print(f"  ✓ new best — saved adapter to {local_adapter_dir}")

    # log_artifacts (plural) uploads the *contents* of the dir under the
    # given artifact_path so the adapter files land at
    # lora_adapter/adapter_config.json etc. — what PeftModel expects.
    mlflow.log_artifacts(local_adapter_dir, artifact_path="lora_adapter")
    mlflow.log_metric("best_val_loss", best_val_loss)

    print(f"\nDone. run_id={run.info.run_id}  best val loss={best_val_loss:.4f}")

# COMMAND ----------

# MAGIC %md
# MAGIC Capture the run id for downstream notebooks.

# COMMAND ----------

dbutils.jobs.taskValues.set(key="train_run_id", value=run.info.run_id)
print(f"Run id stashed for downstream tasks: {run.info.run_id}")

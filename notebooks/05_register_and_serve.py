# Databricks notebook source
# MAGIC %md
# MAGIC # 05 — Register the LoRA model in Unity Catalog
# MAGIC
# MAGIC Wraps the LoRA adapter + base-model reference in an
# MAGIC `mlflow.pyfunc.PythonModel`, logs it to MLflow, and registers it as
# MAGIC a UC model so it can be loaded by downstream notebooks or served.
# MAGIC
# MAGIC Only the small LoRA weights live in the artefact store — the 200 MB
# MAGIC base checkpoint is pulled from HF Hub at load time.

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
uc, tr, md = cfg["unity_catalog"], cfg["training"], cfg["model"]

hf_cache = f"/Volumes/{uc['catalog']}/{uc['schema']}/{uc['hf_cache_volume']}"
os.environ["HF_HOME"] = hf_cache

registered_name = f"{uc['catalog']}.{uc['schema']}.{uc['registered_model']}"
print(f"Will register as: {registered_name}")

# COMMAND ----------

import mlflow

# UC-backed registry, not the legacy workspace registry.
mlflow.set_registry_uri("databricks-uc")

# Recover the latest training run's adapter.
exp = mlflow.get_experiment_by_name(cfg["mlflow"]["experiment_path"])
runs = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="attributes.status = 'FINISHED'",
    order_by=["attributes.start_time DESC"],
    max_results=1,
)
run_id = runs.iloc[0]["run_id"]
print(f"Source run: {run_id}")

local_adapter = mlflow.artifacts.download_artifacts(
    run_id=run_id, artifact_path="lora_adapter"
)
print(f"Adapter staged at: {local_adapter}")

# COMMAND ----------

# Small text files act as artefacts so the pyfunc can read them at load time
# without having to bake them into the wrapper class.
staging = "/tmp/timesfm_inverter_pyfunc"
os.makedirs(staging, exist_ok=True)
Path(staging, "base_model_id.txt").write_text(md["base_id"])
Path(staging, "context_len.txt").write_text(str(tr["context_len"]))
Path(staging, "horizon_len.txt").write_text(str(tr["horizon_len"]))

# COMMAND ----------

import pandas as pd
from mlflow.models import infer_signature

from src.model import TimesFMInverterModel

# Example inputs/outputs for the signature
example_in = pd.DataFrame({
    "context": [[0.0] * tr["context_len"]]
})
example_out = pd.DataFrame({
    "forecast": [[0.0] * tr["horizon_len"]]
})
signature = infer_signature(example_in, example_out)

with mlflow.start_run(run_name="register-timesfm-inverter-lora"):
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=TimesFMInverterModel(),
        artifacts={
            "lora_dir": local_adapter,
            "base_model_id": str(Path(staging, "base_model_id.txt")),
            "context_len": str(Path(staging, "context_len.txt")),
            "horizon_len": str(Path(staging, "horizon_len.txt")),
            "hf_cache": hf_cache,
        },
        code_paths=[str(REPO_ROOT / "src")],
        pip_requirements=[
            "transformers>=4.46",
            "peft>=0.13",
            "accelerate>=1.0",
            "torch",
            "pandas",
            "numpy",
        ],
        signature=signature,
        input_example=example_in,
        registered_model_name=registered_name,
    )

print(f"Registered → {registered_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Smoke test — load latest version and predict

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()
latest = client.search_model_versions(
    f"name='{registered_name}'", order_by=["version_number DESC"], max_results=1
)[0]
print(f"Latest registered version: {latest.version}")

loaded = mlflow.pyfunc.load_model(f"models:/{registered_name}/{latest.version}")

# Build a real-looking context from the silver table
silver = spark.table(f"{uc['catalog']}.{uc['schema']}.{uc['silver_table']}")
sample = (
    silver.where("inverter_id IS NOT NULL")
    .orderBy("ts")
    .toPandas()
    .groupby("inverter_id")["ac_power_kw"]
    .apply(lambda s: s.tolist()[-tr["context_len"] :])
    .iloc[:3]
)
df_in = pd.DataFrame({"context": sample.values})
df_out = loaded.predict(df_in)
print(df_out)
print(f"\nForecast lengths: {[len(f) for f in df_out['forecast']]}")

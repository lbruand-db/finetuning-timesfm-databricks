# Databricks notebook source
# MAGIC %md
# MAGIC # 00 — Setup
# MAGIC
# MAGIC Installs Python deps and provisions the UC schema + Volumes used by
# MAGIC the rest of the notebooks.
# MAGIC
# MAGIC Run once on a serverless GPU environment (A10). Re-running is safe —
# MAGIC every DDL uses `IF NOT EXISTS`.

# COMMAND ----------

# MAGIC %pip install -q \
# MAGIC   "transformers>=4.46" \
# MAGIC   "peft>=0.13" \
# MAGIC   "accelerate>=1.0" \
# MAGIC   "pyarrow>=15" \
# MAGIC   "scikit-learn>=1.4"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

cfg = yaml.safe_load((REPO_ROOT / "conf" / "default.yaml").read_text())
uc = cfg["unity_catalog"]
catalog, schema = uc["catalog"], uc["schema"]
raw_vol, hf_vol = uc["raw_volume"], uc["hf_cache_volume"]
print(f"Target: {catalog}.{schema} | volumes: {raw_vol}, {hf_vol}")

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{raw_vol}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.{hf_vol}")

raw_path = f"/Volumes/{catalog}/{schema}/{raw_vol}"
hf_path = f"/Volumes/{catalog}/{schema}/{hf_vol}"
print(f"Raw volume path:      {raw_path}")
print(f"HF cache volume path: {hf_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next: stage the Kaggle CSVs
# MAGIC
# MAGIC Download these 4 files from
# MAGIC [anikannal/solar-power-generation-data](https://www.kaggle.com/datasets/anikannal/solar-power-generation-data)
# MAGIC and drop them into the `raw` volume listed above (Data → Volume →
# MAGIC Upload):
# MAGIC
# MAGIC - `Plant_1_Generation_Data.csv`
# MAGIC - `Plant_1_Weather_Sensor_Data.csv` *(not used in v1, but keep for completeness)*
# MAGIC - `Plant_2_Generation_Data.csv`
# MAGIC - `Plant_2_Weather_Sensor_Data.csv` *(not used in v1)*
# MAGIC
# MAGIC Then proceed to `01_ingest_inverter_data`.

# COMMAND ----------

import os

found = sorted(os.listdir(raw_path)) if os.path.isdir(raw_path) else []
print(f"Files currently in {raw_path}:")
for f in found:
    print(f"  {f}")
if not found:
    print("  (empty — upload the Kaggle CSVs before running notebook 01)")

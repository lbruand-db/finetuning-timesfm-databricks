# Databricks notebook source
# MAGIC %md
# MAGIC # 02 — EDA + train/test split
# MAGIC
# MAGIC Quick look at the silver table, then split the last `test_days` of
# MAGIC each inverter into a test table and the rest into a train table.

# COMMAND ----------

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

cfg = yaml.safe_load((REPO_ROOT / "conf" / "default.yaml").read_text())
uc = cfg["unity_catalog"]

silver_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['silver_table']}"
train_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['train_table']}"
test_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['test_table']}"
test_days = int(cfg["training"]["test_days"])

# COMMAND ----------

from pyspark.sql import functions as F, Window

silver = spark.table(silver_fqn)

# Per-inverter max timestamp → cutoff = max_ts - test_days
w = Window.partitionBy("plant_id", "inverter_id")
with_cutoff = silver.withColumn(
    "cutoff_ts",
    F.expr(f"max(ts) over (partition by plant_id, inverter_id) - interval {test_days} days"),
)

train = with_cutoff.where(F.col("ts") <= F.col("cutoff_ts")).drop("cutoff_ts")
test = with_cutoff.where(F.col("ts") > F.col("cutoff_ts")).drop("cutoff_ts")

# COMMAND ----------

train.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(train_fqn)
test.write.mode("overwrite").option("overwriteSchema", True).saveAsTable(test_fqn)

print(f"Train: {spark.table(train_fqn).count():,} rows → {train_fqn}")
print(f"Test:  {spark.table(test_fqn).count():,} rows → {test_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Distribution check

# COMMAND ----------

display(
    spark.sql(f"""
        SELECT plant_id, inverter_id,
               COUNT(*)             AS n_train,
               (SELECT COUNT(*) FROM {test_fqn} t
                WHERE t.plant_id = s.plant_id AND t.inverter_id = s.inverter_id) AS n_test,
               MIN(ts) AS train_min, MAX(ts) AS train_max
        FROM {train_fqn} s
        GROUP BY plant_id, inverter_id
        ORDER BY plant_id, inverter_id
    """)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## One inverter's AC power timeline

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

sample_id = (
    spark.table(train_fqn)
    .select("inverter_id")
    .distinct()
    .limit(1)
    .collect()[0]
    .inverter_id
)

train_pdf: pd.DataFrame = (
    spark.table(train_fqn)
    .where(F.col("inverter_id") == sample_id)
    .orderBy("ts")
    .select("ts", "ac_power_kw")
    .toPandas()
)
test_pdf: pd.DataFrame = (
    spark.table(test_fqn)
    .where(F.col("inverter_id") == sample_id)
    .orderBy("ts")
    .select("ts", "ac_power_kw")
    .toPandas()
)

fig, ax = plt.subplots(figsize=(12, 3))
ax.plot(train_pdf["ts"], train_pdf["ac_power_kw"], label="train", lw=0.6)
ax.plot(test_pdf["ts"], test_pdf["ac_power_kw"], label="test", lw=0.6, color="tab:red")
ax.set_title(f"AC power (kW) — inverter {sample_id}")
ax.set_xlabel("time")
ax.legend()
plt.tight_layout()
plt.show()

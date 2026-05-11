# Databricks notebook source
# MAGIC %md
# MAGIC # 01 — Ingest inverter telemetry
# MAGIC
# MAGIC Reads `Plant_{1,2}_Generation_Data.csv` from the raw UC Volume, parses
# MAGIC the two distinct `DATE_TIME` formats, normalises column names, and
# MAGIC writes a single silver Delta table keyed by `(plant_id, inverter_id, ts)`.
# MAGIC
# MAGIC The two plants use different timestamp formats:
# MAGIC - Plant 1: `DD-MM-YYYY HH:MM`
# MAGIC - Plant 2: `YYYY-MM-DD HH:MM:SS`

# COMMAND ----------

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path.cwd().resolve().parents[0]
sys.path.insert(0, str(REPO_ROOT))

cfg = yaml.safe_load((REPO_ROOT / "conf" / "default.yaml").read_text())
uc = cfg["unity_catalog"]

raw_path = f"/Volumes/{uc['catalog']}/{uc['schema']}/{uc['raw_volume']}"
silver_fqn = f"{uc['catalog']}.{uc['schema']}.{uc['silver_table']}"

# COMMAND ----------

from pyspark.sql import functions as F

plant1 = (
    spark.read.option("header", True)
    .csv(f"{raw_path}/Plant_1_Generation_Data.csv")
    .withColumn("ts", F.to_timestamp("DATE_TIME", "dd-MM-yyyy HH:mm"))
)
plant2 = (
    spark.read.option("header", True)
    .csv(f"{raw_path}/Plant_2_Generation_Data.csv")
    .withColumn("ts", F.to_timestamp("DATE_TIME", "yyyy-MM-dd HH:mm:ss"))
)

raw = plant1.unionByName(plant2)

silver = (
    raw.select(
        F.col("ts"),
        F.col("PLANT_ID").cast("string").alias("plant_id"),
        F.col("SOURCE_KEY").cast("string").alias("inverter_id"),
        # Kaggle DC/AC are in watts; convert to kW for readability.
        (F.col("AC_POWER").cast("double") / 1000.0).alias("ac_power_kw"),
        (F.col("DC_POWER").cast("double") / 1000.0).alias("dc_power_kw"),
        F.col("DAILY_YIELD").cast("double").alias("daily_yield"),
    )
    .where(F.col("ts").isNotNull())
    .dropDuplicates(["plant_id", "inverter_id", "ts"])
)

silver.printSchema()
display(silver.limit(5))

# COMMAND ----------

(
    silver.write.mode("overwrite")
    .option("overwriteSchema", True)
    .saveAsTable(silver_fqn)
)
print(f"Wrote silver → {silver_fqn}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity checks

# COMMAND ----------

stats = spark.sql(f"""
    SELECT plant_id,
           COUNT(DISTINCT inverter_id) AS n_inverters,
           MIN(ts) AS ts_min,
           MAX(ts) AS ts_max,
           COUNT(*) AS n_rows
    FROM {silver_fqn}
    GROUP BY plant_id
    ORDER BY plant_id
""")
display(stats)

# COMMAND ----------

per_inverter = spark.sql(f"""
    SELECT plant_id, inverter_id, COUNT(*) AS n_obs
    FROM {silver_fqn}
    GROUP BY plant_id, inverter_id
    ORDER BY n_obs
""")
display(per_inverter)

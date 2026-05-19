# SPEC — 20-minute Presentation: Fine-tuning Time-Series Foundation Models for Schneider Electric

Status: Draft v0.1
Owner: lucas.bruand@databricks.com
Last updated: 2026-05-18

---

## 1. Goal

Build a **20-minute customer-facing presentation** that:

1. Frames *why* fine-tuning time-series foundation models (TS-FMs) matters
   for industrial customers, and Schneider Electric in particular.
2. Surveys the two leading open TS-FMs — **Google TimesFM 2.5** and
   **Amazon Chronos-2** — and explains where fine-tuning is the right
   tool (vs. zero-shot or classical baselines).
3. Anchors the technical pitch on a concrete benchmark: the
   **Databricks `many-model-forecasting` (MMF) solution accelerator**,
   which already integrates Chronos / TimesFM alongside statistical and
   deep-learning baselines on Spark.
4. Walks through a **3-phase roadmap** to land a fine-tuned TS-FM with
   Schneider Electric, grounded in the PoC in this repo
   (TimesFM 2.5 + LoRA on PV inverter telemetry as a UPS-inverter proxy).
5. Positions Databricks against what Siemens (Senseye), ABB (Genix),
   GE Vernova (SmartSignal / Proficy CSense) are shipping.

Output format: **Typst slides** rendered to PDF using
`lbruand-db/typst-dbrx-template`. Source file: `PREZ/main.typ`.
Compiled output: `PREZ/main.pdf`.

## 2. Audience & talk shape

- **Primary audience:** Schneider Electric data/AI leadership +
  EcoStruxure / EcoCare product managers. Mixed technical depth — some
  ML/data-science fluency, some platform/buying mindset.
- **Secondary audience:** Databricks internal review (FE, GTM).
- **Tone:** Customer-first, vendor-neutral on the model layer
  (TimesFM *and* Chronos), Databricks-opinionated on the platform layer
  (UC, MLflow, Serverless GPU, MMF).
- **Length:** 20 min talk + 10 min Q&A. Budget below.
- **Demo:** No live demo; one or two screenshots from the PoC notebooks
  in this repo + an MLflow run screenshot. Live demo is a v2 / follow-up.

## 3. Timing budget (20 min, 21 slides)

| # | Slide                                          | Type (template fn)       | min  | cum  |
|---|------------------------------------------------|--------------------------|------|------|
| 1 | Title                                          | `title-slide`            | 0.5  | 0.5  |
| 2 | Why now: TS-FMs hit production in 2025         | `content-slide`          | 1.5  | 2.0  |
| 3 | Why Schneider, why this use case               | `subtitle-content-slide` | 1.5  | 3.5  |
| 4 | Competitive landscape (Siemens / ABB / GE)     | `three-column-slide`     | 2.0  | 5.5  |
| 5 | Section — Foundation models for time series    | `section-slide`          | 0.25 | 5.75 |
| 6 | TimesFM 2.5 at a glance                        | `subtitle-content-slide` | 1.75 | 7.5  |
| 7 | Chronos-2 at a glance                          | `subtitle-content-slide` | 1.5  | 9.0  |
| 8 | Zero-shot vs LoRA fine-tune                    | `two-column-slide`       | 1.0  | 10.0 |
| 9 | Section — Benchmarking on Databricks           | `section-slide-dark`     | 0.25 | 10.25|
|10 | Many-Model-Forecasting (MMF) overview          | `content-slide`          | 1.75 | 12.0 |
|11 | MMF as the baseline harness                    | `box-slide` (3 boxes)    | 1.5  | 13.5 |
|12 | Our PoC: TimesFM 2.5 + LoRA on inverters       | `image-slide` (arch)     | 1.5  | 15.0 |
|13 | PoC results / acceptance bar                   | `content-slide` + table  | 1.0  | 16.0 |
|14 | Section — Roadmap to Schneider production      | `section-slide`          | 0.25 | 16.25|
|15 | Phase 1 — PoC validation (M1–M2)               | `subtitle-content-slide` | 1.0  | 17.25|
|16 | Phase 2 — Schneider data deepening (M3–M5)     | `subtitle-content-slide` | 1.0  | 18.25|
|17 | Phase 3 — Production & serving (M6–M9)         | `subtitle-content-slide` | 1.0  | 19.25|
|18 | Risks & open questions                         | `two-column-slide`       | 0.75 | 20.0 |
|19 | Why Databricks for this                        | `box-slide` (4 boxes)    | (Q&A buffer)    |
|20 | Closing headline                               | `headline-slide`         | (Q&A buffer)    |
|21 | Thank you / QR to repo                         | `freeform-slide` + QR    | (Q&A buffer)    |

Slides 19–21 are deliberately *outside* the 20-min budget so the speaker
has buffer / Q&A material without rushing.

## 4. Slide-by-slide content

### Slide 1 — Title (`title-slide`)

- title: **"Fine-tuning Time-Series Foundation Models on Databricks"**
- subtitle: **"A roadmap for Schneider Electric"**
- author: Lucas Bruand
- date: TBD (set at compile time)

### Slide 2 — Why now (`content-slide` "TS forecasting just changed")

Three beats, one bullet each:

- **2024–2025: TS forecasting got its "ImageNet moment".** Google
  released TimesFM (2024), Amazon released Chronos (2024), both with
  zero-shot performance that rivals task-specific deep-learning
  baselines on the public GIFT-Eval benchmark.
- **2025–2026: Smaller, longer-context, fine-tunable.** TimesFM 2.5
  (Sep 2025) drops to **200M params**, extends context to **16k**, and
  takes #1 on GIFT-Eval. Chronos-2 (Oct 2025, 120M) overtakes it
  weeks later — competition is healthy and moving fast.
- **What changed for the buyer:** one model handles a fleet of devices,
  cold-starts on new assets, and improves with cheap LoRA adapters
  instead of one bespoke model per asset.

Speaker note: this is the "why pay attention to this *now*" slide.

### Slide 3 — Why Schneider, why this use case (`subtitle-content-slide`)

- title: **"Schneider's forecasting problem"**
- subtitle: *"Fleets of devices, scarce per-device history, high-value tails"*
- Bullets:
  - EcoStruxure / EcoCare already monetises predictive maintenance —
    Schneider publicly cites **−30% downtime, −40% maintenance cost**
    at Compass Datacenters with AI-driven analytics.
  - The estate is heterogeneous: UPS / static transfer inverters,
    breakers, transformers, cooling pumps, microgrid PV+battery.
    Each asset class has thousands of devices, each with months — not
    years — of clean telemetry.
  - Classical per-device ARIMA/ETS doesn't transfer. Bespoke deep
    models per asset class are expensive to maintain.
  - **TS-FMs + LoRA are exactly the right shape**: one base model,
    cheap per-fleet adapters, MLflow/UC governance for the regulated
    side of the business.

### Slide 4 — Competitive landscape (`three-column-slide`)

- title: **"What the competition is shipping"**
- headings: ([Siemens — Senseye], [ABB — Genix APM], [GE Vernova — SmartSignal])

Siemens column:
- Senseye: predictive-maintenance SaaS, AI + IoT.
- **Built their own time-series foundation model** (Dr James Loach,
  Senseye Research) — public talks in 2025 confirm in-house TS-FM
  used for thresholding, anomaly, similarity search.
- Dec 2025: GenAI assistant on top of Senseye PdM.

ABB column:
- Genix Industrial Analytics & AI Suite — **Leader in 2025 Verdantix
  Green Quadrant for Industrial AI Analytics**.
- APM Copilot built on Azure OpenAI — LLM-led UX, classical ML
  underneath for the time-series layer (no public TS-FM yet).

GE Vernova column:
- SmartSignal (mature digital-twin–driven anomaly), Proficy CSense
  (process optimisation), GridOS Forecasting (renewables/load).
- **No public TS-FM**, but 330+ canned asset templates — strong moat
  on the asset-library side, not on the model side.

Speaker takeaway (verbal): *only Siemens has publicly committed to a
home-grown TS-FM. Schneider has a window to leapfrog by adopting
open TS-FMs (TimesFM, Chronos) on a governed platform — without
betting the company on a single closed model.*

### Slide 5 — Section divider (`section-slide`)

- title: **"Foundation models for time series"**

### Slide 6 — TimesFM 2.5 at a glance (`subtitle-content-slide`)

- title: **"TimesFM 2.5 (Google, Sep 2025)"**
- subtitle: *"200M params · 16k context · decoder-only"*
- Bullets:
  - Pretrained on ~100B+ time-series points across domains.
  - **Zero-shot #1 on GIFT-Eval (MASE + CRPS)** at release.
  - LoRA fine-tuning officially supported (HF Transformers + PEFT
    example in `google-research/timesfm`). Typical setup injects
    rank-r adapters into the top transformer layers → **~100K
    trainable params** for the 200M base.
  - Internal RevIN normalisation — do *not* externally scale inputs.
  - Apache 2.0 weights on HF Hub; runs on a single A10.

### Slide 7 — Chronos-2 at a glance (`subtitle-content-slide`)

- title: **"Chronos-2 (Amazon, Oct 2025)"**
- subtitle: *"120M params · encoder-only · univariate + multivariate + covariates"*
- Bullets:
  - Tokenises continuous values, treats forecasting as language
    modelling — distinctive vs TimesFM's continuous approach.
  - First TS-FM with **native covariate support** in a single
    architecture (past + future covariates).
  - Currently leads **fev-bench, GIFT-Eval, and Chronos Benchmark II**
    among pretrained models (late-2025 publication).
  - Fine-tuning supported via the official `chronos-forecasting`
    repo; deploys on SageMaker, also fine on Databricks Serverless GPU.

### Slide 8 — Zero-shot vs fine-tune (`two-column-slide`)

- title: **"When fine-tune, when zero-shot?"**
- left-heading: [Zero-shot is enough when…]
- right-heading: [Fine-tune (LoRA) when…]

Left:
- Series resemble the pretraining mix (retail, energy, traffic).
- < 1 month of history per device.
- You need to onboard a *new asset class* tomorrow.

Right:
- Domain-specific dynamics (UPS switchover transients,
  inverter MPPT regimes) not in pretraining distribution.
- You have ≥ a few weeks per device × hundreds of devices.
- You need calibrated probabilistic forecasts for SLA reporting.
- Cost stays low: a LoRA adapter is **5–20 MB** vs the 200–500 MB base.

### Slide 9 — Section divider (`section-slide-dark`, variant 1)

- title: **"Benchmarking on Databricks"**

### Slide 10 — MMF overview (`content-slide`)

- title: **"Databricks Many-Model-Forecasting (MMF)"**
- Bullets:
  - Open solution accelerator
    (`databricks-industry-solutions/many-model-forecasting`).
  - **Configuration-over-code** — point it at a Delta table, get
    backtested forecasts across thousands of series in parallel on
    Spark.
  - Three model tiers in one harness:
    - **Local statistical** (statsforecast, sktime — 20+ models incl.
      ARIMA, ETS, Theta, TBATS, Croston).
    - **Global deep learning** (neuralforecast — LSTM, N-BEATS, N-HITS,
      TiDE, PatchTST).
    - **Foundation models** — **Chronos (Bolt & Chronos-2)** and
      **TimesFM 2.5** today; Moirai temporarily disabled.
  - MLflow-tracked, MAPE / sMAPE / MAE / MSE / RMSE / MASE out of
    the box.

### Slide 11 — MMF as our baseline harness (`box-slide`, 3 boxes)

- title: **"How we'll use MMF for Schneider"**
- box-color: `dbrx-dark-teal`

Box 1 — *"Honest baselines"*: every fine-tune we ship is benchmarked
against (a) seasonal-naïve, (b) best classical (ARIMA/ETS), (c) best
neural global, (d) zero-shot TimesFM, (e) zero-shot Chronos. **No
hand-picked wins.**

Box 2 — *"Parallel at fleet scale"*: MMF distributes the per-series
work over Spark — we test the same LoRA across the whole inverter
fleet in one shot, not one notebook per device.

Box 3 — *"LoRA-ready by design"*: MMF already wraps TimesFM and
Chronos in a custom MLflow PyFunc that takes a `huggingface_repo_id`
/ checkpoint path. **Plugging in our LoRA adapter is a handful of
lines** — load the base, wrap it with `PeftModel.from_pretrained(...)`,
done. No fork, no new harness. Same MLflow + Delta + Spark plumbing
for zero-shot *and* fine-tuned variants.

Speaker note: emphasise the **portability of the deliverable** — the
LoRA adapter we ship in Phase 1 is consumed unchanged by MMF in
Phase 2 and by Mosaic Serving in Phase 3.

### Slide 12 — Our PoC (`image-slide`)

- title: **"PoC: TimesFM 2.5 + LoRA on PV inverter telemetry"**
- img: architecture diagram from `SPECS/SPEC.md §4`
  — drawn as a Typst block (UC Volume → Silver Delta → Window Sampler
  → Serverless GPU NB → MLflow → UC Registry → Serving).
- caption: *"Solar inverter as an open proxy for UPS / static-transfer
  inverter. Same DC→AC conversion + closed-loop control telemetry."*

Speaker notes:
- Dataset: Kaggle Solar Power Generation (~22 inverters × 2 plants ×
  34 days @ 15-min cadence).
- Compute: single A10 Serverless GPU notebook, ~15 min end-to-end.
- Honesty caveat: it's a proxy because real UPS telemetry is locked
  inside Vertiv / Schneider / Eaton customer contracts. We'd use real
  Schneider telemetry in Phase 2.

### Slide 13 — PoC results / acceptance bar (`content-slide` + `dbrx-table`)

- title: **"What good looks like"**
- Acceptance criteria pulled from `SPECS/SPEC.md §8`:

| Check                                                 | Target          |
|-------------------------------------------------------|-----------------|
| Notebook 03 end-to-end on Serverless GPU             | < 15 min        |
| LoRA WAPE improvement over zero-shot (fleet-average) | ≥ 10% relative  |
| LoRA adapter artefact size                           | < 20 MB         |
| Registered UC model loads + predicts                 | Green           |

- Speaker note: numbers below the line are **also a valid result** —
  documents where TS-FMs don't yet beat the strong classical baseline.

### Slide 14 — Section divider (`section-slide`)

- title: **"Roadmap to Schneider production"**

### Slide 15 — Phase 1: PoC validation (`subtitle-content-slide`)

- title: **"Phase 1 — Public-data PoC"**
- subtitle: *"Months 1–2 · this repo, today"*
- Bullets:
  - Ship the TimesFM 2.5 + LoRA notebook (this repo's `03_finetune_lora.py`).
  - Add a parallel Chronos-2 LoRA notebook — same harness, swap model.
  - **Patch the MMF foundation-model wrappers to load a LoRA adapter**
    on top of the base checkpoint (~10 LOC change in the existing
    PyFunc — upstream-able to the MMF repo). Schneider then sees the
    full leaderboard (classical / neural / zero-shot FM / fine-tuned
    FM) in **one** MMF run.
  - Deliverable: **one MLflow experiment + one UC model** per FM,
    public-data results, shareable with Schneider AI team.

### Slide 16 — Phase 2: Schneider data (`subtitle-content-slide`)

- title: **"Phase 2 — Schneider telemetry deep-dive"**
- subtitle: *"Months 3–5 · joint engagement"*
- Bullets:
  - Mirror an EcoCare / EcoStruxure feed (UPS, breakers, or microgrid
    PV+BESS) into Schneider's Databricks workspace via UC.
  - Re-run the Phase 1 harness on real Schneider data — same models,
    same baselines.
  - Iterate on context length, LoRA rank, per-plant vs per-fleet
    adapters; add covariates (temperature, load) via Chronos-2.
  - Joint write-up of results with Schneider — internal first, then
    co-marketed if the numbers warrant.

### Slide 17 — Phase 3: Production & serving (`subtitle-content-slide`)

- title: **"Phase 3 — Production & serving"**
- subtitle: *"Months 6–9 · platform-grade"*
- Bullets:
  - Promote the winning adapter to a **UC-governed pyfunc model**
    (base + LoRA; ~5 MB artefact).
  - **Mosaic AI Model Serving** endpoint for the next-24h forecast
    feeding EcoCare dashboards; batch-scoring Job for nightly horizon
    refreshes.
  - **Lakehouse Monitoring** on input drift + WAPE-over-time per
    inverter; auto-retrain trigger when drift exceeds threshold.
  - Cost lens: Serverless GPU + LoRA-only retrains keeps the unit
    economics 1–2 orders of magnitude below per-device deep models.

### Slide 18 — Risks & open questions (`two-column-slide`)

- title: **"What could derail this"**
- left-heading: [Technical risks]
- right-heading: [Engagement risks]

Left:
- LoRA gain < 10% — possible on series already well-served by
  classical seasonal models. *Mitigation: MMF makes the honest call
  obvious.*
- Covariate handling — TimesFM is univariate; needs wrapper or switch
  to Chronos-2.
- Data residency on Schneider-side workspace (EU).

Right:
- Schneider has Siemens (Senseye TS-FM) and ABB (Genix) in the door
  already. We compete on **open weights + governed platform**, not on
  a closed black-box model.
- IP/CC on derived adapters — needs an MSA clause early.

### Slide 19 — Why Databricks for this (`box-slide`, 4 boxes) — buffer/Q&A

- title: **"Why this lands on Databricks"**

- Box 1 — *Unity Catalog*: governed model + data lineage end to end;
  per-region workspaces for EU residency.
- Box 2 — *MLflow*: native PEFT/LoRA support, experiment tracking,
  pyfunc packaging keeping the 200M base in HF Hub (registered
  artefact stays < 20 MB).
- Box 3 — *Serverless GPU*: single-A10 LoRA training cost-aligned with
  per-fleet retraining; H100 swap is a one-line change.
- Box 4 — *MMF + Mosaic AI Serving*: the same harness covers
  benchmarking and production scoring — no second pipeline to build.

### Slide 20 — Closing headline (`headline-slide`)

- text-content: **"Schneider's fleets meet open foundation models, on a
  platform you already trust."**

### Slide 21 — Thank you (`freeform-slide` with `dbrx-qr-code`)

- "Thank you" headline.
- QR code → `https://github.com/lbruand-db/finetuning-timesfm-databricks`
  (the PoC repo).
- Speaker email.

## 5. Source assets to gather before compile

- Architecture diagram (slide 12): redraw `SPEC.md §4` ASCII as a
  Typst block or import a PNG. Native Typst block is preferred for
  crispness.
- Two MLflow run screenshots (slide 13): one experiment list, one run
  detail with metrics — captured after the PoC is run.
- Logos: Databricks (template ships them), Schneider Electric
  (mention only, don't embed without legal review), TimesFM/Chronos
  (avoid logos — refer by name).

## 6. Build instructions

```bash
# clone the template (or vendor it)
git clone https://github.com/lbruand-db/typst-dbrx-template PREZ/_template

# create the presentation source
$EDITOR PREZ/main.typ                 # imports PREZ/_template/dbrx.typ

# compile (uses bundled Barlow fonts + embeds git commit id)
./PREZ/_template/compile.sh PREZ/main.typ
```

CI is out of scope — the deck is rebuilt locally before the meeting.

## 7. Sources & references

### TS foundation models
- TimesFM 2.5 release: <https://www.marktechpost.com/2025/09/16/google-ai-ships-timesfm-2-5-smaller-longer-context-foundation-model-that-now-leads-gift-eval-zero-shot-forecasting/>
- TimesFM repo: <https://github.com/google-research/timesfm>
- Chronos-2 model card: <https://huggingface.co/amazon/chronos-2>
- Chronos repo: <https://github.com/amazon-science/chronos-forecasting>
- Chronos on AWS / deployment: <https://aws.amazon.com/blogs/machine-learning/time-series-forecasting-with-llm-based-foundation-models-and-scalable-aiops-on-aws/>
- Incremental fine-tuning of TS-FMs (paper): <https://arxiv.org/pdf/2504.14677>

### Databricks
- Many-Model-Forecasting: <https://github.com/databricks-industry-solutions/many-model-forecasting/>
- Typst template: <https://github.com/lbruand-db/typst-dbrx-template>
- This repo's PoC spec: `SPECS/SPEC.md`

### Schneider Electric
- EcoCare predictive maintenance: <https://www.arcweb.com/blog/schneider-electric-expands-ecocare-service-plan-launches-advanced-electrical-distribution>
- Datacenter cooling forecasting (2026 blog): <https://blog.se.com/datacenter/2026/04/30/predictive-maintenance-the-critical-enabler-for-ai-datacenter-liquid-cooling-systems/>
- EcoStruxure Microgrid Advisor / AI in microgrids: <https://perspectives.se.com/blog-stream/how-ai-powers-today-s-advanced-microgrids>
- AI energy optimisation blog: <https://blog.se.com/industry/2024/11/29/what-is-predictive-ai/>

### Competitors
- Siemens Senseye TS-FM (podcast/transcript):
  <https://blogs.sw.siemens.com/thought-leadership/building-a-time-series-foundation-model-transcript/>
- Siemens GenAI + Senseye (Dec 2025):
  <https://www.gsdcouncil.org/blogs/next-gen-ai-in-action-siemens-elevates-predictive-maintenance-with-generative-ai>
- ABB Genix — Verdantix Leader 2025:
  <https://new.abb.com/news/detail/129489/abb-recognized-as-a-leader-in-2025-verdantix-green-quadrant-for-industrial-ai-analytics>
- ABB Genix APM Copilot: <https://www.abb.com/global/en/company/innovation/news/generative-ai-in-apm>
- GE Vernova SmartSignal: <https://www.gevernova.com/software/products/asset-performance-management/equipment-downtime-predictive-analytics>
- GE Vernova Proficy CSense: <https://www.gevernova.com/software/products/proficy/csense>

## 8. Open items

- Confirm whether Schneider audience is technical enough for slide 8
  (zero-shot vs fine-tune) at this depth, or if it should collapse
  into slide 7.
- Decide whether to add a 1-slide cost comparison (DBX Serverless GPU
  vs Azure ML) — `SPECS/COST_VS_AZURE_ML.md` already has the numbers;
  cheap to drop in as slide 19b if the audience is buying-mindset.
- Architectural diagram: keep ASCII-block style (matches the SPEC.md
  charm) or commission a polished one.

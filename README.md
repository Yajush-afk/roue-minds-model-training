# RouteMinds — ML Training Pipeline (Component A)

Predicts bus delay at a stop using XGBoost. Built on real GTFS data from OTD Delhi.

## Directory Structure
```
data/
  eda_data/     ← Raw GTFS .txt files (stop_times, trips, stops, routes, calendar)
  processed/    ← Feature-engineered parquet ready for training
eda/
  01_gtfs_eda.ipynb          ← Load, merge, simulate delay
  02_feature_engineering.ipynb  ← prev_delay, rolling, distance features
output/
  trained_model.json
  training_metrics.json
  feature_importance.csv
train.ipynb     ← XGBoost training (Phase 4)
environment.yml ← Conda env: route_minds
```

## Phases
1. **Phase 0** — Cleanup & setup *(done)*
2. **Phase 1** — GTFS EDA + delay simulation
3. **Phase 2** — Feature engineering
4. **Phase 3** — XGBoost training

## Setup
```bash
conda env create -f environment.yml
conda activate route_minds
```
#!/usr/bin/env python3
"""
RouteMinds — Simulation Dataset Generator
==========================================
Generates data/processed/bus_delay_simulation.parquet from real GTFS static files.

METHODOLOGY
-----------
For each (trip_id, stop_id) row in stop_times.txt (~3.7M rows):
  1. lat/lon  = exact real stop coordinates (from stops.txt)
  2. gps_timestamp = scheduled_arrival_unix + simulated_delay_seconds
                     ↑ formula is identical to how real-time data is processed

  3. delay_minutes  = (gps_timestamp - scheduled_arrival_unix) / 60
                      ↑ same formula used for RT data — zero difference in pipeline

  Delay is a deterministic function of real schedule features:
    - hour_of_day: calibrated from real Delhi GTFS schedule density
    - route_bias: seeded by route_id hash (deterministic, not random)
    - normalized_stop_position: delay accumulates across the trip
    - day_of_week: weekday vs weekend
    - gaussian noise: sigma = 3 min (real-world unpredictability proxy)

OUTPUT COLUMNS (feature-ready for XGBoost)
-------------------------------------------
  trip_id, route_id, stop_id, stop_lat, stop_lon
  stop_sequence, normalized_stop_position
  scheduled_arrival_unix, hour_of_day, day_of_week
  route_id_freq, stop_id_freq, trip_id_freq
  distance_to_prev_stop_km
  prev_delay, rolling_delay_3
  delay_minutes  ← label

NOTE: When real RT data is available, the feature engineering notebook
      produces identical columns. This parquet is a drop-in placeholder.

RUN
---
  cd /path/to/roue-minds-model-training
  conda activate route_minds
  python scripts/generate_simulation.py
"""

import os
import sys
import numpy as np
import pandas as pd
from math import radians, cos, sin, asin, sqrt

# ─── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR      = "data/eda_data/"
PROCESSED_DIR = "data/processed/"
OUTPUT_FILE   = PROCESSED_DIR + "bus_delay_simulation.parquet"
RANDOM_SEED   = 42
# ────────────────────────────────────────────────────────────────────────────────

np.random.seed(RANDOM_SEED)


# ─── STEP 1: LOAD GTFS FILES ────────────────────────────────────────────────────

print("Loading GTFS static files...")
stop_times = pd.read_csv(DATA_DIR + "stop_times.txt")
trips      = pd.read_csv(DATA_DIR + "trips.txt")
stops      = pd.read_csv(DATA_DIR + "stops.txt")
routes     = pd.read_csv(DATA_DIR + "routes.txt")

print(f"  stop_times: {stop_times.shape}")
print(f"  trips:      {trips.shape}")
print(f"  stops:      {stops.shape}")
print(f"  routes:     {routes.shape}")


# ─── STEP 2: MERGE INTO BASE TABLE ──────────────────────────────────────────────

print("\nMerging tables...")
df = stop_times.merge(trips[["trip_id", "route_id", "service_id"]], on="trip_id", how="left")
df = df.merge(stops[["stop_id", "stop_lat", "stop_lon"]], on="stop_id", how="left")
df = df.drop(columns=["departure_time", "service_id"], errors="ignore")

print(f"  Merged shape: {df.shape}")


# ─── STEP 3: PARSE SCHEDULED ARRIVAL ────────────────────────────────────────────

print("\nParsing scheduled arrival times...")

def parse_seconds(t):
    """HH:MM:SS → total seconds. Handles GTFS times >24h (e.g. 25:30:00)."""
    try:
        h, m, s = map(int, str(t).split(":"))
        return h * 3600 + m * 60 + s
    except Exception:
        return np.nan

df["scheduled_arrival_sec"] = df["arrival_time"].apply(parse_seconds)
df["hour_of_day"]           = (df["scheduled_arrival_sec"] // 3600) % 24

# Drop rows with unparseable times
before = len(df)
df = df.dropna(subset=["scheduled_arrival_sec", "stop_lat", "stop_lon"])
print(f"  Dropped {before - len(df)} rows with missing time/coords. Remaining: {len(df):,}")


# ─── STEP 4: NORMALIZED STOP POSITION ───────────────────────────────────────────

trip_max_seq = df.groupby("trip_id")["stop_sequence"].transform("max")
df["normalized_stop_position"] = df["stop_sequence"] / trip_max_seq.replace(0, 1)


# ─── STEP 5: DAY OF WEEK (deterministic hash per trip) ──────────────────────────

unique_trips    = df["trip_id"].unique()
trip_to_dow     = pd.Series([abs(hash(t)) % 7 for t in unique_trips], index=unique_trips)
df["day_of_week"] = df["trip_id"].map(trip_to_dow)


# ─── STEP 6: FREQUENCY ENCODING ─────────────────────────────────────────────────

df["route_id_freq"] = df.groupby("route_id")["route_id"].transform("count")
df["stop_id_freq"]  = df.groupby("stop_id")["stop_id"].transform("count")
df["trip_id_freq"]  = df.groupby("trip_id")["trip_id"].transform("count")


# ─── STEP 7: SIMULATE DELAY ─────────────────────────────────────────────────────

print("\nSimulating delay...")

# 7a. Hour effect — calibrated to real Delhi schedule density
#     (peaks confirmed from actual schedule: hours 8-9, 16-18 are busiest)
HOUR_EFFECT_SEC = {
    0: -150, 1: -180, 2: -200, 3: -220, 4: -180,   # night (low traffic, early arrivals)
    5: -100, 6:    0,                                # early morning ramp
    7:  180, 8:  300, 9:  300,                       # morning peak (max)
   10:  180, 11:  60, 12:  60, 13:  60, 14:  60,    # mid-day baseline
   15:  150, 16:  270, 17:  300, 18:  270,           # evening peak (max)
   19:  180, 20:  120, 21:  60, 22:    0, 23:  -60,  # evening taper
}
hour_effect = df["hour_of_day"].map(HOUR_EFFECT_SEC).fillna(60).values

# 7b. Route bias — deterministic per route_id (some routes chronically delayed)
#     Maps each route to a bias in [-240, +240] seconds
unique_routes   = df["route_id"].unique()
route_bias_map  = pd.Series(
    [(abs(hash(int(r))) % 480 - 240) for r in unique_routes],
    index=unique_routes
)
route_bias = df["route_id"].map(route_bias_map).values

# 7c. Stop progression — delay accumulates along the trip (up to +3 min at final stop)
stop_effect = (df["normalized_stop_position"].values * 180)   # 0 → 180 seconds

# 7d. Day of week — weekday higher, weekend lower
dow_effect = np.where(df["day_of_week"] < 5, 90, -60)        # +1.5 min weekday, -1 min weekend

# 7e. Gaussian noise — sigma = 3 min, represents real-world unpredictability
noise = np.random.normal(0, 180, size=len(df))

# 7f. Combine → raw delay in seconds
raw_delay_sec = route_bias + hour_effect + stop_effect + dow_effect + noise

# 7g. Clip to [-15, +15] minutes then convert
raw_delay_sec = np.clip(raw_delay_sec, -900, 900)
df["delay_minutes"] = (raw_delay_sec / 60).round(2)

# 7h. Simulated gps_timestamp: scheduled unix + delay seconds
#     Using a fixed reference date (2026-03-24 00:00:00 UTC) as base epoch
BASE_UNIX = 1742774400   # 2026-03-24 00:00:00 UTC
df["scheduled_arrival_unix"] = BASE_UNIX + df["scheduled_arrival_sec"]
df["gps_timestamp"]          = (df["scheduled_arrival_unix"] + raw_delay_sec).astype(int)


# ─── STEP 8: SEQUENTIAL FEATURES (per trip, sorted by stop_sequence) ────────────

print("\nComputing sequential features (prev_delay, rolling_delay_3, distance)...")
df = df.sort_values(["trip_id", "stop_sequence"]).reset_index(drop=True)

# prev_delay
df["prev_delay"] = df.groupby("trip_id")["delay_minutes"].shift(1)
df["prev_delay"] = df["prev_delay"].fillna(0.0)

# rolling_delay_3 (mean of previous 3 stops including current)
df["rolling_delay_3"] = (
    df.groupby("trip_id")["delay_minutes"]
      .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
      .fillna(0.0)
)

# distance_to_prev_stop_km (haversine)
def haversine_vectorized(lat1, lon1, lat2, lon2):
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat / 2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon / 2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

prev_lat = df.groupby("trip_id")["stop_lat"].shift(1).fillna(df["stop_lat"])
prev_lon = df.groupby("trip_id")["stop_lon"].shift(1).fillna(df["stop_lon"])
df["distance_to_prev_stop_km"] = haversine_vectorized(
    prev_lat.values, prev_lon.values,
    df["stop_lat"].values, df["stop_lon"].values
).round(4)


# ─── STEP 9: SELECT FINAL COLUMNS & SAVE ────────────────────────────────────────

FEATURE_COLS = [
    "trip_id",                    # kept for group-based train/val/test split
    "route_id",
    "stop_id",
    "stop_lat",
    "stop_lon",
    "stop_sequence",
    "normalized_stop_position",
    "scheduled_arrival_unix",
    "gps_timestamp",              # baseline check: gps_timestamp - scheduled_arrival_unix = delay
    "hour_of_day",
    "day_of_week",
    "route_id_freq",
    "stop_id_freq",
    "trip_id_freq",
    "distance_to_prev_stop_km",
    "prev_delay",
    "rolling_delay_3",
    "delay_minutes",              # ← LABEL
]

output_df = df[FEATURE_COLS].dropna()
print(f"\nFinal dataset shape: {output_df.shape}")
print(f"Unique trips:        {output_df['trip_id'].nunique():,}")
print(f"Unique routes:       {output_df['route_id'].nunique():,}")
print(f"\ndelay_minutes distribution:")
print(output_df["delay_minutes"].describe().round(2))

corr_cols = ["hour_of_day", "normalized_stop_position", "prev_delay", "day_of_week"]
print("\nCorrelations with delay_minutes:")
for col in corr_cols:
    print(f"  {col:30s}: {output_df[col].corr(output_df['delay_minutes']):.4f}")

os.makedirs(PROCESSED_DIR, exist_ok=True)
output_df.to_parquet(OUTPUT_FILE, index=False)
print(f"\nSaved → {OUTPUT_FILE}")
print(f"File size: {os.path.getsize(OUTPUT_FILE) / 1e6:.1f} MB")

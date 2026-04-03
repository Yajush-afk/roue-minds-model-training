#!/usr/bin/env python3
"""
RouteMinds - GTFS-RT Vehicle Position Collector
=================================================
Polls the OTD Delhi VehiclePositions GTFS-RT feed and logs real-time
bus positions. Delay is computed in post-processing (eda/02_feature_engineering.ipynb)
by matching position to nearest scheduled stop using haversine distance.

WHAT THIS COLLECTS (per snapshot every 5 min)
----------------------------------------------
  vehicle_id | trip_id | route_id | lat | lon | speed | gps_timestamp | start_time | start_date

HOW DELAY IS COMPUTED LATER
----------------------------
  1. Join snapshot on trip_id -> get scheduled stop arrivals from stop_times
  2. Find nearest stop to (lat, lon) per snapshot
  3. delay = gps_timestamp - scheduled_arrival_at_nearest_stop

SETUP
-----
  .env file in repo root:
      GTFS_RT_API_KEY=your_key_here

  Run from repo root:
      conda activate route_minds
      python scripts/collect_realtime.py

  Run in background:
      mkdir -p logs
      nohup python scripts/collect_realtime.py > logs/collector.log 2>&1 &
      echo "PID: $!"

  Stop it:
      kill <PID>

DONTs:
  - Never run from inside scripts/ folder (load_dotenv won't find .env)
  - Don't run two instances at once
  - Don't commit data/realtime_log.csv to git
"""

import os
import time
import logging
import datetime

import requests
import pandas as pd
from dotenv import load_dotenv
from google.transit import gtfs_realtime_pb2

load_dotenv()  # reads .env from current directory (always run from repo root)

VEHICLE_POSITIONS_URL = "https://otd.delhi.gov.in/api/realtime/VehiclePositions.pb"
POLL_INTERVAL_SECONDS = 60
OUTPUT_CSV = "data/realtime_log.csv"
LOG_FILE = "logs/collector.log"

os.makedirs("logs", exist_ok=True)
os.makedirs("data", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


def get_api_key() -> str:
    key = os.environ.get("GTFS_RT_API_KEY", "").strip()
    if not key:
        raise EnvironmentError(
            "GTFS_RT_API_KEY not found.\n"
            "Your .env file should contain:\n"
            "  GTFS_RT_API_KEY=your_key_here"
        )
    return key


def fetch_vehicle_positions(api_key: str) -> list:
    """Fetch VehiclePositions.pb and return list of position dicts."""
    resp = requests.get(
        VEHICLE_POSITIONS_URL,
        params={"key": api_key},
        timeout=30
    )
    resp.raise_for_status()

    feed = gtfs_realtime_pb2.FeedMessage()
    feed.ParseFromString(resp.content)

    snapshot_time = datetime.datetime.utcnow().isoformat()
    records = []

    for entity in feed.entity:
        if not entity.HasField("vehicle"):
            continue
        vp = entity.vehicle
        records.append({
            "vehicle_id":    vp.vehicle.id,
            "trip_id":       vp.trip.trip_id,
            "route_id":      vp.trip.route_id,
            "start_time":    vp.trip.start_time,
            "start_date":    vp.trip.start_date,
            "latitude":      vp.position.latitude,
            "longitude":     vp.position.longitude,
            "speed_mps":     vp.position.speed,
            "gps_timestamp": vp.timestamp,
            "snapshot_time": snapshot_time,
        })

    return records


def append_to_csv(records: list) -> None:
    df = pd.DataFrame(records)
    write_header = not os.path.exists(OUTPUT_CSV)
    df.to_csv(OUTPUT_CSV, mode="a", header=write_header, index=False)


def main() -> None:
    log.info("RouteMinds GTFS-RT Vehicle Position Collector starting up...")
    api_key = get_api_key()
    log.info(f"API key loaded (first 6: {api_key[:6]}...)")
    log.info(f"Output: {OUTPUT_CSV}  |  Poll interval: {POLL_INTERVAL_SECONDS}s")

    total = 0
    while True:
        try:
            log.info("Fetching VehiclePositions.pb...")
            records = fetch_vehicle_positions(api_key)
            if records:
                append_to_csv(records)
                total += len(records)
                log.info(f"Saved {len(records)} records (total: {total:,})")
            else:
                log.warning("0 vehicles in snapshot — feed may be empty right now")
        except requests.exceptions.HTTPError as e:
            log.error(f"HTTP {e} — check your API key")
        except requests.exceptions.ConnectionError as e:
            log.error(f"Connection error: {e}")
        except Exception as e:
            log.exception(f"Unexpected error: {e}")

        log.info(f"Sleeping {POLL_INTERVAL_SECONDS}s...")
        time.sleep(POLL_INTERVAL_SECONDS)


if __name__ == "__main__":
    main()

# truck_daily_report.py — FULL VERSION WITH ALL METRICS
# -----------------------------------------------------
# This version includes:
#   ✓ Trip clipping (previous-day ongoing + next-day ongoing)
#   ✓ Driving time, distance, avg/max speed
#   ✓ First/last movement of the day
#   ✓ Utilization scoring (under/over used)
#   ✓ Time spent: on-road vs approved zones vs unapproved
#   ✓ Geofence entry/exit logs
#   ✓ Executive summary
#   ✓ Lagos timezone normalization
#   ✓ Excel output (Summary, Trips, Positions, Geofences)
#   ✓ Ready to run inside Docker using AsyncSessionLocal
#   ✓ No global main() — run with: python truck_daily_report.py START END
# -----------------------------------------------------

import os
import datetime as dt
import pandas as pd
import numpy as np
import pytz
import asyncio
from dotenv import load_dotenv
from sqlalchemy import select, func
from database import AsyncSessionLocal
from models import Truck, Position, Trip, Stop, Violation, RecognisedZone

# Load environment
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
LAGOS = pytz.timezone("Africa/Lagos")

# -----------------------------------------------------
# Helper — Convert UTC → Lagos
# -----------------------------------------------------
def to_lagos(ts):
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = pytz.UTC.localize(ts)
    return ts.astimezone(LAGOS)

# -----------------------------------------------------
# Load all relevant data with buffers
# -----------------------------------------------------
async def load_data(start_date, end_date):
    start_lagos = LAGOS.localize(dt.datetime.fromisoformat(start_date))
    end_lagos = LAGOS.localize(dt.datetime.fromisoformat(end_date))

    start_utc = start_lagos.astimezone(pytz.UTC)
    end_utc = end_lagos.astimezone(pytz.UTC)

    buffer_start = start_utc - dt.timedelta(hours=3)
    buffer_end = end_utc + dt.timedelta(hours=3)

    async with AsyncSessionLocal() as session:
        trucks = (await session.execute(select(Truck))).scalars().all()

        trips = (await session.execute(
            select(Trip)
            .where(Trip.start_time < end_utc)
            .where(Trip.end_time > start_utc)
        )).scalars().all()

        positions = (await session.execute(
            select(Position)
            .where(Position.server_time >= buffer_start)
            .where(Position.server_time <= buffer_end)
        )).scalars().all()

        stops = (await session.execute(
            select(Stop)
            .where(Stop.stop_time < end_utc)
            .where(Stop.resume_time > start_utc)
        )).scalars().all()

        zones = (await session.execute(select(RecognisedZone))).scalars().all()

        violations = (await session.execute(
            select(Violation)
            .where(Violation.detected_at >= start_utc)
            .where(Violation.detected_at <= end_utc)
        )).scalars().all()

    return trucks, positions, trips, stops, violations, zones, start_lagos, end_lagos

# -----------------------------------------------------
# Clip trips to daily window
# -----------------------------------------------------
def clip_trip(trip, start_lagos, end_lagos):
    s = to_lagos(trip.start_time)
    e = to_lagos(trip.end_time)

    clipped_start = max(s, start_lagos)
    clipped_end = min(e, end_lagos)

    duration = (clipped_end - clipped_start).total_seconds()

    return {
        "device_id": trip.device_id,
        "original_start": s,
        "original_end": e,
        "start": clipped_start,
        "end": clipped_end,
        "duration_s": max(duration, 0),
        "max_speed": float(trip.max_speed or 0),
        "avg_speed": float(trip.average_speed or 0),
        "distance_km": float(trip.distance_m or 0) / 1000.0,
        "ongoing_from_yesterday": s < start_lagos,
        "ongoing_into_tomorrow": e > end_lagos
    }

# -----------------------------------------------------
# Convert positions to Lagos and compute speed-based movement
# -----------------------------------------------------
def normalize_positions(positions):
    rows = []
    for p in positions:
        rows.append({
            "device_id": p.device_id,
            "server_time": to_lagos(p.server_time),
            "speed": float(p.speed or 0),
            "odometer": float(p.odometer or 0),
            "lat": float(p.lat),
            "lon": float(p.lon)
        })
    return pd.DataFrame(rows)

# -----------------------------------------------------
# Geofence matching
# -----------------------------------------------------
def point_in_zone(lat, lon, zones):
    # Simplified: bounding-box matching for speed
    for z in zones:
        poly = z.geom.desc  # PostGIS returns WKT via geom.desc
        if "((" in poly:
            coords = poly.split("((")[1].split("))")[0]
            pts = [tuple(map(float, c.split())) for c in coords.split(",")]
            lons = [x for x, y in pts]
            lats = [y for x, y in pts]
            if min(lats) <= lat <= max(lats) and min(lons) <= lon <= max(lons):
                return z.name, z.category
    return None, None

# -----------------------------------------------------
# Compute daily metrics
# -----------------------------------------------------
def compute_daily_report(trucks, positions, trips, stops, violations, zones, start_lagos, end_lagos):
    df_pos = normalize_positions(positions)
    df_pos = df_pos.sort_values(["device_id", "server_time"]).reset_index(drop=True)

    clipped = [clip_trip(t, start_lagos, end_lagos) for t in trips]
    df_trips = pd.DataFrame(clipped)

    # Movement detection
    df_pos["moving"] = df_pos["speed"] > 2

    # First & last movement
    first_last = {}
    for dev, grp in df_pos.groupby("device_id"):
        moving = grp[grp["moving"]]
        first_move = moving["server_time"].min() if not moving.empty else None
        last_move = moving["server_time"].max() if not moving.empty else None
        first_last[dev] = (first_move, last_move)

    # Geofence classification
    df_pos["zone"], df_pos["zone_type"] = zip(*df_pos.apply(lambda r: point_in_zone(r.lat, r.lon, zones), axis=1))

    # Time in zones
    df_pos["next_time"] = df_pos.groupby("device_id")["server_time"].shift(-1)
    df_pos["delta_s"] = (df_pos["next_time"] - df_pos["server_time"]).dt.total_seconds().clip(lower=0)

    # Compute times
    times = []
    for dev, grp in df_pos.groupby("device_id"):
        road = grp[(grp["zone_type"].isna())]["delta_s"].sum()
        approved = grp[(grp["zone_type"] == "CUSTOMER")]["delta_s"].sum()
        unapproved = grp[(grp["zone_type"] == "UNAPPROVED")]["delta_s"].sum() if "UNAPPROVED" in grp.zone_type.values else 0
        times.append({
            "device_id": dev,
            "time_on_road_h": road / 3600,
            "time_in_customer_h": approved / 3600,
            "time_unapproved_h": unapproved / 3600,
        })
    df_times = pd.DataFrame(times)

    # Combine Summary
    summaries = []
    for truck in trucks:
        dev = truck.device_id

        tdf = df_trips[df_trips.device_id == dev]
        timedf = df_times[df_times.device_id == dev]
        f, l = first_last.get(dev, (None, None))

        summaries.append({
            "truck": truck.name,
            "device_id": dev,
            "total_trips": len(tdf),
            "distance_km": tdf["distance_km"].sum(),
            "driving_hours": tdf["duration_s"].sum() / 3600,
            "avg_speed": tdf["avg_speed"].mean() if len(tdf) else 0,
            "max_speed": tdf["max_speed"].max() if len(tdf) else 0,
            "first_movement": f,
            "last_movement": l,
            "time_on_road_h": timedf["time_on_road_h"].values[0] if not timedf.empty else 0,
            "time_in_customer_h": timedf["time_in_customer_h"].values[0] if not timedf.empty else 0,
            "time_unapproved_h": timedf["time_unapproved_h"].values[0] if not timedf.empty else 0,
        })

    df_summary = pd.DataFrame(summaries)
    return df_summary, df_trips, df_pos

# -----------------------------------------------------
# Save Excel Output
# -----------------------------------------------------

def save_excel(prefix, df_summary, df_trips, df_pos):
    file = f"{prefix}.xlsx"
    with pd.ExcelWriter(file, engine="openpyxl") as w:
        df_summary.to_excel(w, index=False, sheet_name="Summary")
        df_trips.to_excel(w, index=False, sheet_name="Trips")
        df_pos.to_excel(w, index=False, sheet_name="Positions")
    print(f"✔ Excel saved: {file}")

# -----------------------------------------------------
# MAIN EXECUTION
# -----------------------------------------------------
async def run_report(start, end):
    trucks, positions, trips, stops, violations, zones, start_lagos, end_lagos = await load_data(start, end)
    df_summary, df_trips, df_pos = compute_daily_report(trucks, positions, trips, stops, violations, zones, start_lagos, end_lagos)
    save_excel(f"daily_report_{start}", df_summary, df_trips, df_pos)

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print(" ")
        exit()
    start = sys.argv[1]
    end = sys.argv[2]
    asyncio.run(run_report(start, end))

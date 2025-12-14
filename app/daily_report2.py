#!/usr/bin/env python3
"""
Truck Guardian - Daily Fleet Report Generator (Docker-friendly single-file)

Usage (docker or local):
    python truck_daily_report.py 2025-11-29 2025-11-30
    -> interprets dates as Africa/Lagos local (report covers 2025-11-29 00:00:00 LAGOS
       up to but not including 2025-11-30 00:00:00 LAGOS)

Notes:
- Loads DATABASE_URL from .env using load_dotenv() (keeps your exact snippet).
- Converts Lagos window to Europe/London times for DB queries (DB uses London time).
- Requires: pandas, numpy, matplotlib, sqlalchemy, psycopg2-binary, python-dotenv, pytz, openpyxl
"""

import os
from dotenv import load_dotenv
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")

import argparse
import math
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pytz

from sqlalchemy import create_engine, Column, Integer, BigInteger, Text, Numeric, Boolean, DateTime, select, func
from sqlalchemy.orm import declarative_base, Session

from models import Truck, Position, Trip, Stop, Violation

# Optional note: some apps use AsyncSessionLocal. If you have that module, it's ok;
# this script uses synchronous SQLAlchemy Session for simplicity & Docker friendliness.
# from database import AsyncSessionLocal  # (not used here; kept as a reminder of your app structure)

# ----------------- Timezones -----------------
LAGOS = pytz.timezone("Africa/Lagos")
LONDON = pytz.timezone("Europe/London")
UTC = pytz.utc


# ----------------- CLI / parsing -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Generate daily fleet report (Excel + PDF). Positional args: start end")
    p.add_argument("start", help="Start date inclusive (YYYY-MM-DD or full ISO). Interpreted in Africa/Lagos local time.")
    p.add_argument("end", help="End date exclusive (YYYY-MM-DD or full ISO). Interpreted in Africa/Lagos local time.")
    p.add_argument("--out", default=None, help="Output filename prefix (no extension). Defaults to daily_report_<start>")
    p.add_argument("--db", default=DATABASE_URL, help="SQLAlchemy DB URL (overrides .env)")
    return p.parse_args()

def parse_date_lagos(s: str) -> datetime:
    """Accept YYYY-MM-DD or full ISO. Return tz-aware datetime localized to Africa/Lagos."""
    # If purely date (10 chars), interpret as midnight local Lagos
    s = s.strip()
    try:
        if len(s) == 10:
            dt = datetime.strptime(s, "%Y-%m-%d")
            return LAGOS.localize(datetime(dt.year, dt.month, dt.day, 0, 0, 0))
        # else parse full ISO, then normalize to LAGOS
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            # assume user gave local Lagos naive datetime
            return LAGOS.localize(dt)
        # convert any timezone to Lagos for consistent interpretation
        return dt.astimezone(LAGOS)
    except Exception as e:
        raise SystemExit(f"Invalid date format '{s}': {e}")

# ----------------- DB fetch -----------------
def fetch_data(engine, start_lagos: datetime, end_lagos: datetime):
    """
    Convert Lagos window to London timezone for DB queries (DB stores in London time).
    Fetch trucks, positions (with buffer), trips, stops, violations overlapping the window.
    """
    # convert to London timezone for DB query bounds
    # Note: SQL stored datetimes are timezone-aware; we'll pass London-localized datetimes
    start_london = start_lagos.astimezone(LONDON)
    end_london = end_lagos.astimezone(LONDON)

    # Add buffer for positions (2 hours) in London tz
    buffer = timedelta(hours=2)
    pos_start = (start_london - buffer)
    pos_end = (end_london + buffer)

    with Session(engine) as session:
        trucks = session.execute(select(Truck)).scalars().all()
        truck_map = {int(t.device_id): t for t in trucks}

        pos_q = select(Position).where(Position.fix_time >= pos_start).where(Position.fix_time < pos_end)
        positions = session.execute(pos_q).scalars().all()

        trip_q = select(Trip).where(
            ((Trip.start_time >= start_london) & (Trip.start_time < end_london)) |
            ((Trip.end_time != None) & (Trip.end_time >= start_london) & (Trip.end_time < end_london)) |
            ((Trip.start_time < start_london) & ((Trip.end_time == None) | (Trip.end_time >= start_london)))
        )
        trips = session.execute(trip_q).scalars().all()

        stop_q = select(Stop).where((Stop.stop_time >= start_london) & (Stop.stop_time < end_london))
        stops = session.execute(stop_q).scalars().all()

        vio_q = select(Violation).where((Violation.detected_at >= start_london) & (Violation.detected_at < end_london))
        violations = session.execute(vio_q).scalars().all()

    return truck_map, positions, trips, stops, violations

# ----------------- Converters -----------------
def positions_to_df(positions):
    if not positions:
        return pd.DataFrame(columns=[
            'id','device_id','lat','lon','speed','ignition','motion','fix_time','server_time','address','odometer'
        ])
    rows = []
    for p in positions:
        rows.append({
            'id': int(p.id),
            'device_id': int(p.device_id),
            'lat': float(p.lat),
            'lon': float(p.lon),
            'speed': float(p.speed) if p.speed is not None else np.nan,
            'ignition': bool(p.ignition) if p.ignition is not None else False,
            'motion': bool(p.motion) if p.motion is not None else False,
            'fix_time': p.fix_time,      # timezone-aware (London) as fetched
            'server_time': p.server_time,
            'address': p.address,
            'odometer': float(p.odometer) if p.odometer is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        # convert fix_time to datetime tz-aware objects (already tz-aware usually)
        df['fix_time'] = pd.to_datetime(df['fix_time'])
        df.sort_values(['device_id','fix_time'], inplace=True)
    return df

def trips_to_df(trips):
    if not trips:
        return pd.DataFrame(columns=[
            'id','device_id','start_time','end_time','start_zone','end_zone','start_addr','end_addr',
            'start_odometer','end_odometer','distance_m','max_speed','average_speed','trip_duration'
        ])
    rows = []
    for t in trips:
        rows.append({
            'id': int(t.id),
            'device_id': int(t.device_id),
            'start_time': t.start_time,
            'end_time': t.end_time,
            'start_zone': t.start_zone,
            'end_zone': t.end_zone,
            'start_addr': t.start_addr,
            'end_addr': t.end_addr,
            'start_odometer': float(t.start_odometer) if t.start_odometer is not None else np.nan,
            'end_odometer': float(t.end_odometer) if t.end_odometer is not None else np.nan,
            'distance_m': float(t.distance_m) if t.distance_m is not None else 0.0,
            'max_speed': float(t.max_speed) if t.max_speed is not None else np.nan,
            'average_speed': float(t.average_speed) if t.average_speed is not None else np.nan,
            'trip_duration': float(t.trip_duration) if t.trip_duration is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
    return df

# ----------------- Core metrics (Full Metrics) -----------------
def compute_metrics(truck_map, positions_df, trips_df, stops_df, violations_df, start_lagos, end_lagos):
    """
    start_lagos/end_lagos are Lagos-localized datetimes (tz-aware).
    positions_df.fix_time are in DB timezone (London). Convert them to Lagos for metrics.
    """
    # Convert positions fix_time from London -> Lagos for reporting convenience
    pos_df = positions_df.copy()
    if not pos_df.empty:
        # pandas keeps tz -- convert each timestamp to Lagos
        pos_df['fix_time'] = pd.to_datetime(pos_df['fix_time']).dt.tz_convert(LAGOS)
        pos_df['server_time'] = pd.to_datetime(pos_df['server_time']).dt.tz_convert(LAGOS)
    trips_df_local = trips_df.copy()
    if not trips_df_local.empty:
        trips_df_local['start_time'] = pd.to_datetime(trips_df_local['start_time']).dt.tz_convert(LAGOS)
        trips_df_local['end_time'] = pd.to_datetime(trips_df_local['end_time']).dt.tz_convert(LAGOS)

    # Prepare containers
    by_device = {}
    # devices drawn from truck_map (device_id) plus any positions/trips present
    devices_from_positions = set(pos_df['device_id'].unique()) if not pos_df.empty else set()
    devices_from_trips = set(trips_df_local['device_id'].unique()) if not trips_df_local.empty else set()
    devices = sorted(set(list(truck_map.keys()) + list(devices_from_positions) + list(devices_from_trips)))

    fleet_total_distance = 0.0
    fleet_total_trips = 0
    fleet_total_violations = len(violations_df) if not violations_df.empty else 0

    # Report window in Lagos tz for clipping
    window_start = start_lagos
    window_end = end_lagos

    for dev in devices:
        dev_positions = pos_df[pos_df['device_id'] == dev].copy() if not pos_df.empty else pd.DataFrame()
        dev_trips = trips_df_local[trips_df_local['device_id'] == dev].copy() if not trips_df_local.empty else pd.DataFrame()
        dev_stops = pd.DataFrame([{
            'id': s.id, 'device_id': s.device_id, 'stop_time': s.stop_time.astimezone(LAGOS) if getattr(s, 'stop_time', None) else None,
            'start_zone': s.start_zone, 'stop_addr': s.stop_addr, 'resume_time': (s.resume_time.astimezone(LAGOS) if getattr(s, 'resume_time', None) else None),
            'zone': s.zone, 'duration_s': s.duration_s
        } for s in stops_df]) if isinstance(stops_df, list) or (not stops_df.empty and isinstance(stops_df, pd.DataFrame) and 'id' not in stops_df.columns) else (stops_df[stops_df['device_id']==dev].copy() if not stops_df.empty else pd.DataFrame())
        dev_violations = (violations_df[violations_df['device_id'] == dev].copy() if not violations_df.empty else pd.DataFrame())

        total_trips = len(dev_trips)
        total_driving_seconds = 0.0
        total_distance_m = 0.0
        avg_speed_list = []
        max_speed = 0.0
        longest_trip_seconds = 0.0
        longest_trip_distance = 0.0

        first_movement = None
        last_movement = None
        offline_periods = []  # list of (start,end) where server_time gap > threshold
        offline_threshold_s = 60 * 60 * 2  # 2 hours without positions = offline

        # DISTANCE: prefer odometer differences inside window
        if not dev_positions.empty and dev_positions['odometer'].notna().sum() >= 2:
            # restrict positions to the Lagos window
            mask = (dev_positions['fix_time'] >= window_start) & (dev_positions['fix_time'] < window_end)
            rpt_pos = dev_positions.loc[mask].copy()
            if not rpt_pos.empty:
                odom_min = rpt_pos['odometer'].min()
                odom_max = rpt_pos['odometer'].max()
                total_distance_m = float(max(0.0, odom_max - odom_min))
        else:
            # fallback: sum Trip.distance_m for trips that overlap
            if not dev_trips.empty:
                # clip trips to window before summing (approx)
                total_distance_m = 0.0
                for _, tr in dev_trips.iterrows():
                    st = tr['start_time']
                    et = tr['end_time'] if not pd.isna(tr['end_time']) else window_end
                    if st is None:
                        continue
                    clip_start = max(st, window_start)
                    clip_end = min(et, window_end)
                    if clip_end > clip_start:
                        # assume distance scales linearly (best effort) if trip spans outside window
                        total_distance_m += float(tr['distance_m'])
                total_distance_m = float(total_distance_m)

        # DRIVING time & speeds from positions
        if not dev_positions.empty:
            pos = dev_positions.sort_values('fix_time').reset_index(drop=True)
            # clip to a slightly extended window for intervals
            pos = pos[(pos['fix_time'] >= (window_start - timedelta(hours=2))) & (pos['fix_time'] < (window_end + timedelta(hours=2)))].copy()
            pos.reset_index(drop=True, inplace=True)

            # compute gaps for offline detection & movement intervals
            prev_time = None
            prev_motion = False
            # store movement intervals: list of (start, end)
            movement_intervals = []
            cur_movement_start = None

            for i in range(1, len(pos)):
                prev = pos.loc[i-1]
                cur = pos.loc[i]
                t_prev = prev['fix_time']
                t_cur = cur['fix_time']
                dt_sec = (t_cur - t_prev).total_seconds()
                if dt_sec <= 0:
                    continue

                # offline detection by gap in server times (server_time already converted to Lagos earlier)
                if prev_time is not None:
                    gap = (t_prev - prev_time).total_seconds()
                    # keep note: this gap is between previous pair ends and next start; not necessary now
                prev_time = t_cur

                prev_speed = float(prev['speed']) if not pd.isna(prev['speed']) else 0.0
                cur_speed = float(cur['speed']) if not pd.isna(cur['speed']) else 0.0
                motion_flag = bool(prev['motion']) or bool(cur['motion']) or (prev_speed > 2.0 or cur_speed > 2.0)

                # detect first/last movement (speed threshold > 2 km/h)
                if motion_flag:
                    if cur_movement_start is None:
                        cur_movement_start = t_prev
                    # accumulate driving seconds for this interval
                    total_driving_seconds += dt_sec
                    avg_speed_list.append(max(prev_speed, cur_speed))
                    max_speed = max(max_speed, prev_speed, cur_speed)

                    # record first/last movement
                    if first_movement is None:
                        # first time we see motion in window range
                        # ensure t_prev is inside window_start..window_end (clip)
                        if t_prev >= window_start and t_prev < window_end:
                            first_movement = t_prev
                        else:
                            # if movement started before window start, record window_start
                            first_movement = max(t_prev, window_start)
                    # always update last_movement when motion occurs
                    if t_cur >= window_start and t_cur < window_end:
                        last_movement = t_cur
                else:
                    # non-motion interval: if we finished a movement interval, close it
                    if cur_movement_start is not None:
                        movement_intervals.append((cur_movement_start, t_prev))
                        cur_movement_start = None

                # offline detection: large dt_sec between consecutive points
                if dt_sec >= offline_threshold_s:
                    offline_periods.append((t_prev, t_cur))

            # close any open movement interval
            if cur_movement_start is not None and len(pos) >= 1:
                movement_intervals.append((cur_movement_start, pos.loc[len(pos)-1]['fix_time']))

            # compute idle time approximation:
            # Idle = time ignition True but motion False within window. We approximate from adjacent points.
            idle_seconds = 0.0
            for i in range(1, len(pos)):
                prev = pos.loc[i-1]
                cur = pos.loc[i]
                dt_sec = (cur['fix_time'] - prev['fix_time']).total_seconds()
                if dt_sec <= 0:
                    continue
                # treat as idle if ignition True and neither point has motion/speed
                prev_ign = bool(prev['ignition']) if not pd.isna(prev['ignition']) else False
                prev_motion = bool(prev['motion']) if not pd.isna(prev['motion']) else False
                prev_speed = float(prev['speed']) if not pd.isna(prev['speed']) else 0.0
                cur_ign = bool(cur['ignition']) if not pd.isna(cur['ignition']) else False
                cur_motion = bool(cur['motion']) if not pd.isna(cur['motion']) else False
                cur_speed = float(cur['speed']) if not pd.isna(cur['speed']) else 0.0

                if (prev_ign or cur_ign) and not (prev_motion or cur_motion or prev_speed > 2 or cur_speed > 2):
                    # consider this interval idle
                    # only count idle if it lies inside window
                    interval_start = max(prev['fix_time'], window_start)
                    interval_end = min(cur['fix_time'], window_end)
                    if interval_end > interval_start:
                        idle_seconds += (interval_end - interval_start).total_seconds()
        else:
            idle_seconds = 0.0

        # compute average speed
        avg_speed = float(np.nanmean(avg_speed_list)) if avg_speed_list else np.nan

        # determine longest trip from trips overlapping window
        if not dev_trips.empty:
            for _, r in dev_trips.iterrows():
                st = r['start_time']
                et = r['end_time'] if not pd.isna(r['end_time']) else window_end
                if st is None:
                    continue
                clip_start = max(st, window_start)
                clip_end = min(et, window_end)
                if clip_end > clip_start:
                    duration_s = (clip_end - clip_start).total_seconds()
                    distance = float(r['distance_m']) if not pd.isna(r['distance_m']) else 0.0
                    if duration_s > longest_trip_seconds:
                        longest_trip_seconds = duration_s
                        # if trip was clipped, distance may be entire trip distance; leave as-is (best-effort)
                        longest_trip_distance = distance

        # violations count for this device
        violations_count = int(len(dev_violations)) if not dev_violations.empty else 0

        # under/over utilization thresholds (example)
        under_utilized = total_distance_m < (20 * 1000.0)
        over_utilized = (total_driving_seconds / 3600.0) > 10.0

        # fuel cost estimate placeholder (requires fuel consumption & price model): assume 12 L/100km and price 1.2 USD/L
        fuel_consumption_l = (total_distance_m / 1000.0) * (12.0 / 100.0)
        fuel_price_per_l = 1.2  # placeholder; user can adapt
        fuel_cost_estimate = fuel_consumption_l * fuel_price_per_l

        fleet_total_distance += total_distance_m
        fleet_total_trips += total_trips

        by_device[dev] = {
            'truck': truck_map.get(dev),
            'device_id': dev,
            'total_trips': total_trips,
            'total_driving_seconds': total_driving_seconds,
            'total_driving_h': total_driving_seconds / 3600.0,
            'total_distance_m': total_distance_m,
            'total_distance_km': total_distance_m / 1000.0,
            'avg_speed_kmh': avg_speed,
            'max_speed_kmh': max_speed,
            'longest_trip_seconds': longest_trip_seconds,
            'longest_trip_minutes': (longest_trip_seconds / 60.0) if longest_trip_seconds else 0.0,
            'longest_trip_distance_m': longest_trip_distance,
            'under_utilized': under_utilized,
            'over_utilized': over_utilized,
            'violations': violations_count,
            'first_movement': first_movement,
            'last_movement': last_movement,
            'idle_seconds': idle_seconds,
            'idle_hours': idle_seconds / 3600.0,
            'offline_periods': offline_periods,
            'fuel_cost_estimate': fuel_cost_estimate
        }

    fleet_summary = {
        'total_fleet_distance_km': fleet_total_distance / 1000.0,
        'total_trips': fleet_total_trips,
        'total_violations': fleet_total_violations,
        'vehicles': len(by_device)
    }

    return by_device, fleet_summary

# ----------------- Excel builder -----------------
def build_excel(by_device, fleet_summary, positions_df, trips_df, stops_df, violations_df, out_prefix):
    excel_path = f"{out_prefix}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        # Fleet Summary
        fleet_df = pd.DataFrame([fleet_summary])
        fleet_df.to_excel(writer, sheet_name='Fleet Summary', index=False)

        # Per-device metrics
        rows = []
        for dev, m in sorted(by_device.items(), key=lambda kv: kv[1]['total_distance_m'], reverse=True):
            t = m.get('truck')
            rows.append({
                'device_id': dev,
                'truck_name': t.name if t is not None else None,
                'total_trips': m['total_trips'],
                'distance_km': m['total_distance_km'],
                'driving_hours': m['total_driving_h'],
                'avg_speed_kmh': m['avg_speed_kmh'],
                'max_speed_kmh': m['max_speed_kmh'],
                'violations': m['violations'],
                'first_movement': m['first_movement'],
                'last_movement': m['last_movement'],
                'idle_hours': m['idle_hours'],
                'fuel_cost_estimate': m['fuel_cost_estimate'],
                'under_utilized': m['under_utilized'],
                'over_utilized': m['over_utilized'],
            })
        devices_df = pd.DataFrame(rows)
        devices_df.to_excel(writer, sheet_name='By Vehicle', index=False)

        # Raw data sheets
        if not positions_df.empty:
            # convert fix_time to Lagos for the export (assume positions_df timestamps in London)
            tmp = positions_df.copy()
            tmp['fix_time'] = pd.to_datetime(tmp['fix_time']).dt.tz_convert(LAGOS)
            tmp['server_time'] = pd.to_datetime(tmp['server_time']).dt.tz_convert(LAGOS)
            tmp.to_excel(writer, sheet_name='Positions', index=False)
        if not trips_df.empty:
            tmp = trips_df.copy()
            tmp['start_time'] = pd.to_datetime(tmp['start_time']).dt.tz_convert(LAGOS)
            tmp['end_time'] = pd.to_datetime(tmp['end_time']).dt.tz_convert(LAGOS)
            tmp.to_excel(writer, sheet_name='Trips', index=False)
        if not stops_df.empty:
            # if stops_df is a list converted earlier it will already be fine; otherwise ensure proper tz
            try:
                tmp = stops_df.copy()
                if 'stop_time' in tmp.columns:
                    tmp['stop_time'] = pd.to_datetime(tmp['stop_time']).dt.tz_convert(LAGOS)
                if 'resume_time' in tmp.columns:
                    tmp['resume_time'] = pd.to_datetime(tmp['resume_time']).dt.tz_convert(LAGOS)
                tmp.to_excel(writer, sheet_name='Stops', index=False)
            except Exception:
                # fallback: convert list-of-dicts to df
                tmp = pd.DataFrame(stops_df)
                tmp.to_excel(writer, sheet_name='Stops', index=False)
        if not violations_df.empty:
            tmp = violations_df.copy()
            if 'detected_at' in tmp.columns:
                tmp['detected_at'] = pd.to_datetime(tmp['detected_at']).dt.tz_convert(LAGOS)
            tmp.to_excel(writer, sheet_name='Violations', index=False)

    return excel_path

# ----------------- PDF builder -----------------
def build_pdf(by_device, fleet_summary, out_prefix, start_lagos, end_lagos):
    pdf_path = f"{out_prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        title = f"Fleet Daily Executive Summary\n{start_lagos.isoformat()} to {end_lagos.isoformat()} ({LAGOS.zone})"
        ax.text(0.5, 0.92, title, ha='center', va='center', fontsize=16, weight='bold')

        txt = (
            f"Total fleet distance (km): {fleet_summary['total_fleet_distance_km']:.1f}\n"
            f"Total trips: {fleet_summary['total_trips']}\n"
            f"Total violations: {fleet_summary['total_violations']}\n"
            f"Vehicles reporting: {fleet_summary['vehicles']}\n"
        )
        ax.text(0.02, 0.7, txt, fontsize=12, va='top')

        # small table of top 8 vehicles by distance
        dev_list = sorted(by_device.values(), key=lambda x: x['total_distance_m'], reverse=True)
        topN = dev_list[:8]
        table_rows = []
        for m in topN:
            t = m.get('truck')
            table_rows.append([t.name if t else str(m['device_id']),
                               f"{m['total_distance_km']:.1f}",
                               f"{m['total_trips']}",
                               f"{m['total_driving_h']:.2f}"])
        col_labels = ['Truck','Distance (km)','Trips','Driving hours']
        table = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='left', loc='lower left')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.2)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: By-vehicle charts (distance & trips)
        fig, axs = plt.subplots(2, 1, figsize=(11, 8.5))
        devices = [m.get('truck').name if m.get('truck') else str(m['device_id']) for m in by_device.values()]
        distances = [m['total_distance_km'] for m in by_device.values()]
        trips = [m['total_trips'] for m in by_device.values()]

        # bar distance
        axs[0].barh(devices, distances)
        axs[0].set_xlabel('Distance (km)')
        axs[0].set_title('Distance per vehicle')

        # bar trips
        axs[1].barh(devices, trips)
        axs[1].set_xlabel('Trips')
        axs[1].set_title('Trips per vehicle')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 3: Speed / utilization scatter
        fig, ax = plt.subplots(figsize=(11,8.5))
        x = [m['total_driving_h'] for m in by_device.values()]
        y = [m['total_distance_km'] for m in by_device.values()]
        labels = [m.get('truck').name if m.get('truck') else str(m['device_id']) for m in by_device.values()]
        ax.scatter(x, y)
        for i, txt in enumerate(labels):
            ax.annotate(txt, (x[i], y[i]))
        ax.set_xlabel('Driving hours')
        ax.set_ylabel('Distance (km)')
        ax.set_title('Utilization: driving hours vs distance')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 4: Additional tables - Idle & Violations
        fig, ax = plt.subplots(figsize=(11,8.5))
        ax.axis('off')
        rows = []
        for m in sorted(by_device.values(), key=lambda x: x['idle_seconds'], reverse=True)[:15]:
            t = m.get('truck')
            rows.append([t.name if t else str(m['device_id']),
                         f"{m['idle_hours']:.2f}", f"{m['violations']}", f"{m['fuel_cost_estimate']:.2f}"])
        col_labels = ['Truck', 'Idle hours', 'Violations', 'Fuel cost estimate']
        table = ax.table(cellText=rows, colLabels=col_labels, cellLoc='left', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    return pdf_path

# ----------------- Main -----------------
def main():
    args = parse_args()
    start_lagos = parse_date_lagos(args.start)
    end_lagos = parse_date_lagos(args.end)

    out_prefix = args.out if args.out else f"daily_report_{start_lagos.strftime('%Y-%m-%d')}"
    db_url = args.db
    if not db_url:
        raise SystemExit("DATABASE_URL not set. Set it in your .env or pass --db.")

    engine = create_engine(db_url, future=True)

    print(f"Fetching data between (Lagos local) {start_lagos.isoformat()} and {end_lagos.isoformat()} ...")
    truck_map, positions, trips, stops, violations = fetch_data(engine, start_lagos, end_lagos)

    positions_df = positions_to_df(positions)
    trips_df = trips_to_df(trips)
    stops_df = pd.DataFrame([{
        'id': s.id, 'device_id': s.device_id, 'stop_time': s.stop_time, 'start_zone': s.start_zone,
        'stop_addr': s.stop_addr, 'resume_time': s.resume_time, 'zone': s.zone, 'duration_s': s.duration_s
    } for s in stops]) if stops else pd.DataFrame()
    violations_df = pd.DataFrame([{
        'id': v.id, 'device_id': v.device_id, 'detected_at': v.detected_at, 'lat': v.lat, 'lon': v.lon,
        'address': v.address, 'minutes_in_unapproved': v.minutes_in_unapproved, 'active': v.active
    } for v in violations]) if violations else pd.DataFrame()

    print("Computing metrics...")
    by_device, fleet_summary = compute_metrics(truck_map, positions_df, trips_df, stops_df, violations_df, start_lagos, end_lagos)

    print("Generating Excel...")
    excel_path = build_excel(by_device, fleet_summary, positions_df, trips_df, stops_df, violations_df, out_prefix)
    print(f"Excel saved to {excel_path}")

    print("Generating PDF dashboard...")
    pdf_path = build_pdf(by_device, fleet_summary, out_prefix, start_lagos, end_lagos)
    print(f"PDF saved to {pdf_path}")

    print("Done. Files:")
    print(f" - {excel_path}")
    print(f" - {pdf_path}")

if __name__ == "__main__":
    main()

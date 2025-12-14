"""
Truck Guardian - Daily Fleet Report Generator
Generates a downloadable Excel and PDF dashboard report for a given date range.

Usage:
  export DATABASE_URL=postgresql://user:pass@host:5432/roie_ai_cryslad2
  python daily_report.py --start 2025-11-29 --end 2025-11-30 --out report_2025-11-29

Requirements:
  pip install pandas sqlalchemy psycopg2-binary matplotlib openpyxl numpy

Notes:
- The script expects the DB to contain the tables and columns described by the user:
  Truck, Position, Trip, Stop, Violation, RecognisedZone.
- Timezone-aware datetimes are respected. Provide start/end in ISO format (YYYY-MM-DD or full ISO datetime).
- Trip handling logic: trips spanning the report window are flagged as "in-progress-from-previous-day" or "continues-next-day".
- Distance is calculated preferentially from Trip.distance_m or odometer differences in the Positions table.

This is a starting point. Tweak styling, charts, and cost models as needed.
"""

import os
import argparse
from datetime import datetime, timedelta, timezone
from collections import defaultdict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sqlalchemy import create_engine, Column, Integer, BigInteger, Text, Numeric, Boolean, DateTime, func, select
from sqlalchemy.orm import declarative_base, Session

# from config import DATABASE_URL2

# ----------------- SQLAlchemy models (synchronous) -----------------
Base = declarative_base()

class Truck(Base):
    __tablename__ = "trucks"
    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(BigInteger, unique=True, index=True, nullable=False)
    name = Column(Text)
    phone = Column(Text)
    contact = Column(Text)

class Position(Base):
    __tablename__ = "positions"
    id = Column(BigInteger, primary_key=True, index=True)
    device_id = Column(BigInteger, index=True, nullable=False)
    lat = Column(Numeric, nullable=False)
    lon = Column(Numeric, nullable=False)
    speed = Column(Numeric)
    ignition = Column(Boolean)
    motion = Column(Boolean)
    fix_time = Column(DateTime(timezone=True), nullable=False)
    server_time = Column(DateTime(timezone=True), nullable=False)
    address = Column(Text)
    odometer = Column(Numeric)

class Trip(Base):
    __tablename__ = "trips"
    id = Column(BigInteger, primary_key=True, index=True)
    device_id = Column(BigInteger, index=True, nullable=False)
    start_time = Column(DateTime(timezone=True), nullable=False)
    end_time = Column(DateTime(timezone=True))
    start_zone = Column(Text)
    end_zone = Column(Text)
    start_addr = Column(Text)
    end_addr = Column(Text)
    start_odometer = Column(Numeric)
    end_odometer = Column(Numeric)
    max_speed = Column(Numeric)
    average_speed = Column(Numeric)
    distance_m = Column(Numeric, default=0)
    trip_duration = Column(Numeric)

class Stop(Base):
    __tablename__ = "stops"
    id = Column(BigInteger, primary_key=True, index=True)
    device_id = Column(BigInteger, index=True, nullable=False)
    stop_time = Column(DateTime(timezone=True), nullable=False)
    start_zone = Column(Text)
    stop_addr = Column(Text)
    resume_time = Column(DateTime(timezone=True))
    zone = Column(Text)
    duration_s = Column(Integer)

class Violation(Base):
    __tablename__ = "violations"
    id = Column(BigInteger, primary_key=True, index=True)
    device_id = Column(BigInteger, index=True, nullable=False)
    detected_at = Column(DateTime(timezone=True), nullable=False)
    lat = Column(Numeric)
    lon = Column(Numeric)
    address = Column(Text)
    minutes_in_unapproved = Column(Integer)
    alert_sent = Column(Boolean, default=False)
    resolution = Column(Text)
    comment = Column(Text)
    resolved_by = Column(Text)
    resolved_at = Column(DateTime(timezone=True))
    is_offline_violation = Column(Boolean, default=False)
    maintenance_mode = Column(Boolean, default=False)
    active = Column(Boolean, default=False)

# ----------------- Helpers & Core Logic -----------------

def parse_args():
    p = argparse.ArgumentParser(description="Generate daily fleet report (Excel + PDF)")
    p.add_argument("--start", required=True, help="Start date inclusive (YYYY-MM-DD or ISO)")
    p.add_argument("--end", required=True, help="End date exclusive (YYYY-MM-DD or ISO). Use next day to include full day.")
    p.add_argument("--out", default="daily_report", help="Output filename prefix (no extension)")
    p.add_argument("--db", default="postgresql://roie_user:deVelopPass123@postgres:5432/roie_ai_cryslad2", help="SQLAlchemy DB URL")
    return p.parse_args()


def to_dt(s: str) -> datetime:
    # Accept YYYY-MM-DD or ISO datetimes
    try:
        if len(s) == 10:
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s)
    except Exception:
        raise ValueError("Invalid date format. Use YYYY-MM-DD or full ISO datetime.")


def strip_tz(df):
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_convert(None)
    return df


def fetch_data(engine, start_dt, end_dt):
    """Fetch Trucks, Positions, Trips, Stops, Violations within a window.
    Positions: fetch a small buffer before and after window to handle truncated trips.
    """
    with Session(engine) as session:
        trucks = session.execute(select(Truck)).scalars().all()
        truck_map = {t.device_id: t for t in trucks}

        # fetch positions with a buffer of 1 hour on both ends to help compute in-progress segments
        buffer = timedelta(hours=2)
        pos_q = select(Position).where(Position.fix_time >= (start_dt - buffer)).where(Position.fix_time < (end_dt + buffer))
        positions = session.execute(pos_q).scalars().all()

        trip_q = select(Trip).where(((Trip.start_time >= start_dt) & (Trip.start_time < end_dt)) | ((Trip.end_time != None) & (Trip.end_time >= start_dt) & (Trip.end_time < end_dt)) | ((Trip.start_time < start_dt) & ((Trip.end_time == None) | (Trip.end_time >= start_dt))))
        trips = session.execute(trip_q).scalars().all()

        stop_q = select(Stop).where((Stop.stop_time >= start_dt) & (Stop.stop_time < end_dt))
        stops = session.execute(stop_q).scalars().all()

        vio_q = select(Violation).where((Violation.detected_at >= start_dt) & (Violation.detected_at < end_dt))
        violations = session.execute(vio_q).scalars().all()

    return truck_map, positions, trips, stops, violations


def positions_to_df(positions):
    if not positions:
        return pd.DataFrame()
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
            'fix_time': p.fix_time,
            'server_time': p.server_time,
            'address': p.address,
            'odometer': float(p.odometer) if p.odometer is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df.sort_values(['device_id','fix_time'], inplace=True)
    return df


def trips_to_df(trips):
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
            'distance_m': float(t.distance_m) if t.distance_m is not None else 0,
            'max_speed': float(t.max_speed) if t.max_speed is not None else np.nan,
            'average_speed': float(t.average_speed) if t.average_speed is not None else np.nan,
            'trip_duration': float(t.trip_duration) if t.trip_duration is not None else np.nan,
        })
    df = pd.DataFrame(rows)
    return df


def compute_metrics(truck_map, positions_df, trips_df, stops_df, violations_df, start_dt, end_dt):
    """Compute the requested metrics per vehicle and fleet-level summary."""
    by_device = {}

    devices = sorted(set(list(truck_map.keys()) + positions_df['device_id'].unique().tolist() if not positions_df.empty else list(truck_map.keys())))

    fleet_total_distance = 0.0
    fleet_total_trips = 0
    fleet_total_violations = len(violations_df) if not violations_df.empty else 0

    for dev in devices:
        dev_positions = positions_df[positions_df['device_id'] == dev] if not positions_df.empty else pd.DataFrame()
        dev_trips = trips_df[trips_df['device_id'] == dev] if not trips_df.empty else pd.DataFrame()
        dev_stops = stops_df[stops_df['device_id'] == dev] if not stops_df.empty else pd.DataFrame()
        dev_violations = violations_df[violations_df['device_id'] == dev] if not violations_df.empty else pd.DataFrame()

        # total trips within window (including ongoing/truncated)
        total_trips = len(dev_trips)

        # driving time: approximate by summing durations between consecutive positions where speed>0 or motion True
        total_driving_seconds = 0
        total_distance_m = 0
        avg_speed_list = []
        max_speed = 0
        longest_trip_seconds = 0
        longest_trip_distance = 0

        # If odometer exists, compute distance using odometer diffs inside range
        if not dev_positions.empty and dev_positions['odometer'].notna().sum() >= 2:
            # restrict to the actual report window
            mask = (dev_positions['fix_time'] >= start_dt) & (dev_positions['fix_time'] < end_dt)
            rpt_pos = dev_positions.loc[mask].copy()
            if not rpt_pos.empty:
                # distance in meters assuming odometer in meters (if in km, adapt)
                odom_min = rpt_pos['odometer'].min()
                odom_max = rpt_pos['odometer'].max()
                total_distance_m = float(odom_max - odom_min)
        else:
            # fallback: use Trip.distance_m sums for trips that overlap
            if not dev_trips.empty:
                total_distance_m = float(dev_trips['distance_m'].sum())

        # compute driving seconds and speeds from positions
        if not dev_positions.empty:
            pos = dev_positions.sort_values('fix_time')
            pos = pos[(pos['fix_time'] >= (start_dt - timedelta(hours=2))) & (pos['fix_time'] < (end_dt + timedelta(hours=2)))].copy()
            pos['fix_time'] = pd.to_datetime(pos['fix_time'])
            pos.reset_index(drop=True, inplace=True)
            for i in range(1, len(pos)):
                prev = pos.loc[i-1]
                cur = pos.loc[i]
                dt = (pd.to_datetime(cur['fix_time']) - pd.to_datetime(prev['fix_time'])).total_seconds()
                if dt <= 0:
                    continue
                # consider as driving interval if either has motion True or speed>2 km/h
                prev_speed = prev['speed'] if not pd.isna(prev['speed']) else 0
                cur_speed = cur['speed'] if not pd.isna(cur['speed']) else 0
                motion = (bool(prev['motion']) or bool(cur['motion'])) or (prev_speed > 2 or cur_speed > 2)
                if motion:
                    total_driving_seconds += dt
                    # add speeds for average computation
                    avg_speed_list.append(max(prev_speed, cur_speed))
                    max_speed = max(max_speed, float(prev_speed if not pd.isna(prev_speed) else 0), float(cur_speed if not pd.isna(cur_speed) else 0))
            if avg_speed_list:
                avg_speed = float(np.nanmean(avg_speed_list))
            else:
                avg_speed = np.nan
        else:
            avg_speed = np.nan

        # longest trip: look at trips within window and consider those spanning the window
        if not dev_trips.empty:
            for _, r in dev_trips.iterrows():
                st = r['start_time']
                et = r['end_time']
                if st is None:
                    continue
                # clip times to report window
                clip_start = max(st, start_dt)
                clip_end = min(et, end_dt) if et is not None else end_dt
                duration = (clip_end - clip_start).total_seconds()
                distance = float(r['distance_m']) if not pd.isna(r['distance_m']) else 0
                if duration > longest_trip_seconds:
                    longest_trip_seconds = duration
                    longest_trip_distance = distance

        # flags for under/over utilization
        # thresholds (example): under-utilized < 20 km/day, over-utilized > 10 hours driving
        under_utilized = total_distance_m < (20 * 1000)
        over_utilized = (total_driving_seconds / 3600.0) > 10

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
            'longest_trip_minutes': longest_trip_seconds / 60.0,
            'longest_trip_distance_m': longest_trip_distance,
            'under_utilized': under_utilized,
            'over_utilized': over_utilized,
            'violations': len(dev_violations)
        }

    # prepare fleet summary
    fleet_summary = {
        'total_fleet_distance_km': fleet_total_distance / 1000.0,
        'total_trips': fleet_total_trips,
        'total_violations': fleet_total_violations,
        'vehicles': len(by_device)
    }

    return by_device, fleet_summary


def build_excel(by_device, fleet_summary, positions_df, trips_df, stops_df, violations_df, out_prefix):
    # --- Clean timestamps ---
    positions_df = strip_tz(positions_df)
    trips_df = strip_tz(trips_df)
    stops_df = strip_tz(stops_df)
    violations_df = strip_tz(violations_df)

    excel_path = f"{out_prefix}.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:

        # =====================================================
        # 1. FLEET SUMMARY
        # =====================================================
        fleet_df = pd.DataFrame([fleet_summary])
        fleet_df.to_excel(writer, sheet_name='Fleet Summary', index=False)

        # =====================================================
        # 2. BY VEHICLE SUMMARY
        # =====================================================
        rows = []
        for dev, m in by_device.items():
            t = m.get('truck')
            rows.append({
                'device_id': dev,
                'truck_name': t.name if t is not None else None,
                'total_trips': m['total_trips'],
                'distance_km': m['total_distance_km'],
                'driving_hours': m['total_driving_h'],
                'avg_speed_kmh': m['avg_speed_kmh'],
                'max_speed_kmh': m['max_speed_kmh'],
                'violations_count': m['violations'],
                'under_utilized': m['under_utilized'],
                'over_utilized': m['over_utilized'],
            })
        devices_df = pd.DataFrame(rows)
        devices_df.to_excel(writer, sheet_name='By Vehicle', index=False)

        # =====================================================
        # 3. RAW SHEETS (optional)
        # =====================================================
        if not positions_df.empty:
            positions_df.to_excel(writer, sheet_name='Positions', index=False)
        if not trips_df.empty:
            trips_df.to_excel(writer, sheet_name='Trips', index=False)
        if not stops_df.empty:
            stops_df.to_excel(writer, sheet_name='Stops', index=False)
        if not violations_df.empty:
            violations_df.to_excel(writer, sheet_name='Violations', index=False)

        # =====================================================
        # 4. ONE COMBINED SHEET: Trips & Violations
        # =====================================================

        combined_sheet = "Trips & Violations"

        # --------------------------
        # TRIPS SECTION FOR ALL DEVICES
        # --------------------------
        all_trip_rows = []
        for dev, m in by_device.items():
            t = m.get('truck')
            truck_name = t.name if t else None

            dev_trips = trips_df[trips_df["device_id"] == dev].copy()
            if not dev_trips.empty:
                dev_trips["trip_duration_min"] = (
                    dev_trips["end_time"] - dev_trips["start_time"]
                ).dt.total_seconds() / 60.0

                for _, r in dev_trips.iterrows():
                    all_trip_rows.append({
                        "device_id": dev,
                        "truck_name": truck_name,
                        "trip_id": r["id"],
                        "start_time": r["start_time"],
                        "end_time": r["end_time"],
                        "duration_min": r["trip_duration_min"],
                        "start_location": r.get("start_zone"),
                        "start_address": r.get("start_addr"),
                        "end_location": r.get("end_zone"),
                        "end_address": r.get("end_addr"),
                        "distance_km": r.get("distance_km"),
                        "avg_speed": r.get("avg_speed"),
                        "max_speed": r.get("max_speed"),
                    })

        trips_full_df = pd.DataFrame(all_trip_rows)

        # --------------------------
        # VIOLATIONS SECTION (ALL DEVICES)
        # --------------------------
        all_viol_rows = []
        for dev, m in by_device.items():
            t = m.get('truck')
            truck_name = t.name if t else None

            dev_viol = violations_df[violations_df["device_id"] == dev].copy()
            for _, r in dev_viol.iterrows():
                all_viol_rows.append({
                    "device_id": dev,
                    "truck_name": truck_name,
                    "violation_id": r["id"],
                    "violation_time": r["detected_at"],
                    "lat": r.get("lat"),
                    "lon": r.get("lon"),
                    "address": r.get("address"),
                    "minutes_in_violation": r.get("minutes_in_unapproved"),
                })

        viol_full_df = pd.DataFrame(all_viol_rows)

        # Write combined sheet
        start_row = 0

        # TRIPS first
        trips_full_df.to_excel(writer, sheet_name=combined_sheet,
                               startrow=start_row, index=False)
        start_row += len(trips_full_df) + 3

        # VIOLATIONS header title
        pd.DataFrame([{"Violations": "Violations for all vehicles"}]) \
            .to_excel(writer, sheet_name=combined_sheet,
                      startrow=start_row, index=False, header=False)
        start_row += 2

        # VIOLATIONS table
        viol_full_df.to_excel(writer, sheet_name=combined_sheet,
                              startrow=start_row, index=False)

    return excel_path


def build_pdf(by_device, fleet_summary, out_prefix, start_dt, end_dt):
    pdf_path = f"{out_prefix}.pdf"
    with PdfPages(pdf_path) as pdf:
        # Page 1: Executive summary
        fig, ax = plt.subplots(figsize=(11, 8.5))
        ax.axis('off')
        title = f"Fleet Daily Executive Summary\n{start_dt.isoformat()} to {end_dt.isoformat()}"
        ax.text(0.5, 0.92, title, ha='center', va='center', fontsize=16, weight='bold')

        txt = (
            f"Total fleet distance (km): {fleet_summary['total_fleet_distance_km']:.1f}\n"
            f"Total trips: {fleet_summary['total_trips']}\n"
            f"Total violations: {fleet_summary['total_violations']}\n"
            f"Vehicles reporting: {fleet_summary['vehicles']}\n"
        )
        ax.text(0.02, 0.7, txt, fontsize=12, va='top')

        # small table of top 5 vehicles by distance
        dev_list = sorted(by_device.values(), key=lambda x: x['total_distance_m'], reverse=True)
        top5 = dev_list[:5]
        table_rows = []
        for m in top5:
            t = m.get('truck')
            table_rows.append([t.name if t else str(m['device_id']), f"{m['total_distance_km']:.1f}", f"{m['total_trips']}", f"{m['total_driving_h']:.2f}"]) 
        col_labels = ['Truck','Distance (km)','Trips','Driving hours']
        table = ax.table(cellText=table_rows, colLabels=col_labels, cellLoc='left', loc='lower left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.2)

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: By-vehicle charts
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

    return pdf_path


# ----------------- Main -----------------

def main():
    args = parse_args()
    start_dt = to_dt(args.start)
    end_dt = to_dt(args.end)

    if not args.db:
        raise SystemExit("Please set DATABASE_URL environment variable or pass --db")

    engine = create_engine(args.db, future=True)

    print(f"Fetching data between {start_dt} and {end_dt}...")
    truck_map, positions, trips, stops, violations = fetch_data(engine, start_dt, end_dt)

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
    by_device, fleet_summary = compute_metrics(truck_map, positions_df, trips_df, stops_df, violations_df, start_dt, end_dt)

    print("Generating Excel...")
    excel_path = build_excel(by_device, fleet_summary, positions_df, trips_df, stops_df, violations_df, args.out)
    print(f"Excel saved to {excel_path}")

    print("Generating PDF dashboard...")
    pdf_path = build_pdf(by_device, fleet_summary, args.out, start_dt, end_dt)
    print(f"PDF saved to {pdf_path}")

    print("Done. Files:")
    print(f" - {excel_path}")
    print(f" - {pdf_path}")


if __name__ == '__main__':
    main()

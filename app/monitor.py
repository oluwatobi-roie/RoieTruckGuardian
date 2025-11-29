# app/monitor.py
import datetime as dt
from typing import List, Optional

from sqlalchemy import select, update
from sqlalchemy.orm import selectinload

from database import AsyncSessionLocal
from models import Position, Violation, DeviceStatus, Trip, Truck
from geozones import zone_at
from alerts import send_alert

from logging_config import get_logger

# from utils.variable import *


logger = get_logger("monitor", "monitor.log")

UTC = dt.timezone.utc

# ----- thresholds (tweakable) -----
TRIP_START_SPEED = 10.0    # km/h — conservative to avoid false starts
TRIP_START_COUNT = 10      # consecutive points to confirm start
TRIP_END_SPEED = 1.0       # km/h — to detect stop
TRIP_END_COUNT = 10        # consecutive points to confirm end

STATIONARY_SPEED = 1.0     # km/h threshold for "stationary" for parking violation
STATIONARY_COUNT = 10      # number of recent points that must be <= STATIONARY_SPEED
MIN_SUSTAINED_STOP_SECONDS = 300  # 5 minutes sustained stop to consider "parked"

PARKING_VIOLATION_THRESHOLD_MINUTES = 60  # minutes to trigger alert
RETRIGGER_COOLDOWN_MINUTES = 10  # don't create a new violation within this minutes of deactivation


# =====================================================================
# Helper: trip start/end/stationary detection (positions list is newest-first)
# =====================================================================
async def _is_trip_start(positions: List[Position]) -> bool:
    """
    Conservative trip-start detection: require TRIP_START_COUNT recent points > TRIP_START_SPEED.
    positions: newest-first
    """
    count = 0
    for p in positions[:TRIP_START_COUNT]:
        try:
            sp = float(p.speed or 0)
        except Exception:
            sp = 0.0
        if sp > TRIP_START_SPEED:
            count += 1
    return count >= TRIP_START_COUNT


async def _is_trip_end(positions: List[Position]) -> bool:
    """
    Trip end: recent points indicate vehicle stopped (<= TRIP_END_SPEED).
    positions: newest-first
    """
    count = 0
    for p in positions[:TRIP_END_COUNT]:
        try:
            sp = float(p.speed or 0)
        except Exception:
            sp = 0.0
        if sp <= TRIP_END_SPEED:
            count += 1
    return count >= TRIP_END_COUNT


async def _is_stationary(positions: List[Position]) -> bool:
    """
    Quick check: are the most recent STATIONARY_COUNT points at or below STATIONARY_SPEED?
    positions: newest-first
    """
    count = 0
    for p in positions[:STATIONARY_COUNT]:
        try:
            sp = float(p.speed or 0)
        except Exception:
            sp = 0.0
        if sp <= STATIONARY_SPEED:
            count += 1
    return count >= STATIONARY_COUNT


async def get_stationary_start(positions: List[Position], min_stop_seconds: int = MIN_SUSTAINED_STOP_SECONDS) -> Optional[Position]:
    """
    Determine sustained stationary start (parked start) in the provided positions window.
    - positions is newest-first.
    - Returns the earliest Position object representing the start of a sustained stop (oldest point of the run),
      or None if no sustained stationary run >= min_stop_seconds is found.
    """
    if not positions:
        return None

    # convert to oldest -> newest for run detection
    pts = list(reversed(positions))
    stationary_flags = []
    for p in pts:
        try:
            sp = float(p.speed or 0)
        except Exception:
            sp = 0.0
        stationary_flags.append(sp <= STATIONARY_SPEED)

    run_start_idx = None
    for i, flag in enumerate(stationary_flags):
        if flag:
            if run_start_idx is None:
                run_start_idx = i
        else:
            if run_start_idx is not None:
                start_pos = pts[run_start_idx]
                end_pos = pts[i - 1]
                span = (end_pos.server_time - start_pos.server_time).total_seconds()
                if span >= min_stop_seconds:
                    return start_pos
                run_start_idx = None

    # If run continues to end
    if run_start_idx is not None:
        start_pos = pts[run_start_idx]
        end_pos = pts[-1]
        span = (end_pos.server_time - start_pos.server_time).total_seconds()
        if span >= min_stop_seconds:
            return start_pos

    return None


# =====================================================================
# Trip create / close helpers
# =====================================================================
async def _start_trip(db, device_id: int, start_pos: Position, start_zone: Optional[str] = None) -> Trip:
    trip = Trip(
        device_id=device_id,
        start_time=start_pos.server_time,
        start_zone=start_zone,
        start_addr=start_pos.address,
        start_odometer=start_pos.odometer,
    )
    db.add(trip)
    await db.commit()
    await db.refresh(trip)
    logger.info(f"[trip] Started trip id={trip.id} device={device_id} at {trip.start_time}")
    return trip


async def _end_trip(db, device_id: int, end_pos: Position) -> Optional[Trip]:
    q = await db.execute(
        select(Trip)
        .where(Trip.device_id == device_id, Trip.end_time.is_(None))
        .order_by(Trip.start_time.desc())
        .limit(1)
    )
    trip = q.scalar_one_or_none()
    if not trip:
        return None

    trip.end_time = end_pos.server_time
    trip.end_addr = end_pos.address
    # Optionally compute distance_m here by aggregating positions between trip.start_time and trip.end_time
    await db.commit()
    logger.info(f"[trip] Ended trip id={trip.id} device={device_id} at {trip.end_time}")
    return trip


# =====================================================================
# Utility: create or re-send violation alerts safely (active/inactive model)
# =====================================================================
async def create_or_send_alert(db, device_id, minutes, latest, *, is_offline):
    """
    - Only consider active==True violations as the running violation to update or alert.
    - If an active violation exists: update minutes/location and send alert only if alert_sent is False.
    - If no active violation exists:
        - If a recent inactive (but unresolved) violation exists and is within RETRIGGER_COOLDOWN_MINUTES -> skip creating new violation
        - Otherwise create a NEW violation (active=True) and send alert once.
    """
    # --------------- Load Device Name -----------------------------
    truck_row = await db.execute(select(Truck).where(Truck.device_id == device_id))
    truck = truck_row.scalar_one_or_none()

    if truck:
        truck_name = truck.name
        logger.info(f"[create_or_send_alert] device={device_id} / {truck_name} minutes={minutes} is_offline={is_offline}")
    else:
        truck_name = None
        logger.warning(f"[create_or_send_alert] No truck found for device_id={device_id}")

    try:
        # 0) COOLDOWN: check for recently deactivated (inactive but unresolved) violation
        recent_q = await db.execute(
            select(Violation)
            .where(
                Violation.device_id == device_id,
                Violation.active == False,
                Violation.resolved_at.is_(None),
                Violation.is_offline_violation == is_offline
            )
            .order_by(Violation.detected_at.desc())
            .limit(1)
        )
        recent = recent_q.scalar_one_or_none()
        if recent:
            delta = dt.datetime.now(UTC) - recent.detected_at
            if delta.total_seconds() < RETRIGGER_COOLDOWN_MINUTES * 60:
                logger.info(f"[create_or_send_alert] Recent inactive violation id={recent.id} detected {delta.total_seconds()/60:.1f}m ago — skipping new creation (cooldown)")
                # Update its minutes/location for monitoring visibility (do not reactivate)
                recent.minutes_in_unapproved = minutes
                recent.lat = latest.lat
                recent.lon = latest.lon
                recent.address = latest.address
                await db.commit()
                return recent

        # 1) Fetch active violation of same type (if any). Lock row to avoid races.
        existing_q = await db.execute(
            select(Violation)
            .where(
                Violation.device_id == device_id,
                Violation.active == True,
                Violation.is_offline_violation == is_offline
            )
            .order_by(Violation.detected_at.desc())
            .limit(1)
            .with_for_update()
        )

        existing = existing_q.scalar_one_or_none()

        if existing:
            logger.info(f"[create_or_send_alert] Active violation exists id={existing.id} alert_sent={existing.alert_sent}")
            # Send alert only once
            if not existing.alert_sent:
                logger.info(f"[create_or_send_alert] Sending alert for existing violation id={existing.id}")
                try:
                    await send_alert(existing, truck.name)
                    existing.alert_sent = True
                    # update minutes_in_unapproved and location
                    existing.minutes_in_unapproved = minutes
                    existing.lat = latest.lat
                    existing.lon = latest.lon
                    existing.address = latest.address
                    await db.commit()
                except Exception:
                    logger.exception(f"[create_or_send_alert] Error sending alert for existing violation id={existing.id}")
            else:
                # Update minutes and location for monitoring purposes but do NOT resend alert
                existing.minutes_in_unapproved = minutes
                existing.lat = latest.lat
                existing.lon = latest.lon
                existing.address = latest.address
                await db.commit()
            return existing

        # 2) No active violation: create a new one and send alert once ONLY when minutes threshold reached.
        if minutes < PARKING_VIOLATION_THRESHOLD_MINUTES and not is_offline:
            logger.info(f"[create_or_send_alert] Parked minutes={minutes} < {PARKING_VIOLATION_THRESHOLD_MINUTES}; not creating violation yet")
            return None

        detected_at = dt.datetime.now(UTC)

        viol = Violation(
            device_id=device_id,
            detected_at=detected_at,
            lat=latest.lat,
            lon=latest.lon,
            address=latest.address,
            minutes_in_unapproved=minutes,
            is_offline_violation=is_offline,
            alert_sent=False,
            active=True,
        )

        db.add(viol)
        await db.commit()
        await db.refresh(viol)

        logger.info(f"[create_or_send_alert] New violation id={viol.id}, sending alert")
        try:
            # send_alert expects the violation object
            await send_alert(viol)
            viol.alert_sent = True
            await db.commit()
        except Exception:
            logger.exception(f"[create_or_send_alert] Error sending alert for new violation id={viol.id}")
            # Keep the violation record so we can retry sending later.

        return viol

    except Exception as e:
        tn = truck_name or device_id
        logger.exception(f"[create_or_send_alert] ERROR for device {device_id} / {tn}: {e}")
        raise


# =====================================================================
# Resolve offline violations automatically when device comes back online
# (Parking violations are NOT resolved automatically — they are merely set inactive on trip start)
# =====================================================================
async def _resolve_offline_violations_on_online(db, device_id, resolved_at):
    q = await db.execute(
        select(Violation)
        .where(
            Violation.device_id == device_id,
            Violation.is_offline_violation == True,
            Violation.resolved_at.is_(None)
        )
    )
    for v in q.scalars().all():
        v.resolved_at = resolved_at
        v.resolution = "device_back_online"
        v.resolved_by = "system_online"
    await db.commit()


# =====================================================================
# Mark active violations inactive (when a trip starts). Do NOT set resolved_at.
# This preserves the violation row for admin to resolve later.
# =====================================================================
async def _deactivate_active_violations(db, device_id):
    await db.execute(
        update(Violation)
        .where(
            Violation.device_id == device_id,
            Violation.active == True
        )
        .values(active=False)
    )
    await db.commit()


# =====================================================================
# MAIN EVALUATOR
# =====================================================================
async def evaluator(device_id: int) -> None:
    logger.info(f"[evaluator] Starting evaluation for device {device_id}")

    try:
        async with AsyncSessionLocal() as db:
            # ---------------------- Device ID & Truck ------------------------
            truck_row = await db.execute(
                select(Truck).where(Truck.device_id == device_id)
            )
            truck = truck_row.scalar_one_or_none()

            truck_name = truck.name if truck else "UNKNOWN"
            logger.info(f"[evaluator] TruckName={truck_name} device_id={device_id}")
            
            if truck:
                logger.info(f"[evaluator] Truck: {truck_name}")
            else:
                logger.warning(f"[evaluator] No truck found for device_id={device_id}")

            # ------------------- TIME WINDOW --------------------
            now = dt.datetime.now(UTC)
            since = now - dt.timedelta(minutes=70)

            pos_rows = await db.execute(
                select(Position)
                .where(
                    Position.device_id == device_id,
                    Position.server_time >= since
                )
                .order_by(Position.server_time.desc())
            )

            positions = pos_rows.scalars().all()
            logger.info(f"[evaluator] Loaded {len(positions)} positions for device {device_id} / {truck_name}")

            if len(positions) < 2:
                logger.info("[evaluator] Not enough data → skipping")
                return

            latest = positions[0]

            # ------------------- ZONE LOOKUP --------------------
            latest_zone = await zone_at(db, latest.lat, latest.lon)
            logger.info(
                f"[evaluator] Zone lookup device {device_id} / {truck_name} : "
                f"{'IN zone ' + latest_zone['name'] if latest_zone else 'OUTSIDE zone'}"
            )
            # ------------------- TRIP START DETECTION --------------------
            if await _is_trip_start(positions):
                # Check if there's an already-open trip for this device
                q = await db.execute(
                    select(Trip)
                    .where(Trip.device_id == device_id, Trip.end_time.is_(None))
                    .limit(1)
                )
                open_trip = q.scalar_one_or_none()

                if not open_trip:
                    # Start trip
                    latest_zone_name = await zone_at(db, latest.lat, latest.lon)
                    await _start_trip(db, device_id, latest, start_zone=(latest_zone_name["name"] if latest_zone_name else None))

                    # Deactivate active violations ONLY IF movement is sustained (not GPS jitter)
                    sustained_movement = not await _is_stationary(positions)
                    if sustained_movement:
                        await _deactivate_active_violations(db, device_id)
                        logger.info(f"[evaluator] Deactivated active violations for device {device_id} / {truck_name} due to sustained movement")
                    else:
                        logger.info(f"[evaluator] Detected trip start but vehicle not showing sustained movement - not deactivating violations for device {device_id} / {truck_name}")

            # ------------------- TRIP END DETECTION --------------------
            if await _is_trip_end(positions):
                ended = await _end_trip(db, device_id, latest)
                if ended:
                    logger.info(f"[evaluator] Trip ended for device {device_id} / {truck_name} , trip id={ended.id}")

            # =====================================================
            # PARKED VIOLATION (out of zone for >= threshold) – only when latest is outside zone
            # =====================================================
            if not latest_zone:
                logger.info(f"[evaluator] Scanning backward to detect parked start for {device_id} / {truck_name}")

                # If vehicle is not stationary now, don't create parking violation
                if not await _is_stationary(positions):
                    logger.info(f"[evaluator] Device {device_id} / {truck_name} not stationary → skipping parked violation check")
                else:
                    # Prefer trip-end anchor if a trip ended in the window
                    parked_start_pos: Optional[Position] = None

                    # Find most recent trip that ended and whose end_time is within our window
                    q_trip = await db.execute(
                        select(Trip)
                        .where(Trip.device_id == device_id, Trip.end_time.isnot(None))
                        .order_by(Trip.end_time.desc())
                        .limit(1)
                    )
                    recent_trip = q_trip.scalar_one_or_none()
                    if recent_trip and recent_trip.end_time:
                        # if trip ended within our window (>= since), use its end_time as parked start anchor
                        if recent_trip.end_time >= positions[-1].server_time and recent_trip.end_time <= latest.server_time:
                            # find the earliest position at/after the trip end
                            for p in reversed(positions):
                                if p.server_time >= recent_trip.end_time:
                                    parked_start_pos = p
                                    break
                            if parked_start_pos is None:
                                parked_start_pos = positions[-1]

                    # Otherwise fallback to sustained-stationary detection
                    if parked_start_pos is None:
                        parked_start_pos = await get_stationary_start(positions, min_stop_seconds=MIN_SUSTAINED_STOP_SECONDS)

                    if parked_start_pos is None:
                        logger.info(f"[evaluator] No sustained stationary period found for device {device_id} / {truck_name} → skipping parked violation check")
                    else:
                        minutes = int((latest.server_time - parked_start_pos.server_time).total_seconds() / 60)
                        logger.info(f"[evaluator] Device {device_id} / {truck_name} parked_start={parked_start_pos.server_time} | parked_minutes={minutes}")

                        # Only create violation when threshold reached and no active violation exists
                        await create_or_send_alert(db=db, device_id=device_id, minutes=minutes, latest=latest, is_offline=False)

            # =====================================================
            # OFFLINE VIOLATION
            # =====================================================
            status = await db.get(DeviceStatus, device_id)

            if status:
                now_utc = dt.datetime.now(UTC)
                # If device is offline for >= 60 minutes -> create offline violation
                if status.last_seen is None or (now_utc - status.last_seen).total_seconds() >= 60 * 60:
                    minutes_off = int((now_utc - (status.last_seen or now_utc)).total_seconds() / 60)
                    logger.info(f"[evaluator] Device {device_id} / {truck_name} last_seen={status.last_seen} offline for {minutes_off} minutes")
                    if minutes_off >= 60:
                        await create_or_send_alert(db=db, device_id=device_id, minutes=minutes_off, latest=latest, is_offline=True)
                else:
                    # Device is now online → resolve offline violations automatically
                    await _resolve_offline_violations_on_online(db, device_id, now_utc)

        logger.info(f"[evaluator] Completed evaluation for device {device_id} / {truck_name}")

    except Exception as e:
        logger.exception(f"[evaluator] ERROR evaluating device {device_id} / {truck_name} : {e}")
        raise






















# =====================================================================
# Yesterday Trip Report generator (call from cron / scheduler)
# =====================================================================
async def generate_yesterday_trip_report(report_date: Optional[dt.date] = None):
    """
    Returns list of dicts describing trips for the specified date (defaults to yesterday UTC).
    Each dict includes start/end times, duration (seconds), max/avg speed, start/end addresses (or zone names).
    """
    if report_date is None:
        report_date = (dt.datetime.utcnow() - dt.timedelta(days=1)).date()

    start_dt = dt.datetime.combine(report_date, dt.time.min).replace(tzinfo=UTC)
    end_dt = dt.datetime.combine(report_date, dt.time.max).replace(tzinfo=UTC)

    report = []

    async with AsyncSessionLocal() as db:
        q = await db.execute(
            select(Trip)
            .where(
                Trip.start_time >= start_dt,
                Trip.start_time <= end_dt
            )
            .order_by(Trip.start_time)
        )
        trips = q.scalars().all()

        for trip in trips:
            pos_q = await db.execute(
                select(Position)
                .where(
                    Position.device_id == trip.device_id,
                    Position.server_time >= trip.start_time,
                    Position.server_time <= (trip.end_time if trip.end_time else end_dt)
                )
                .order_by(Position.server_time)
            )
            pos_list = pos_q.scalars().all()
            if not pos_list:
                continue

            speeds = [float(p.speed or 0) for p in pos_list]
            max_speed = max(speeds) if speeds else 0
            avg_speed = (sum(speeds) / len(speeds)) if speeds else 0
            duration_s = None
            if trip.end_time:
                duration_s = (trip.end_time - trip.start_time).total_seconds()

            # zone lookup for start/end
            start_zone_obj = await zone_at(db, pos_list[0].lat, pos_list[0].lon)
            end_zone_obj = None
            if trip.end_time:
                end_zone_obj = await zone_at(db, pos_list[-1].lat, pos_list[-1].lon)

            report.append({
                "trip_id": trip.id,
                "device_id": trip.device_id,
                "start_time": trip.start_time,
                "end_time": trip.end_time,
                "duration_s": duration_s,
                "max_speed": max_speed,
                "avg_speed": avg_speed,
                "start_address": (start_zone_obj.name if start_zone_obj else (trip.start_addr or pos_list[0].address)),
                "end_address": (end_zone_obj.name if end_zone_obj else (trip.end_addr or pos_list[-1].address)),
            })

    return report

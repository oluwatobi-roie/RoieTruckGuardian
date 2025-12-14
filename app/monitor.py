# app/monitor.py
import datetime as dt
from typing import List, Optional

from sqlalchemy import select, update, func, and_
from sqlalchemy.orm import selectinload

from database import AsyncSessionLocal
from models import Position, Violation, DeviceStatus, Trip, Truck, Stop
from geozones import zone_at
from alerts import send_alert

from logging_config import get_logger

# from utils.variable import *


logger = get_logger("monitor", "monitor.log")


UTC = dt.timezone.utc

# ----- thresholds (tweakable) -----
# TRIP_START_SPEED = 5.0    # km/h — conservative to avoid false starts
# TRIP_START_COUNT = 2      # consecutive points to confirm start
# TRIP_END_SPEED = 2.0       # km/h — to detect stop
# TRIP_END_COUNT = 3        # consecutive points to confirm end

# STATIONARY_SPEED = 1.0     # km/h threshold for "stationary" for parking violation
# STATIONARY_COUNT = 10      # number of recent points that must be <= STATIONARY_SPEED
# MIN_SUSTAINED_STOP_SECONDS = 300  # 5 minutes sustained stop to consider "parked"

PARKING_VIOLATION_THRESHOLD_MINUTES = 60  # minutes to trigger alert
RETRIGGER_COOLDOWN_MINUTES = 10  # don't create a new violation within this minutes of deactivation


# SPEED_THRESHOLD = 1.0  # km/h
# CONSECUTIVE_REQUIRED = 2  # number of consecutive readings needed to confirm movement


TRIP_START_SPEED = 5.0
TRIP_START_COUNT = 10     # ≈ 30 seconds of movement

TRIP_END_SPEED = 2.0
TRIP_END_COUNT = 10      # ≈ 1 minute of low-speed

STATIONARY_SPEED = 1.0
STATIONARY_COUNT = 10

MIN_SUSTAINED_STOP_SECONDS = 300

SPEED_THRESHOLD = 1.0
CONSECUTIVE_REQUIRED = 4


# =====================================================================
# Helper: Find True start time of trips.
# =====================================================================
async def _find_true_start_time(db, device_id: int, detected_start: Position):
    """
    Detect exact start time by scanning for the moment speed becomes >1 km/h
    for a required number of consecutive readings.
    Ignores GPS noise.
    """
    lookback_start = detected_start.server_time - dt.timedelta(minutes=10)

    q = (
        select(Position)
        .where(
            and_(
                Position.device_id == device_id,
                Position.server_time >= lookback_start,
                Position.server_time <= detected_start.server_time
            )
        )
        .order_by(Position.server_time.asc())
    )
    res = await db.execute(q)
    rows = res.scalars().all()

    if not rows:
        return detected_start  # fallback

    consecutive = 0
    candidate_start = None

    for pos in rows:
        speed = float(pos.speed or 0)

        if speed >= SPEED_THRESHOLD:
            # start counting consecutive movement points
            consecutive += 1

            if consecutive == 1:
                candidate_start = pos  # potential movement start

            if consecutive >= CONSECUTIVE_REQUIRED:
                return candidate_start  # confirmed true start

        else:
            # reset if we see <1 km/h again
            consecutive = 0
            candidate_start = None

    # If we never got enough consecutive points → fallback
    return detected_start

# =====================================================================
# Helper: Find True End time of trips.
# =====================================================================
async def _find_true_end_time(db, device_id: int, detected_end: Position):
    """
    Scan forward and backward around detected_end to confirm the real 
    end of movement using noise filtering.
    """
    from datetime import timedelta
    from sqlalchemy import select, and_

    lookback_start = detected_end.server_time - timedelta(minutes=10)

    q = (
        select(Position)
        .where(
            and_(
                Position.device_id == device_id,
                Position.server_time >= lookback_start,
                Position.server_time <= detected_end.server_time
            )
        )
        .order_by(Position.server_time.asc())
    )

    res = await db.execute(q)
    rows = res.scalars().all()
    if not rows:
        return detected_end

    consec = 0
    candidate = None

    # Trip ends when speed remains BELOW threshold N consecutive times
    for pos in reversed(rows):  # reverse scan
        speed = float(pos.speed or 0)

        if speed < SPEED_THRESHOLD:
            consec += 1
            if consec == 1:
                candidate = pos
            if consec >= CONSECUTIVE_REQUIRED:
                return candidate
        else:
            break  # we hit motion again → stop scanning

    return detected_end


# =====================================================================
# Helper: trip start/end/stationary detection (positions list is newest-first)
# =====================================================================
async def _is_trip_start(positions: List[Position]) -> bool:
    """
    Trip start detection using TRIP_START_COUNT oldest-to-newest points.
    Requires consecutive speeds > TRIP_START_SPEED.
    """
    if len(positions) < TRIP_START_COUNT:
        return False

    # convert newest->oldest to oldest->newest
    ordered = list(reversed(positions))

    consecutive = 0
    for p in ordered[-TRIP_START_COUNT:]:  # last N in time order
        sp = float(p.speed or 0)
        if sp > TRIP_START_SPEED:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= TRIP_START_COUNT:
            return True

    return False

async def _is_trip_end(positions: List[Position]) -> bool:
    """
    Trip end detection using TRIP_END_COUNT oldest-to-newest points.
    Requires consecutive speeds <= TRIP_END_SPEED.
    """
    if len(positions) < TRIP_END_COUNT:
        return False

    ordered = list(reversed(positions))

    consecutive = 0
    for p in ordered[-TRIP_END_COUNT:]:
        sp = float(p.speed or 0)
        if sp <= TRIP_END_SPEED:
            consecutive += 1
        else:
            consecutive = 0

        if consecutive >= TRIP_END_COUNT:
            return True

    return False



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


# --- Get Sustained Movement ---

async def get_sustained_movement_start(positions: List[Position], min_move_seconds: int = 30) -> Optional[Position]:
    """
    Find the earliest Position that begins a sustained movement run >= min_move_seconds.
    positions: newest-first (same convention as your other helpers)
    Returns the Position at the beginning of the sustained movement run (oldest in that run),
    or None if no sustained movement run found.
    """
    if not positions:
        return None

    pts = list(reversed(positions))  # oldest -> newest
    move_flags = []
    for p in pts:
        try:
            sp = float(p.speed or 0.0)
        except Exception:
            sp = 0.0
        move_flags.append(sp > TRIP_START_SPEED)  # or use STATIONARY_SPEED depending on how strict you want

    run_start_idx = None
    for i, flag in enumerate(move_flags):
        if flag:
            if run_start_idx is None:
                run_start_idx = i
        else:
            if run_start_idx is not None:
                start_pos = pts[run_start_idx]
                end_pos = pts[i - 1]
                span_s = (end_pos.server_time - start_pos.server_time).total_seconds()
                if span_s >= min_move_seconds:
                    return start_pos
                run_start_idx = None

    # run continues to end
    if run_start_idx is not None:
        start_pos = pts[run_start_idx]
        end_pos = pts[-1]
        span_s = (end_pos.server_time - start_pos.server_time).total_seconds()
        if span_s >= min_move_seconds:
            return start_pos

    return None



# =====================================================================
# Trip create / close helpers
# =====================================================================
async def _start_trip(db, device_id: int, start_pos: Position, start_zone: Optional[str] = None) -> Trip:
    """
    Start a trip using refined start time with noise filtering.
    ALSO: logically ends any open STOP by filling resume_time + duration.
    """

    # ------------------------------
    # 1. Compute TRUE start time
    # ------------------------------
    true_start = await _find_true_start_time(db, device_id, start_pos)

    # ------------------------------
    # 2. Close the previous STOP
    # ------------------------------
    # Find the most recent stop with no resume_time
    q = await db.execute(
        select(Stop)
        .where(
            Stop.device_id == device_id,
            Stop.resume_time.is_(None)
        )
        .order_by(Stop.stop_time.desc())
        .limit(1)
    )
    open_stop = q.scalar_one_or_none()

    if open_stop:
        # Fill resume_time with refined start time
        open_stop.resume_time = true_start.server_time

        # Compute duration
        open_stop.duration_s = int(
            (open_stop.resume_time - open_stop.stop_time).total_seconds()
        )

        logger.info(
            f"[stop] Closed stop id={open_stop.id} device={device_id} "
            f"stop_time={open_stop.stop_time} resume={open_stop.resume_time} "
            f"duration={open_stop.duration_s}s"
        )

    # ------------------------------
    # 3. Create new trip
    # ------------------------------
    trip = Trip(
        device_id=device_id,
        start_time=true_start.server_time,
        start_zone=start_zone,
        start_addr=true_start.address,
        start_odometer=true_start.odometer,
    )

    db.add(trip)

    # ------------------------------
    # 4. Commit & refresh
    # ------------------------------
    await db.commit()
    await db.refresh(trip)

    logger.info(
        f"[trip] Start refined (noise filtered): device={device_id} "
        f"detected={start_pos.server_time} → true={true_start.server_time}"
    )

    return trip



# =====================================================================
# Trip create / close helpers
# When we End a trip
# We need to roll back to pick the exact time the trip ended, and fix that time inside stop table
# =====================================================================
async def _end_trip(db, device_id: int, end_pos: Position) -> Optional[Trip]:
    # ------------------------------------------
    # 1. Find open trip
    # ------------------------------------------
    q = await db.execute(
        select(Trip)
        .where(Trip.device_id == device_id, Trip.end_time.is_(None))
        .order_by(Trip.start_time.desc())
        .limit(1)
    )
    trip = q.scalar_one_or_none()
    if not trip:
        return None

    # ------------------------------------------
    # 2. Determine TRUE end time (noise filtered)
    # ------------------------------------------
    true_end_pos = await _find_true_end_time(db, device_id, end_pos)

    # ------------------------------------------
    # 3. Lookup zone
    # ------------------------------------------
    end_zone = await zone_at(db, true_end_pos.lat, true_end_pos.lon)

    # ------------------------------------------
    # 4. Update trip end fields
    # ------------------------------------------
    trip.end_time = true_end_pos.server_time
    trip.end_addr = true_end_pos.address
    trip.end_zone = end_zone["name"] if end_zone else None
    trip.end_odometer = true_end_pos.odometer

    # ------------------------------------------
    # 5. Compute distance
    # ------------------------------------------
    if trip.start_odometer is not None and true_end_pos.odometer is not None:
        dist = float(true_end_pos.odometer) - float(trip.start_odometer)
        trip.distance_m = max(dist, 0)

    # ------------------------------------------
    # 6. Duration
    # ------------------------------------------
    if trip.end_time and trip.start_time:
        trip.trip_duration = (trip.end_time - trip.start_time).total_seconds()

    # ------------------------------------------
    # 7. Speed stats
    # ------------------------------------------
    speed_query = await db.execute(
        select(
            func.max(Position.speed),
            func.avg(Position.speed)
        )
        .where(
            Position.device_id == device_id,
            Position.server_time >= trip.start_time,
            Position.server_time <= trip.end_time
        )
    )
    max_speed, avg_speed = speed_query.one()
    trip.max_speed = max_speed or 0
    trip.average_speed = avg_speed or 0

    # ------------------------------------------
    # 8. Create STOP entry (parking event)
    # ------------------------------------------
    stop_zone = end_zone["name"] if end_zone else None

    stop = Stop(
        device_id=device_id,
        stop_time=true_end_pos.server_time,   # moment vehicle actually stopped
        start_zone=trip.end_zone,
        stop_addr=true_end_pos.address,
        resume_time=None,
        zone=stop_zone,
        duration_s=None  # will be computed when motion resumes
    )

    db.add(stop)

    # ------------------------------------------
    # 9. Commit
    # ------------------------------------------
    await db.commit()
    await db.refresh(trip)

    logger.info(
        f"[trip] Ended trip id={trip.id} device={device_id} at {trip.end_time} "
        f"distance={trip.distance_m}m duration={trip.trip_duration}s "
        f"max_speed={trip.max_speed} avg_speed={trip.average_speed}"
    )
    logger.info(
        f"[stop] Created stop for device={device_id} at {stop.stop_time} "
        f"zone={stop.zone}"
    )

    return trip


# =====================================================================
# Utility: create or re-send violation alerts safely (active/inactive model)
# =====================================================================
async def create_or_send_alert(db, device_id, minutes, latest, *, is_offline):
    """
    PARKING-SAFE VERSION
    - When violation begins → store location ONCE.
    - While active → only update minutes_in_unapproved (NOT coordinates).
    - Do not overwrite address/lat/lon unless it's an offline violation re-check.
    """

    # Load truck name
    truck_row = await db.execute(select(Truck).where(Truck.device_id == device_id))
    truck = truck_row.scalar_one_or_none()
    truck_name = truck.name if truck else None

    logger.info(
        f"[create_or_send_alert] device={device_id} / {truck_name} "
        f"minutes={minutes} is_offline={is_offline}"
    )

    try:
        # -----------------------------------------------------------
        # 0) Cooldown check: recently inactive violation
        # -----------------------------------------------------------
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
                logger.info(
                    f"[create_or_send_alert] Cooldown active — "
                    f"recent inactive violation id={recent.id}, "
                    f"{delta.total_seconds()/60:.1f} minutes ago"
                )
                # Update only minutes, NOT address
                recent.minutes_in_unapproved = minutes
                await db.commit()
                return recent

        # -----------------------------------------------------------
        # 1) Active violation (same type)
        # -----------------------------------------------------------
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
            logger.info(
                f"[create_or_send_alert] Active violation exists id={existing.id} "
                f"alert_sent={existing.alert_sent}"
            )

            # Update minutes ONLY
            existing.minutes_in_unapproved = minutes
            await db.commit()

            # Only send alert once
            if not existing.alert_sent:
                logger.info(
                    f"[create_or_send_alert] Sending alert for existing violation id={existing.id}"
                )
                try:
                    await send_alert(existing, truck_name)
                    existing.alert_sent = True
                    await db.commit()
                except Exception:
                    logger.exception(
                        f"[create_or_send_alert] Error sending alert for violation {existing.id}"
                    )

            return existing

        # -----------------------------------------------------------
        # 2) No active violation → create a new one
        # -----------------------------------------------------------
        if minutes < PARKING_VIOLATION_THRESHOLD_MINUTES and not is_offline:
            logger.info(
                f"[create_or_send_alert] minutes={minutes} < threshold "
                f"{PARKING_VIOLATION_THRESHOLD_MINUTES}; skip creation"
            )
            return None

        # Create new violation: **store address/lat/lon ONCE**
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
            await send_alert(viol, truck_name)
            viol.alert_sent = True
            await db.commit()
        except Exception:
            logger.exception(
                f"[create_or_send_alert] Error sending alert for new violation id={viol.id}"
            )

        return viol

    except Exception as e:
        logger.exception(
            f"[create_or_send_alert] ERROR for device {device_id} / {truck_name}: {e}"
        )
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



# # =====================================================================
# # MAIN EVALUATOR (CLEAN + CONSISTENT WITH NEW VIOLATION ENGINE)
# # =====================================================================
async def evaluator(device_id: int) -> None:
    logger.info(f"[evaluator] Starting evaluation for device {device_id}")

    try:
        async with AsyncSessionLocal() as db:

            # ---------------------- Device & Truck ----------------------
            truck_row = await db.execute(
                select(Truck).where(Truck.device_id == device_id)
            )
            truck = truck_row.scalar_one_or_none()
            truck_name = truck.name if truck else "UNKNOWN"

            logger.info(f"[evaluator] TruckName={truck_name} device_id={device_id}")

            # ---------------------- Load recent positions ----------------------
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

            # --------------------- Zone Lookup ----------------------
            latest_zone = await zone_at(db, latest.lat, latest.lon)
            zone_label = f"IN zone {latest_zone['name']}" if latest_zone else "OUTSIDE zone"

            logger.info(
                f"[evaluator] Zone lookup device {device_id} / {truck_name} : {zone_label}"
            )

            # ------------------- TRIP START DETECTION --------------------
            # if await _is_trip_start(positions):

            #     # Check if there's already an open trip
            #     q = await db.execute(
            #         select(Trip)
            #         .where(Trip.device_id == device_id, Trip.end_time.is_(None))
            #         .limit(1)
            #     )
            #     open_trip = q.scalar_one_or_none()

            #     if not open_trip:
            #         latest_zone_name = await zone_at(db, latest.lat, latest.lon)
            #         zone_name = latest_zone_name["name"] if latest_zone_name else None

            #         await _start_trip(db, device_id, latest, start_zone=zone_name)

            #         sustained_movement = not await _is_stationary(positions)
            #         if sustained_movement:
            #             await _deactivate_active_violations(db, device_id)
            #             logger.info(
            #                 f"[evaluator] Deactivated active violations for {device_id} / {truck_name} (movement detected)"
            #             )
            #         else:
            #             logger.info(
            #                 f"[evaluator] Trip start detected but insufficient movement — not deactivating violations for {device_id} / {truck_name}"
            #             )

            # ------------------- TRIP START DETECTION (robust, movement-first) --------------------
            # Goal: start a trip when we detect sustained movement and there is no open trip.
            # _is_trip_start() remains an advisory check but should not be the only gate.
            # ------------------- TRIP START DETECTION (robust, movement-first) --------------------
            try:
                trip_start_advice = await _is_trip_start(positions)
            except Exception:
                trip_start_advice = False

            # Check for an already open trip
            q = await db.execute(
                select(Trip)
                .where(Trip.device_id == device_id, Trip.end_time.is_(None))
                .limit(1)
            )
            open_trip = q.scalar_one_or_none()

            # Determine whether we have recent stationary/movement
            sustained_movement_flag = not await _is_stationary(positions)

            # NEW: require sustained movement in seconds before actually starting a trip
            movement_anchor = await get_sustained_movement_start(positions, min_move_seconds=30)  # tune 30s
            movement_ok = movement_anchor is not None

            # Decide to start a trip:
            # - advisory (trip_start_advice) is ok but prefer movement_ok
            # - or sustained_movement_flag + movement_ok
            should_start_trip = (trip_start_advice and (movement_ok or sustained_movement_flag)) and (open_trip is None)
            # Alternative: if you want to allow immediate starts when advisory is True, remove movement_ok gating.

            if should_start_trip:
                latest_zone_name = await zone_at(db, latest.lat, latest.lon)
                zone_name = latest_zone_name["name"] if latest_zone_name else None

                await _start_trip(db, device_id, latest, start_zone=zone_name)

                if movement_ok:
                    await _deactivate_active_violations(db, device_id)
                    logger.info(
                        f"[evaluator] Deactivated active violations for {device_id} / {truck_name} (movement detected)"
                    )
                else:
                    logger.info(
                        f"[evaluator] Trip start detected but movement run too short — not deactivating violations for {device_id} / {truck_name}"
                    )


            # ------------------- TRIP END DETECTION --------------------
            # Only consider ending a trip if we observe a trip-end pattern AND the stationary run is sustained
            try:
                trip_end_advice = await _is_trip_end(positions)
            except Exception:
                trip_end_advice = False

            if trip_end_advice:
                # Check whether we have a sustained stationary run (time-based) before ending
                stationary_anchor = await get_stationary_start(positions, min_stop_seconds=MIN_SUSTAINED_STOP_SECONDS)
                if stationary_anchor:
                    ended = await _end_trip(db, device_id, latest)
                    if ended:
                        logger.info(
                            f"[evaluator] Trip ended for device {device_id} / {truck_name}, trip id={ended.id}"
                        )
                else:
                    logger.info(
                        f"[evaluator] Trip-end candidate but stationary run shorter than {MIN_SUSTAINED_STOP_SECONDS}s -> skipping end"
                    )

            # =====================================================
            # PARKED VIOLATION (only if OUTSIDE zone and STATIONARY)
            # =====================================================
            if not latest_zone:

                logger.info(
                    f"[evaluator] Scanning backward for parked start: device {device_id} / {truck_name}"
                )

                # If moving → no parking violation
                if not await _is_stationary(positions):
                    logger.info(
                        f"[evaluator] Device {device_id} / {truck_name} not stationary → skipping parked violation"
                    )
                else:
                    # 1) Prefer trip-end anchor
                    parked_start_pos: Optional[Position] = None

                    q_trip = await db.execute(
                        select(Trip)
                        .where(Trip.device_id == device_id, Trip.end_time.isnot(None))
                        .order_by(Trip.end_time.desc())
                        .limit(1)
                    )
                    recent_trip = q_trip.scalar_one_or_none()

                    if recent_trip and recent_trip.end_time:
                        if recent_trip.end_time >= positions[-1].server_time and recent_trip.end_time <= latest.server_time:
                            for p in reversed(positions):
                                if p.server_time >= recent_trip.end_time:
                                    parked_start_pos = p
                                    break
                            if parked_start_pos is None:
                                parked_start_pos = positions[-1]

                    # 2) Fallback to sustained-stationary detection
                    if parked_start_pos is None:
                        parked_start_pos = await get_stationary_start(
                            positions,
                            min_stop_seconds=MIN_SUSTAINED_STOP_SECONDS
                        )

                    if parked_start_pos is None:
                        logger.info(
                            f"[evaluator] No sustained stationary period found → skipping parked violation for device {device_id} / {truck_name}"
                        )
                    else:
                        minutes = int(
                            (latest.server_time - parked_start_pos.server_time).total_seconds() / 60
                        )

                        logger.info(
                            f"[evaluator] Device {device_id} / {truck_name} parked_start={parked_start_pos.server_time} | parked_minutes={minutes}"
                        )

                        # Let create_or_send_alert manage thresholds & states
                        await create_or_send_alert(
                            db=db,
                            device_id=device_id,
                            minutes=minutes,
                            latest=latest,
                            is_offline=False
                        )

            # =====================================================
            # OFFLINE VIOLATION
            # =====================================================
            status = await db.get(DeviceStatus, device_id)

            if status:
                now_utc = dt.datetime.now(UTC)

                # If offline for >= 60 minutes
                if status.last_seen is None or (now_utc - status.last_seen).total_seconds() >= 60 * 60:

                    minutes_off = int(
                        (now_utc - (status.last_seen or now_utc)).total_seconds() / 60
                    )

                    logger.info(
                        f"[evaluator] Device {device_id} / {truck_name} offline for {minutes_off} minutes"
                    )

                    if minutes_off >= 60:
                        await create_or_send_alert(
                            db=db,
                            device_id=device_id,
                            minutes=minutes_off,
                            latest=latest,
                            is_offline=True
                        )
                else:
                    # Device came back online → auto-resolve offline violations
                    await _resolve_offline_violations_on_online(db, device_id, now_utc)

        logger.info(f"[evaluator] Completed evaluation for device {device_id} / {truck_name}")
    except Exception as e:
        logger.exception(
            f"[evaluator] ERROR evaluating device {device_id} / {truck_name} : {e}"
        )
        raise



# # =====================================================================
# # MAIN EVALUATOR (OPTIMIZED, IMPROVED FLOW)
# # =====================================================================
# async def evaluator(device_id: int) -> None:
#     logger.info(f"[evaluator] Starting evaluation for device {device_id}")

#     try:
#         async with AsyncSessionLocal() as db:

#             # ---------------------- Device & Truck ----------------------
#             truck_row = await db.execute(
#                 select(Truck).where(Truck.device_id == device_id)
#             )
#             truck = truck_row.scalar_one_or_none()
#             truck_name = truck.name if truck else "UNKNOWN"

#             logger.info(f"[evaluator] TruckName={truck_name} device_id={device_id}")

#             # ---------------------- Load recent positions ----------------------
#             now = dt.datetime.now(UTC)
#             since = now - dt.timedelta(minutes=70)

#             pos_rows = await db.execute(
#                 select(Position)
#                 .where(
#                     Position.device_id == device_id,
#                     Position.server_time >= since
#                 )
#                 .order_by(Position.server_time.desc())
#             )
#             positions = pos_rows.scalars().all()

#             logger.info(
#                 f"[evaluator] Loaded {len(positions)} positions for device {device_id} / {truck_name}"
#             )

#             if len(positions) < 2:
#                 logger.info("[evaluator] Not enough data → skipping")
#                 return

#             latest = positions[0]

#             # --------------------- Single zone lookup ----------------------
#             latest_zone = await zone_at(db, latest.lat, latest.lon)
#             zone_name = latest_zone["name"] if latest_zone else None
#             zone_label = f"IN zone {zone_name}" if latest_zone else "OUTSIDE zone"

#             logger.info(
#                 f"[evaluator] Zone lookup device {device_id} / {truck_name} : {zone_label}"
#             )

#             # ------------------- Precompute commonly-used values --------------------
#             # compute stationary once (used for trip start decision and parked violation)
#             is_stationary = await _is_stationary(positions)
#             sustained_movement = not is_stationary

#             # get current open trip (if any) once
#             q = await db.execute(
#                 select(Trip)
#                 .where(Trip.device_id == device_id, Trip.end_time.is_(None))
#                 .limit(1)
#             )
#             open_trip = q.scalar_one_or_none()

#             logger.info(
#                 f"[evaluator] open_trip={'yes' if open_trip else 'no'} sustained_movement={sustained_movement} is_stationary={is_stationary}"
#             )

#             # ------------------- TRIP END DETECTION (if a trip is open) --------------------
#             # Prefer to end an existing trip before starting another
#             # ------------------- TRIP END DETECTION --------------------
#             if open_trip:
#                 try:
#                     if await _is_trip_end(positions):
#                         ended = await _end_trip(db, device_id, latest)

#                         if ended:
#                             logger.info(
#                                 f"[evaluator] Trip ended for device {device_id} / {truck_name}, trip id={ended.id}"
#                             )

#                             # Refresh open_trip
#                             open_trip = None

#                 except Exception as e:
#                     logger.exception(
#                         f"[evaluator] Error during trip end detection for device {device_id}: {e}"
#                     )

#             # ------------------- TRIP START DETECTION (movement-first, advisory allowed) --------------------
#             # Don't attempt to start a trip if there is already one open.
#             if open_trip is None:
#                 try:
#                     try:
#                         trip_start_advice = await _is_trip_start(positions)
#                     except Exception:
#                         logger.exception(f"[evaluator] _is_trip_start failed for device {device_id}, treating as False")
#                         trip_start_advice = False

#                     should_start_trip = trip_start_advice or sustained_movement

#                     logger.info(
#                         f"[evaluator] trip_start_advice={trip_start_advice} should_start_trip={should_start_trip}"
#                     )

#                     if should_start_trip:
#                         await _start_trip(db, device_id, latest, start_zone=zone_name)

#                         # Now safe to proceed with violations
#                         if sustained_movement:
#                             try:
#                                 await _deactivate_active_violations(db, device_id)
#                                 await db.commit()
#                             except Exception:
#                                 logger.exception("[evaluator] deactivate violations failed")

#                 except Exception as e:
#                     logger.exception(
#                         f"[evaluator] Error during trip start detection for device {device_id}: {e}"
#                     )
#             # -------------------- PARKED VIOLATION (only if OUTSIDE zone and STATIONARY) --------------------
#             if not latest_zone:
#                 logger.info(
#                     f"[evaluator] Scanning backward for parked start: device {device_id} / {truck_name}"
#                 )

#                 if not is_stationary:
#                     logger.info(
#                         f"[evaluator] Device {device_id} / {truck_name} not stationary → skipping parked violation"
#                     )
#                 else:
#                     # 1) Prefer trip-end anchor (use last finished trip if it falls within window)
#                     parked_start_pos: Optional[Position] = None

#                     q_trip = await db.execute(
#                         select(Trip)
#                         .where(Trip.device_id == device_id, Trip.end_time.isnot(None))
#                         .order_by(Trip.end_time.desc())
#                         .limit(1)
#                     )
#                     recent_trip = q_trip.scalar_one_or_none()

#                     if recent_trip and recent_trip.end_time:
#                         try:
#                             # if recent_trip ended within the positions window, anchor parked start to that time
#                             if recent_trip.end_time >= positions[-1].server_time and recent_trip.end_time <= latest.server_time:
#                                 for p in reversed(positions):
#                                     if p.server_time >= recent_trip.end_time:
#                                         parked_start_pos = p
#                                         break
#                                 if parked_start_pos is None:
#                                     parked_start_pos = positions[-1]
#                         except Exception:
#                             logger.exception("[evaluator] error while anchoring parked_start_pos to recent_trip")

#                     # 2) Fallback to sustained-stationary detection
#                     if parked_start_pos is None:
#                         try:
#                             parked_start_pos = await get_stationary_start(
#                                 positions,
#                                 min_stop_seconds=MIN_SUSTAINED_STOP_SECONDS
#                             )
#                         except Exception:
#                             logger.exception("[evaluator] get_stationary_start failed")

#                     if parked_start_pos is None:
#                         logger.info(
#                             f"[evaluator] No sustained stationary period found → skipping parked violation for device {device_id} / {truck_name}"
#                         )
#                     else:
#                         minutes = int(
#                             (latest.server_time - parked_start_pos.server_time).total_seconds() / 60
#                         )

#                         logger.info(
#                             f"[evaluator] Device {device_id} / {truck_name} parked_start={parked_start_pos.server_time} | parked_minutes={minutes}"
#                         )

#                         # create_or_send_alert does DB commits internally; call it to manage thresholds & states
#                         try:
#                             await create_or_send_alert(
#                                 db=db,
#                                 device_id=device_id,
#                                 minutes=minutes,
#                                 latest=latest,
#                                 is_offline=False
#                             )
#                         except Exception:
#                             logger.exception("[evaluator] create_or_send_alert failed for parked violation")

#             # =====================================================
#             # OFFLINE VIOLATION
#             # =====================================================
#             status = await db.get(DeviceStatus, device_id)

#             if status:
#                 now_utc = dt.datetime.now(UTC)

#                 # If offline for >= 60 minutes
#                 if status.last_seen is None or (now_utc - status.last_seen).total_seconds() >= 60 * 60:

#                     minutes_off = int(
#                         (now_utc - (status.last_seen or now_utc)).total_seconds() / 60
#                     )

#                     logger.info(
#                         f"[evaluator] Device {device_id} / {truck_name} offline for {minutes_off} minutes"
#                     )

#                     if minutes_off >= 60:
#                         try:
#                             await create_or_send_alert(
#                                 db=db,
#                                 device_id=device_id,
#                                 minutes=minutes_off,
#                                 latest=latest,
#                                 is_offline=True
#                             )
#                         except Exception:
#                             logger.exception("[evaluator] create_or_send_alert failed for offline violation")
#                 else:
#                     # Device came back online → auto-resolve offline violations
#                     try:
#                         await _resolve_offline_violations_on_online(db, device_id, now_utc)
#                         await db.commit()
#                     except Exception:
#                         logger.exception("[evaluator] _resolve_offline_violations_on_online failed")

#         logger.info(f"[evaluator] Completed evaluation for device {device_id} / {truck_name}")
#     except Exception as e:
#         logger.exception(
#             f"[evaluator] ERROR evaluating device {device_id} / {truck_name} : {e}"
#         )
#         raise

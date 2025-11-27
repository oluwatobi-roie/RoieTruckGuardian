# app/monitor.py
import datetime as dt
import pytz
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from database import AsyncSessionLocal
from models import Position, Violation, DeviceStatus
from geozones import zone_at
from alerts import send_alert

from logging_config import get_logger
logger = get_logger("monitor", "monitor.log")


# =====================================================================
# Utility: create or re-send violation alerts safely
# =====================================================================
async def create_or_send_alert(db, device_id, minutes, latest, *, is_offline):
    logger.info(
        f"[create_or_send_alert] device={device_id} minutes={minutes} "
        f"is_offline={is_offline}"
    )

    try:
        # Always fetch the latest unresolved violation (if any)
        existing_q = await db.execute(
            select(Violation)
            .where(
                Violation.device_id == device_id,
                Violation.resolved_at.is_(None),
                Violation.is_offline_violation == is_offline,
            )
            .order_by(Violation.detected_at.desc())
            .limit(1)
            .with_for_update()
        )

        existing = existing_q.scalar_one_or_none()

        # CASE 1: existing violation found but alert not sent yet → send once
        if existing:
            if not existing.alert_sent:
                logger.info(
                    f"[create_or_send_alert] Existing violation id={existing.id}, sending alert"
                )
                await send_alert(existing)
                existing.alert_sent = True
                await db.commit()
            else:
                logger.info(
                    f"[create_or_send_alert] Violation id={existing.id} already alerted → skipping"
                )
            return

        # CASE 2: No active violation → create new violation
        detected_at = dt.datetime.now(dt.timezone.utc)

        viol = Violation(
            device_id=device_id,
            detected_at=detected_at,
            lat=latest.lat,
            lon=latest.lon,
            address=latest.address,
            minutes_in_unapproved=minutes,
            is_offline_violation=is_offline,
            alert_sent=False,
        )

        db.add(viol)
        await db.commit()
        await db.refresh(viol)

        logger.info(f"[create_or_send_alert] New violation id={viol.id}, sending alert")
        await send_alert(viol)

        viol.alert_sent = True
        await db.commit()

    except Exception as e:
        logger.exception(f"[create_or_send_alert] ERROR for device {device_id}: {e}")
        raise

# =====================================================================
# MAIN EVALUATOR
# =====================================================================
async def evaluator(device_id: int) -> None:
    logger.info(f"[evaluator] Starting evaluation for device {device_id}")

    try:
        async with AsyncSessionLocal() as db:

            # ------------------- TIME WINDOW --------------------
            now = dt.datetime.now(dt.timezone.utc)
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
    
            logger.info(
                f"[evaluator] Loaded {len(positions)} positions for device {device_id}"
            )

            if len(positions) < 2:
                logger.info(f"[evaluator] Not enough data → skipping")
                return

            latest = positions[0]

            # ------------------- ZONE LOOKUP --------------------
            latest_zone = await zone_at(db, latest.lat, latest.lon)

            logger.info(
                f"[evaluator] Zone lookup device {device_id}: "
                f"{'IN zone' if latest_zone else 'OUTSIDE zone'}"
            )

            # =====================================================
            # PARKED VIOLATION (out of zone for >= 60min)
            # =====================================================
            if not latest_zone:
                logger.info(
                    f"[evaluator] Scanning backward to detect parked start for {device_id}"
                )

                start = positions[0]  # default = latest
                found_inside = False

                # Walk from newest -> oldest
                for idx in range(1, len(positions)):
                    p = positions[idx]
                    inside = await zone_at(db, p.lat, p.lon)

                    if inside:
                        found_inside = True
                        start = positions[idx - 1]  # last outside point before entering zone
                        break

                # If we NEVER found an inside-zone point
                if not found_inside:
                    start = positions[-1]  # oldest position → start of 70-min window

                minutes = int((latest.server_time - start.server_time).total_seconds() / 60)

                logger.info(
                    f"[evaluator] Device {device_id} OUTSIDE zone for {minutes} mins"
                )

                if minutes >= 60:
                    await create_or_send_alert(
                        db=db,
                        device_id=device_id,
                        minutes=minutes,
                        latest=latest,
                        is_offline=False
                    )

            # ===========================================================
            # OFFLINE VIOLATION
            # ===========================================================
            status = await db.get(DeviceStatus, device_id)

            if status and not status.online and status.offline_since:

                minutes_off = int(
                    (dt.datetime.utcnow() - status.offline_since).total_seconds() / 60
                )

                logger.info(
                    f"[evaluator] Device {device_id} offline for {minutes_off} minutes"
                )

                if minutes_off >= 60:

                    if not latest_zone:
                        logger.info(
                            f"[evaluator] OFFLINE VIOLATION triggered for device {device_id}"
                        )
                        await create_or_send_alert(
                            db=db,
                            device_id=device_id,
                            minutes=minutes_off,
                            latest=latest,
                            is_offline=True
                        )

        logger.info(f"[evaluator] Completed evaluation for device {device_id}")

    except Exception as e:
        logger.exception(
            f"[evaluator] ERROR evaluating device {device_id}: {e}"
        )
        raise

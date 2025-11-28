# worker.py (rewritten - drop-in)
import datetime as dt
import redis
import json
import asyncio

from database import AsyncSessionLocal
from models import Position, Truck, Trip, DeviceStatus
from crud import upsert_device_status, to_dt
from monitor import evaluator
from config import ALLOWED_DEVICES
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.dialects.postgresql import insert as pg_insert
import asyncpg

from logging_config import get_logger

# ------------ Variable Declaration -----------
logger = get_logger("worker", "worker.log")

# Redis stream config (same names to avoid breaking other code)
r = redis.from_url("redis://redis:6379", decode_responses=False)

STREAM = "traccar"
GROUP = "worker-group"
CONSUMER = "worker-1"


# ---------- Ensure Consumer Group ----------
async def init_group():
    try:
        # Start reading only NEW messages from now → id="$", create stream if missing
        r.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
        logger.info("Consumer group created.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group already exists.")
        else:
            raise


# ---------- Save logic ----------
# save_with_retry now accepts (db, payload) so we don't open nested sessions.
@retry(
    wait=wait_exponential_jitter(initial=2, max=30),
    stop=stop_after_attempt(7),
    retry=retry_if_exception_type(
        (DBAPIError, OperationalError, OSError, asyncpg.CannotConnectNowError)
    ),
)
async def save_with_retry(db, payload: dict) -> None:
    """
    Retry wrapper around save_to_db. Expects an active AsyncSessionLocal() session (db).
    """
    device_id = payload["position"]["deviceId"]
    try:
        logger.info(f"Saving with retry for device {device_id}")
        await save_to_db(db, payload)
    except Exception as e:
        logger.exception(f"Failure in save_with_retry for device {device_id}: {e}")
        raise


async def save_to_db(db, payload: dict) -> None:
    """
    Persist position + ensure device status updated.
    This function uses the provided session (db) and commits changes itself.
    """
    p = payload["position"]
    device_id = p["deviceId"]
    logger.info(f"Saving position for Device {device_id}")

    try:
        # Upsert truck (ignore duplicates)
        stmt = pg_insert(Truck).values(
            device_id=device_id,
            name=payload["device"].get("name"),
            phone=payload["device"].get("phone"),
            contact=payload["device"].get("contact")
        ).on_conflict_do_nothing(index_elements=["device_id"])

        await db.execute(stmt)
        logger.info(f"Upserted truck: {device_id}")

        # Build Position
        pos = Position(
            device_id=device_id,
            lat=p.get("latitude"),
            lon=p.get("longitude"),
            speed=p.get("speed"),
            ignition=p.get("attributes", {}).get("ignition"),
            motion=p.get("attributes", {}).get("motion"),
            fix_time=to_dt(p["fixTime"]),
            server_time=to_dt(p["serverTime"]),
            address=p.get("address")
        )

        db.add(pos)
        await db.commit()
        await db.refresh(pos)
        logger.info(f"[Worker] Saved new position row for {device_id} id={pos.id}")

        # Update device status via helper so logic is centralized
        online = payload.get("device", {}).get("status") == "online"
        await upsert_device_status(db, device_id, online, pos.server_time)
        logger.info(f"Updated device_status for {device_id} online={online}")

        # IMPORTANT: do NOT run violation/evaluator here — worker will call evaluator after commit
        # Also do NOT create offline violations here — evaluator handles that using DeviceStatus

    except Exception as e:
        logger.exception(f"save_to_db() failed for device {device_id}: {e}")
        raise


# ---------- Main Worker Loop ----------
async def worker():
    logger.info("Worker starting, initializing consumer group...")
    await init_group()

    logger.info("Worker listening for Redis Stream messages...")

    while True:
        try:
            # Blocking read via executor (call will block the threadpool, not the event loop)
            msgs = await asyncio.get_event_loop().run_in_executor(
                None,
                r.xreadgroup,
                GROUP,
                CONSUMER,
                {STREAM: ">"},
                100,
                5000  # block 5 seconds
            )

            if msgs:
                total = sum(len(rec[1]) for rec in msgs)
                logger.info(f"Fetched {total} records from stream")

            # iterate messages (msgs can be None or [] if timeout)
            for _, records in msgs or []:
                for _id, fields in records:
                    try:
                        # load payload
                        try:
                            payload = json.loads(fields[b"data"])
                        except Exception as je:
                            logger.exception(f"Failed to json-decode record {_id}: {je}")
                            # malformed message — ack & delete to avoid poison-pill
                            r.xack(STREAM, GROUP, _id)
                            r.xdel(STREAM, _id)
                            continue

                        # device filter (ensure types align)
                        device_id = payload["position"]["deviceId"]
                        # If ALLOWED_DEVICES contains strings ensure comparison is normalized
                        if isinstance(device_id, int) and device_id not in ALLOWED_DEVICES:
                            if str(device_id) not in ALLOWED_DEVICES:
                                logger.warning(f"Skipping unauthorized device: {device_id}")
                                r.xack(STREAM, GROUP, _id)
                                r.xdel(STREAM, _id)
                                continue
                        elif isinstance(device_id, str) and device_id not in ALLOWED_DEVICES:
                            logger.warning(f"Skipping unauthorized device: {device_id}")
                            r.xack(STREAM, GROUP, _id)
                            r.xdel(STREAM, _id)
                            continue

                        logger.info(f"Processing device: {device_id}")

                        # Use a single DB session per message
                        async with AsyncSessionLocal() as db:
                            # Save to DB (with retry using the provided session)
                            await save_with_retry(db, payload)
                            logger.info(f"Saved data for {device_id}, running evaluator...")

                            # Run evaluator (which will open its own session as needed)
                            await evaluator(device_id)
                            logger.info(f"Evaluator completed for {device_id}")

                        # If everything succeeded: ACK and DELETE
                        try:
                            r.xack(STREAM, GROUP, _id)
                            r.xdel(STREAM, _id)
                        except Exception as rexc:
                            logger.exception(f"Failed to ack/xdel message {_id}: {rexc}")

                    except Exception as e:
                        # DO NOT ack/delete on general processing failure so message can be retried
                        logger.exception(f"Error processing record ID {_id}: {e}")

        except Exception as e:
            logger.exception(f"Worker loop encountered an error: {e}")

        # small sleep to avoid tight loop in case of unexpected fast failures
        await asyncio.sleep(0.1)


if __name__ == "__main__":
    logger.info("Worker starting up...")
    asyncio.run(worker())

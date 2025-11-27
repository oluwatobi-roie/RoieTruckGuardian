import redis, json, asyncio
from database import AsyncSessionLocal
from models import Position, Truck
from crud import upsert_device_status, to_dt
from monitor import evaluator
from config import ALLOWED_DEVICES
from tenacity import retry, wait_exponential_jitter, stop_after_attempt, retry_if_exception_type
from sqlalchemy.exc import DBAPIError, OperationalError
from sqlalchemy.dialects.postgresql import insert as pg_insert
import asyncpg

# --- new import ---
from logging_config import get_logger

# create worker logger
logger = get_logger("worker", "worker.log")

# Create Redis for caching
r = redis.from_url("redis://redis:6379", decode_responses=False)

STREAM = "traccar"
GROUP = "worker-group"
CONSUMER = "worker-1"


# ---------- Ensure Consumer Group ----------
async def init_group():
    try:
        # Start reading only NEW messages from now → id="$"
        r.xgroup_create(STREAM, GROUP, id="$", mkstream=True)
        logger.info("Consumer group created.")
    except redis.exceptions.ResponseError as e:
        if "BUSYGROUP" in str(e):
            logger.info("Consumer group already exists.")
        else:
            raise


@retry(
    wait=wait_exponential_jitter(initial=2, max=30),
    stop=stop_after_attempt(7),
    retry=retry_if_exception_type(
        (DBAPIError, OperationalError, OSError, asyncpg.CannotConnectNowError)
    ),
)
async def save_with_retry(payload: dict) -> None:
    try:
        logger.info(f"Saving with retry for device {payload['position']['deviceId']}")
        async with AsyncSessionLocal() as db:
            await save_to_db(db, payload)
    except Exception as e:
        logger.exception(
            f"Failure in save_with_retry for device "
            f"{payload['position']['deviceId']}: {e}"
        )
        raise


# ---------- missing function ----------
async def save_to_db(db, payload: dict) -> None:
    p = payload["position"]
    device_id = p["deviceId"]
    logger.info(f"Saving position for Device {device_id}")

    try:
        stmt = pg_insert(Truck).values(
            device_id=device_id,
            name=payload["device"]["name"],
            phone=payload["device"]["phone"],
            contact=payload["device"]["contact"]
        ).on_conflict_do_nothing(index_elements=["device_id"])

        await db.execute(stmt)
        logger.info(f"Upserted truck: {device_id}")

        pos = Position(
            device_id=device_id,
            lat=p["latitude"],
            lon=p["longitude"],
            speed=p["speed"],
            ignition=p["attributes"].get("ignition"),
            motion=p["attributes"].get("motion"),
            fix_time=to_dt(p["fixTime"]),
            server_time=to_dt(p["serverTime"]),
            address=p["address"]
        )

        db.add(pos)
        await db.commit()
        logger.info(f"Saved new position row for {device_id}")

        online = payload["device"]["status"] == "online"
        await upsert_device_status(db, device_id, online, pos.server_time)
        logger.info(f"Updated device_status for {device_id} online={online}")

    except Exception as e:
        logger.exception(f"save_to_db() failed for device {device_id}: {e}")
        raise


# ---------- NEW worker loop using consumer-group ----------
async def worker():
    logger.info("Worker starting, initializing consumer group...")
    await init_group()

    logger.info("Worker listening for Redis Stream messages...")

    while True:
        try:
            # Only new messages ">"
            msgs = await asyncio.get_event_loop().run_in_executor(
                None,
                r.xreadgroup,
                GROUP,
                CONSUMER,
                {STREAM: ">"},
                100,
                5000  # 5-second block
            )

            if msgs:
                logger.info(f"Fetched {sum(len(rec[1]) for rec in msgs)} records from stream")

            for _, records in msgs or []:
                for _id, fields in records:
                    try:
                        payload = json.loads(fields[b"data"])
                        device_id = payload["position"]["deviceId"]

                        if device_id not in ALLOWED_DEVICES:
                            logger.warning(f"Skipping unauthorized device: {device_id}")
                            # ACK and delete unauthorized
                            r.xack(STREAM, GROUP, _id)
                            r.xdel(STREAM, _id)
                            continue

                        logger.info(f"Processing device: {device_id}")

                        async with AsyncSessionLocal() as db:
                            await save_with_retry(payload)
                            logger.info(f"Saved data for {device_id}, running evaluator...")
                            await evaluator(device_id)
                            logger.info(f"Evaluator completed for {device_id}")

                        # Mark message done + delete
                        r.xack(STREAM, GROUP, _id)
                        r.xdel(STREAM, _id)

                    except Exception as e:
                        logger.exception(f"Error processing record ID {_id}: {e}")
                        # Do NOT delete if processing failed

        except Exception as e:
            logger.exception(f"Worker loop encountered an error: {e}")

        await asyncio.sleep(0.1)


if __name__ == "__main__":
    logger.info("Worker starting up...")
    asyncio.run(worker())

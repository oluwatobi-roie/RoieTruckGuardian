import redis, json, time, asyncio
from fastapi import APIRouter, HTTPException

from config import ALLOWED_DEVICES, REDIS_URL
from logging_config import get_logger

# DB imports for resolving violations
import datetime as dt
from database import AsyncSessionLocal
from models import Violation

# Redis connection (binary mode)
r = redis.from_url(REDIS_URL, decode_responses=False)

router = APIRouter()
logger = get_logger("webhook", "webhook.log")


# ---------------------------------------------------
#                TRACCAR WEBHOOK
# ---------------------------------------------------
@router.post("/webhook")
async def traccar_hook(payload: dict):

    # Extract position block
    p = payload.get("position")
    if not p:
        raise HTTPException(status_code=400, detail="no position")

    device_id = p.get("deviceId")
    if device_id is None:
        raise HTTPException(status_code=400, detail="missing deviceId")

    # Normalize allow-list lookup (handles int vs string mismatch)
    if str(device_id) not in {str(x) for x in ALLOWED_DEVICES}:
        logger.info(f"Ignored device {device_id} (not in ALLOWED_DEVICES)")
        return {"ok": False, "reason": "ignored"}

    # Convert payload to string for Redis
    json_str = json.dumps(payload, ensure_ascii=False)

    # Push to Redis inside executor (non-blocking)
    await asyncio.get_event_loop().run_in_executor(
        None,
        r.xadd,
        "traccar",
        {"ts": time.time(), "data": json_str},
    )

    logger.info(f"Payload stored for device {device_id}")
    return {"ok": True}



# ---------------------------------------------------
#         RESOLVE A VIOLATION MANUALLY
# ---------------------------------------------------
@router.post("/violations/{viol_id}/resolve")
async def resolve_violation(viol_id: int, payload: dict):

    async with AsyncSessionLocal() as db:
        v = await db.get(Violation, viol_id)
        if not v:
            raise HTTPException(status_code=404, detail="Violation not found")

        v.resolved_at = dt.datetime.now(dt.timezone.utc)
        v.resolution = payload.get("resolution")
        v.resolved_by = payload.get("resolved_by")

        await db.commit()

        logger.info(
            f"Violation {viol_id} resolved by={v.resolved_by}, reason={v.resolution}"
        )

        return {"ok": True}

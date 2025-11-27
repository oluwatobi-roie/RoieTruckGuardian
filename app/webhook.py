import redis, json, time, asyncio
from fastapi import APIRouter, HTTPException
from config import ALLOWED_DEVICES, REDIS_URL   # redis://redis:6379

r = redis.from_url(REDIS_URL, decode_responses=False)
router = APIRouter()

from logging_config import get_logger

# create worker logger
logger = get_logger("webhook", "webhook.log")

@router.post("/webhook")
async def traccar_hook(payload: dict):
 
    p = payload.get("position")
    # logger.info(f"Payload = {p}")
    # --- new import ---

    if not p:
        raise HTTPException(status_code=400, detail="no position")
    device_id = p["deviceId"]
    # allow-list check (hot path)
    if device_id not in ALLOWED_DEVICES:
        return {"ok": False, "reason": "ignored"}

    # push to stream (< 1 ms)
    await asyncio.get_event_loop().run_in_executor(
        None, r.xadd, "traccar", {"ts": time.time(), "data": json.dumps(payload)})
    return {"ok": True}
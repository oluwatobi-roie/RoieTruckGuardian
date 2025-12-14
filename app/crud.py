from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from models import DeviceStatus
import datetime as dt
from logging_config import get_logger

async def upsert_device_status(db: AsyncSession, device_id:int, online:bool, ts:dt.datetime):
    row = await db.get(DeviceStatus, device_id)
    if row:
        row.online        = online
        row.last_seen     = ts
        row.offline_since = None if online else ts
    else:
        db.add(DeviceStatus(device_id=device_id, online=online, last_seen=ts,
                            offline_since=None if online else ts))
    await db.commit()



def to_dt(v):
    if isinstance(v, dt.datetime):
        # ensure tz-aware
        return v if v.tzinfo else v.replace(tzinfo=dt.timezone.utc)

    if isinstance(v, str):
        dt_obj = dt.datetime.fromisoformat(v)
        return dt_obj if dt_obj.tzinfo else dt_obj.replace(tzinfo=dt.timezone.utc)

    return None




trip_debug_logger = get_logger("debug", "debug.log")
def log_trip_debug( device_id: int, sampling_intervals: list[int] = None, last_positions: list = None, trip_end_triggered: bool = False, zone_suppression_info: str = None):
    """
    Log structured debugging information for trip start/end detection.
    """
    trip_debug_logger.debug(f"[Device {device_id}] Debug Report:")

    if sampling_intervals:
        trip_debug_logger.debug(
            f"[Device {device_id}] Sampling intervals (sec): {sampling_intervals}"
        )

    if last_positions:
        # last_positions: list[(timestamp, speed)]
        pos_dump = ", ".join(
            f"({ts}, {spd} km/h)" for ts, spd in last_positions
        )
        trip_debug_logger.debug(
            f"[Device {device_id}] Last positions before decision: {pos_dump}"
        )

    if trip_end_triggered:
        trip_debug_logger.debug(
            f"[Device {device_id}] _is_trip_end returned TRUE"
        )

    if zone_suppression_info:
        trip_debug_logger.debug(
            f"[Device {device_id}] Zone suppression: {zone_suppression_info}"
        )
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, insert, update
from models import DeviceStatus
import datetime as dt

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

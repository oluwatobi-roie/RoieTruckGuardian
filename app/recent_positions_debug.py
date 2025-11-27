import asyncio
import datetime as dt
from database import AsyncSessionLocal
from sqlalchemy import select
from models import Position, DeviceStatus

async def debug():
    async with AsyncSessionLocal() as db:
        # get device ids we normally evaluate (from device_status)
        ds = (await db.execute(select(DeviceStatus.device_id).distinct())).fetchall()
        device_ids = [r[0] for r in ds]
        print("Checking devices:", device_ids)

        now_utc = dt.datetime.utcnow()
        print("Now (utc naive):", now_utc, " tzinfo:", now_utc.tzinfo)

        for device_id in device_ids:
            # latest position for device
            row = (await db.execute(
                select(Position)
                .where(Position.device_id == device_id)
                .order_by(Position.server_time.desc())
                .limit(1)
            )).scalar_one_or_none()

            if not row:
                print(f"device={device_id}: no positions at all")
                continue

            latest_time = row.server_time
            # compute minutes since latest (try safely)
            try:
                delta = dt.datetime.utcnow() - latest_time
                minutes_ago = int(delta.total_seconds() / 60)
            except Exception as e:
                minutes_ago = None
            print(f"device={device_id} latest_server_time={latest_time} tzinfo={getattr(latest_time,'tzinfo',None)} minutes_ago={minutes_ago}")

        # Also show how many positions exist in last 70 minutes for all devices
        since = dt.datetime.utcnow() - dt.timedelta(minutes=70)
        rows = (await db.execute(
            select(Position.device_id).where(Position.server_time >= since).distinct()
        )).fetchall()
        print("Devices WITH positions in last 70 minutes:", [r[0] for r in rows])

if __name__ == '__main__':
    asyncio.run(debug())

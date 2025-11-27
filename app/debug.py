import asyncio
from database import AsyncSessionLocal
from sqlalchemy import select
from models import Position, DeviceStatus

async def debug():
    async with AsyncSessionLocal() as db:
        print("\n--- DEBUG START ---\n")

        all_positions = (await db.execute(select(Position))).scalars().all()
        print("TOTAL POSITIONS:", len(all_positions))

        pos_devices = (await db.execute(select(Position.device_id).distinct())).fetchall()
        print("POSITION DEVICE IDS:", pos_devices)

        ds_devices = (await db.execute(select(DeviceStatus.device_id).distinct())).fetchall()
        print("DEVICE_STATUS IDS:", ds_devices)

        print("\n--- DEBUG END ---\n")

if __name__ == "__main__":
    asyncio.run(debug())

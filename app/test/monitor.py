from database import AsyncSessionLocal
from sqlalchemy import select
from models import Position

async def test():
    async with AsyncSessionLocal() as db:
        rows = (await db.execute(select(Position))).scalars().all()
        print("Total positions:", len(rows))

import asyncio
asyncio.run(test())
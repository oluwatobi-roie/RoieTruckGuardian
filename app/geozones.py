from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

async def zone_at(db: AsyncSession, lat: float, lon: float):
    row = await db.execute(
        text("""
            SELECT name, category
            FROM recognised_zones
            WHERE ST_Covers(
                geom,
                ST_SetSRID(ST_MakePoint(:lon, :lat), 4326)::geography
            )
        """).bindparams(lon=lon, lat=lat)
    )
    
    z = row.fetchone()
    return {"name": z.name, "category": z.category} if z else None
import datetime as dt, pandas as pd, pytz
from database import AsyncSessionLocal
from sqlalchemy import select, text
LAGOS = pytz.timezone("Africa/Lagos")
async def last_24h_report():
    async with AsyncSessionLocal() as db:
        since = dt.datetime.utcnow() - dt.timedelta(hours=24)
        sql = """
        SELECT v.id, v.device_id, t.name, v.detected_at AT TIME ZONE 'UTC' AT TIME ZONE 'Africa/Lagos' as detected_lagos,
        v.address, v.minutes_in_unapproved, v.resolution, v.comment
        FROM violations v
        JOIN trucks t ON t.device_id = v.device_id
        WHERE v.detected_at >= :since
        ORDER BY v.detected_at DESC
        """
        df = pd.read_sql(sql, db.bind, params={"since":since})
        df.to_csv("/reports/violations_last24h.csv", index=False)
        return df
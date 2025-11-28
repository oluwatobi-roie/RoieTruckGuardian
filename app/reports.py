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


from sqlalchemy import func



async def generate_yesterday_trip_report(report_date=None):
    """
    report_date: date object for the day to report (defaults to yesterday)
    Returns list of dicts per trip with start/end, duration, max_speed, avg_speed, start/end addresses replaced by zone name if in recognised zone.
    """
    if report_date is None:
        report_date = (dt.datetime.utcnow() - dt.timedelta(days=1)).date()

    start_dt = dt.datetime.combine(report_date, dt.time.min).replace(tzinfo=dt.timezone.utc)
    end_dt = dt.datetime.combine(report_date, dt.time.max).replace(tzinfo=dt.timezone.utc)

    async with AsyncSessionLocal() as db:
        # get trips that started or ended during the day
        q = await db.execute(
            select(Trip)
            .where(
                Trip.start_time >= start_dt,
                Trip.start_time <= end_dt
            )
            .order_by(Trip.start_time)
        )
        trips = q.scalars().all()
        report = []

        for trip in trips:
            # compute metrics from positions during trip
            pos_q = await db.execute(
                select(Position)
                .where(
                    Position.device_id == trip.device_id,
                    Position.server_time >= trip.start_time,
                    Position.server_time <= (trip.end_time if trip.end_time else end_dt)
                )
                .order_by(Position.server_time)
            )
            pos_list = pos_q.scalars().all()
            if not pos_list:
                continue

            speeds = [float(p.speed or 0) for p in pos_list]
            max_speed = max(speeds)
            avg_speed = sum(speeds) / len(speeds) if speeds else 0

            duration = None
            if trip.end_time:
                duration = (trip.end_time - trip.start_time).total_seconds()

            # replace start/end addresses if in recognised zone
            start_zone = await zone_at(db, pos_list[0].lat, pos_list[0].lon)
            end_zone = None
            if trip.end_time:
                end_zone = await zone_at(db, pos_list[-1].lat, pos_list[-1].lon)

            report.append({
                "trip_id": trip.id,
                "device_id": trip.device_id,
                "start_time": trip.start_time,
                "end_time": trip.end_time,
                "duration_s": duration,
                "max_speed": max_speed,
                "avg_speed": avg_speed,
                "start_address": start_zone.name if start_zone else trip.start_addr or pos_list[0].address,
                "end_address": end_zone.name if end_zone else trip.end_addr or pos_list[-1].address,
            })

    return report

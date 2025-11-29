from sqlalchemy import (BigInteger, Boolean, Column, Integer, Numeric, Text, DateTime, ForeignKey, Double)
from database import Base
from sqlalchemy.sql import func
from geoalchemy2 import Geography

class Truck(Base):
    __tablename__ = "trucks"
    id        = Column(Integer, primary_key=True, index=True)
    device_id = Column(BigInteger, unique=True, index=True, nullable=False)
    name      = Column(Text)
    phone     = Column(Text)
    contact   = Column(Text)

class Position(Base):
    __tablename__ = "positions"
    id          = Column(BigInteger, primary_key=True, index=True)
    device_id   = Column(BigInteger, index=True, nullable=False)
    lat         = Column(Numeric, nullable=False)
    lon         = Column(Numeric, nullable=False)
    speed       = Column(Numeric)
    ignition    = Column(Boolean)
    motion      = Column(Boolean)
    fix_time    = Column(DateTime(timezone=True), nullable=False)
    server_time = Column(DateTime(timezone=True), nullable=False)
    address     = Column(Text)
    odometer    = Column(Numeric)

class Trip(Base):
    __tablename__ = "trips"
    id            = Column(BigInteger, primary_key=True, index=True)
    device_id     = Column(BigInteger, index=True, nullable=False)
    start_time    = Column(DateTime(timezone=True), nullable=False)
    end_time      = Column(DateTime(timezone=True))
    start_zone    = Column(Text)
    end_zone      = Column(Text)
    start_addr    = Column(Text)
    end_addr      = Column(Text)
    start_odometer = Column(Numeric)
    end_odometer   = Column(Numeric)
    max_speed      = Column(Numeric)          # Maximum speed during the trip
    average_speed  = Column(Numeric)          # Average speed (computed at end)
    distance_m     = Column(Numeric, default=0)  # end - start odometer
    trip_duration  = Column(Numeric)             # seconds (computed at end)

class Stop(Base):
    __tablename__ = "stops"

    id          = Column(BigInteger, primary_key=True, index=True)
    device_id   = Column(BigInteger, index=True, nullable=False)
    stop_time   = Column(DateTime(timezone=True), nullable=False)
    start_zone  = Column(Text)
    stop_addr   = Column(Text)
    resume_time = Column(DateTime(timezone=True))
    zone        = Column(Text)
    duration_s  = Column(Integer)  # computed after resume
    

class Violation(Base):
    __tablename__ = "violations"
    id                    = Column(BigInteger, primary_key=True, index=True)
    device_id             = Column(BigInteger, index=True, nullable=False)
    detected_at           = Column(DateTime(timezone=True), nullable=False)
    lat                   = Column(Numeric)
    lon                   = Column(Numeric)
    address               = Column(Text)
    minutes_in_unapproved = Column(Integer)
    alert_sent            = Column(Boolean, default=False)
    resolution            = Column(Text)
    comment               = Column(Text)
    resolved_by           = Column(Text)
    resolved_at           = Column(DateTime(timezone=True))
    is_offline_violation = Column(Boolean, default=False)  # True if violation triggered by offline
    maintenance_mode     = Column(Boolean, default=False) # resolver can tick “Under maintenance”
    active =  Column(Boolean, default=False) 


class RecognisedZone(Base):
    __tablename__ = "recognised_zones"
    id         = Column(Integer, primary_key=True, index=True)
    name       = Column(Text, nullable=False)
    category   = Column(Text, nullable=False)   # CUSTOMER / RESTING / OFFICIAL_OFFICE
    geom       = Column(Geography('POLYGON', 4326), nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())


class DeviceStatus(Base):
    __tablename__ = "device_status"
    device_id   = Column(BigInteger, primary_key=True)
    online      = Column(Boolean, default=True)
    last_seen   = Column(DateTime(timezone=True))
    offline_since = Column(DateTime(timezone=True))   # NULL when online


class ViolationAttachment(Base):
    __tablename__ = "violation_attachments"
    id          = Column(BigInteger, primary_key=True)
    violation_id = Column(BigInteger, ForeignKey("violations.id"))
    file_path   = Column(Text)
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())
CREATE EXTENSION IF NOT EXISTS postgis;

CREATE TABLE IF NOT EXISTS recognised_zones (
    id          SERIAL PRIMARY KEY,
    name        TEXT NOT NULL,
    category    TEXT CHECK (category IN ('CUSTOMER','RESTING','OFFICIAL_OFFICE')),
    geom        GEOGRAPHY(POLYGON,4326) NOT NULL,
    created_at  TIMESTAMPTZ DEFAULT NOW()
);


CREATE TABLE IF NOT EXISTS trucks (
    id          SERIAL PRIMARY KEY,
    device_id   BIGINT UNIQUE NOT NULL,
    name        TEXT,
    phone       TEXT,
    contact     TEXT
);


CREATE TABLE IF NOT EXISTS positions (
    id          BIGSERIAL PRIMARY KEY,
    device_id   BIGINT NOT NULL,
    lat         DOUBLE PRECISION NOT NULL,
    lon         DOUBLE PRECISION NOT NULL,
    speed       DOUBLE PRECISION,
    ignition    BOOLEAN,
    motion      BOOLEAN,
    fix_time    TIMESTAMPTZ NOT NULL,
    server_time TIMESTAMPTZ NOT NULL,
    address     TEXT
);


CREATE TABLE IF NOT EXISTS trips (
    id          BIGSERIAL PRIMARY KEY,
    device_id   BIGINT NOT NULL,
    start_time  TIMESTAMPTZ NOT NULL,
    end_time    TIMESTAMPTZ,
    start_zone  TEXT,
    end_zone    TEXT,
    start_addr  TEXT,
    end_addr    TEXT,
    distance_m  DOUBLE PRECISION DEFAULT 0
);


CREATE TABLE IF NOT EXISTS violations (
    id          BIGSERIAL PRIMARY KEY,
    device_id   BIGINT NOT NULL,
    detected_at TIMESTAMPTZ NOT NULL,
    lat         DOUBLE PRECISION,
    lon         DOUBLE PRECISION,
    address     TEXT,
    minutes_in_unapproved INTEGER,
    alert_sent  BOOLEAN DEFAULT FALSE,
    resolution  TEXT,          -- 'IGNORE', 'UNDER_MAINTENANCE', 'TRUE_VIOLATION', ...
    comment     TEXT,
    resolved_by TEXT,
    resolved_at TIMESTAMPTZ,
    is_offline_violation BOOLEAN NOT NULL DEFAULT FALSE,
    maintenance_mode BOOLEAN NOT NULL DEFAULT FALSE,
    active BOOLEAN NOT NULL DEFAULT False

);


--  Spatial index
CREATE INDEX idx_positions_latlon ON positions USING GIST (ST_MakePoint(lon, lat));
CREATE INDEX idx_zones_geom ON recognised_zones USING GIST (geom);


CREATE TABLE IF NOT EXISTS device_status (
    device_id     BIGINT PRIMARY KEY,
    online        BOOLEAN DEFAULT TRUE,
    last_seen     TIMESTAMPTZ,
    offline_since TIMESTAMPTZ
);


-- INSERT INTO recognised_zones (name,category,geom) VALUES
-- ('Apapa Customer A','CUSTOMER', ST_GeogFromText('POLYGON((3.3 6.45, 3.31 6.45, 3.31 6.46, 3.3 6.46, 3.3 6.45))')),
-- ('Ikeja Resting','RESTING',  ST_GeogFromText('POLYGON((3.35 6.56, 3.36 6.56, 3.36 6.57, 3.35 6.57, 3.35 6.56))'));
-- -- Sample zones (Lagos). Replace with real polygons. INSERT INTO recognised_zones (name,category,geom) VALUES ('Apapa Customer A','CUSTOMER', ST_GeogFromText('POLYGON((3.3 6.45, 3.31 6.45, 3.31 6.46, 3.3 6.46, 3.3 6.45))')), ('Ikeja Resting','RESTING', ST_GeogFromText('POLYGON((3.35 6.56, 3.36 6.56, 3.36 6.57, 3.35 6.57, 3.35 6.56))'));

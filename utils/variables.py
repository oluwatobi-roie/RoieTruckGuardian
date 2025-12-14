'''
Define variables used across the entire applications
'''


TRIP_START_SPEED = 10.0    # km/h — conservative to avoid false starts
TRIP_START_COUNT = 10      # consecutive points to confirm start
TRIP_END_SPEED = 1.0       # km/h — to detect stop
TRIP_END_COUNT = 10        # consecutive points to confirm end

STATIONARY_SPEED = 1.0     # km/h threshold for "stationary" for parking violation
STATIONARY_COUNT = 10      # number of recent points that must be <= STATIONARY_SPEED
MIN_SUSTAINED_STOP_SECONDS = 300  # 5 minutes sustained stop to consider "parked"

PARKING_VIOLATION_THRESHOLD_MINUTES = 60  # minutes to trigger alert
RETRIGGER_COOLDOWN_MINUTES = 10  # don't create a new violation within this minutes of deactivation

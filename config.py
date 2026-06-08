# Basic settings for the drone EKF demo.
# The project can run on a real PX4 .ulg flight log if one is available.
# By default this version prefers real PX4 data and does not silently fall back to simulation.

N_STEPS = 50000
DT = 0.05
SEED = 7

# Use real public PX4 data when possible.
USE_PX4_LOGS_IF_AVAILABLE = True
AUTO_DOWNLOAD_PX4_LOG = True
PX4_DOWNLOAD_MAX_LOGS = 5
PX4_DOWNLOAD_FOLDER = "data/downloaded"
PX4_MAX_SAMPLES = 20000
PX4_TARGET_DT = 0.05

# Sensor noise values. Units are roughly meters, m/s^2, and meters.
GPS_STD = 0.45
IMU_STD = 0.18
BARO_STD = 5.0

# These are only used by the simulated fallback dataset.
GPS_OUTLIER_PROB = 0.035
GPS_OUTLIER_SCALE = 28.0

# Mahalanobis-distance threshold for rejecting bad GPS points.
# Larger = less strict. Smaller = more strict.
GPS_GATE_THRESHOLD = 60.0

# Monte Carlo runs for the small repeatability test at the end.
MONTE_CARLO_RUNS = 5000

# Set True only when you want a quick demo without real logs.
ALLOW_SIMULATION_FALLBACK = False

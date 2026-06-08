# Basic settings for the drone EKF demo.
# Change these numbers if you want to test a noisier sensor setup.

N_STEPS = 50000
DT = 0.05
SEED = 11

# Sensor noise values. Units are roughly meters, m/s^2, and meters.
GPS_STD = 2.5
IMU_STD = 0.18
BARO_STD = 0.8

# A few GPS samples are intentionally corrupted so the filter has to reject them.
GPS_OUTLIER_PROB = 0.035
GPS_OUTLIER_SCALE = 28.0

# Mahalanobis-distance threshold for rejecting bad GPS points.
# Larger = less strict. Smaller = more strict.
GPS_GATE_THRESHOLD = 4.0

# Monte Carlo runs for the small repeatability test at the end.
MONTE_CARLO_RUNS = 10000

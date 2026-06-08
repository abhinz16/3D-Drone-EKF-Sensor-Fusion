import numpy as np


def true_acceleration(t):
    """Small synthetic flight pattern.

    The motion is not meant to copy a real flight controller. I just wanted a
    smooth path with turns, climbs, and descents so the EKF has something more
    interesting than a straight line to estimate.
    """
    ax = 0.45 * np.sin(0.45 * t) + 0.18 * np.cos(0.10 * t)
    ay = 0.35 * np.cos(0.38 * t) - 0.10 * np.sin(0.18 * t)
    az = 0.16 * np.sin(0.30 * t) - 0.04
    return np.array([ax, ay, az], dtype=float)


def make_dataset(n_steps, dt, seed, gps_std, imu_std, baro_std,
                 gps_outlier_prob, gps_outlier_scale):
    rng = np.random.default_rng(seed)

    # State is [x, y, z, vx, vy, vz].
    truth = np.zeros((n_steps, 6), dtype=float)
    imu = np.zeros((n_steps, 3), dtype=float)
    gps = np.zeros((n_steps, 3), dtype=float)
    barometer = np.zeros(n_steps, dtype=float)
    gps_is_outlier = np.zeros(n_steps, dtype=bool)

    truth[0, :] = np.array([0.0, 0.0, 4.0, 3.0, 0.7, 0.25])

    for k in range(1, n_steps):
        t = k * dt
        a = true_acceleration(t)

        truth[k, 0:3] = truth[k - 1, 0:3] + truth[k - 1, 3:6] * dt + 0.5 * a * dt * dt
        truth[k, 3:6] = truth[k - 1, 3:6] + a * dt

        # Keep altitude positive. This makes plots easier to read.
        if truth[k, 2] < 1.0:
            truth[k, 2] = 1.0
            truth[k, 5] = abs(truth[k, 5]) * 0.4

        imu[k, :] = a + rng.normal(0.0, imu_std, 3)
        gps[k, :] = truth[k, 0:3] + rng.normal(0.0, gps_std, 3)
        barometer[k] = truth[k, 2] + rng.normal(0.0, baro_std)

        if rng.random() < gps_outlier_prob:
            gps_is_outlier[k] = True
            gps[k, :] += rng.normal(0.0, gps_outlier_scale, 3)

    # Give the first sample sensible sensor readings too.
    imu[0, :] = true_acceleration(0.0) + rng.normal(0.0, imu_std, 3)
    gps[0, :] = truth[0, 0:3] + rng.normal(0.0, gps_std, 3)
    barometer[0] = truth[0, 2] + rng.normal(0.0, baro_std)

    return {
        "time": np.arange(n_steps) * dt,
        "truth": truth,
        "imu": imu,
        "gps": gps,
        "barometer": barometer,
        "gps_is_outlier": gps_is_outlier,
    }

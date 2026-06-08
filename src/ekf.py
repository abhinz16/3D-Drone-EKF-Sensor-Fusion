import numpy as np
from tqdm import tqdm


class DroneEKF:
    """Small 3D position/velocity Kalman filter used for this project.

    State vector:
        [x, y, z, vx, vy, vz]

    The filter can use acceleration during prediction for the simulated data.
    For PX4 logs, acceleration is usually in the body frame and needs attitude
    compensation before it can be used as world-frame acceleration. To avoid a
    misleading estimate, the real-log path uses a constant-velocity prediction
    and lets GPS carry the position correction.
    """

    def __init__(self, dt, gps_std, baro_std, gps_gate_threshold, warmup_gps_updates=20):
        self.dt = float(dt)
        self.gps_gate_threshold = float(gps_gate_threshold)
        self.warmup_gps_updates = int(warmup_gps_updates)
        self.gps_update_count = 0

        self.x = np.zeros(6, dtype=float)

        # Start fairly uncertain so the first valid GPS points can pull the
        # filter into the correct local frame.
        self.P = np.diag([25.0, 25.0, 25.0, 8.0, 8.0, 8.0])

        self.H_gps = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.R_gps = np.eye(3) * float(gps_std) ** 2

        self.H_baro = np.array([[0, 0, 1, 0, 0, 0]], dtype=float)
        self.R_baro = np.array([[float(baro_std) ** 2]], dtype=float)

    def initialize(self, first_gps, second_gps=None, dt=None):
        self.x[0:3] = np.asarray(first_gps, dtype=float)
        self.x[3:6] = 0.0

        if second_gps is not None and dt is not None and dt > 0:
            v0 = (np.asarray(second_gps, dtype=float) - np.asarray(first_gps, dtype=float)) / dt
            # This avoids a wild initial velocity if the first GPS interval is bad.
            self.x[3:6] = np.clip(v0, -8.0, 8.0)

    def predict(self, accel=None, use_accel=False, q_pos=0.03, q_vel=0.35):
        dt = self.dt
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        self.x = F @ self.x

        if use_accel and accel is not None:
            B = np.array([
                [0.5 * dt * dt, 0, 0],
                [0, 0.5 * dt * dt, 0],
                [0, 0, 0.5 * dt * dt],
                [dt, 0, 0],
                [0, dt, 0],
                [0, 0, dt],
            ], dtype=float)
            self.x = self.x + B @ np.asarray(accel, dtype=float)

        self.P = F @ self.P @ F.T + Q

    def update(self, z, H, R):
        z = np.atleast_1d(z).astype(float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y

        I = np.eye(self.P.shape[0])
        KH = K @ H
        self.P = (I - KH) @ self.P @ (I - KH).T + K @ R @ K.T

    def update_barometer(self, z_altitude):
        self.update(np.array([z_altitude]), self.H_baro, self.R_baro)

    def update_gps_with_gate(self, z_gps):
        z_gps = np.asarray(z_gps, dtype=float)
        residual = z_gps - self.H_gps @ self.x
        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps
        score = float(residual.T @ np.linalg.inv(S) @ residual)

        if self.gps_update_count < self.warmup_gps_updates:
            self.update(z_gps, self.H_gps, self.R_gps)
            self.gps_update_count += 1
            return True, score

        if score > self.gps_gate_threshold:
            return False, score

        self.update(z_gps, self.H_gps, self.R_gps)
        self.gps_update_count += 1
        return True, score


def _is_real_px4_data(data):
    return str(data.get("source", "")).lower().startswith("px4") or data.get("source_file")


def run_filter(data, dt, gps_std, baro_std, gps_gate_threshold, show_progress=True):
    real_px4 = _is_real_px4_data(data)

    # Real PX4 logs already contain GPS/local-position information. IMU samples
    # are usually body-frame acceleration, so using them directly as world-frame
    # acceleration makes the estimate drift. Barometer height can also be in a
    # different relative frame from the local-position reference. For that reason
    # the real-log mode uses GPS as the main absolute measurement and keeps baro
    # out of the main EKF update. The barometer is still plotted for comparison.
    use_accel = not real_px4
    use_barometer = not real_px4

    if real_px4:
        gps_std = min(float(gps_std), 0.45)
        baro_std = max(float(baro_std), 5.0)
        q_pos = 0.02
        q_vel = 0.60
    else:
        q_pos = 0.08
        q_vel = 0.50

    ekf = DroneEKF(dt, gps_std, baro_std, gps_gate_threshold)
    second_gps = data["gps"][1] if len(data["gps"]) > 1 else None
    ekf.initialize(data["gps"][0], second_gps=second_gps, dt=dt)

    n_steps = len(data["time"])
    estimates = np.zeros((n_steps, 6), dtype=float)
    gps_used = np.zeros(n_steps, dtype=bool)
    gate_scores = np.zeros(n_steps, dtype=float)

    step_iter = range(n_steps)
    if show_progress:
        step_iter = tqdm(step_iter, desc="EKF update", unit="step")

    for k in step_iter:
        ekf.predict(data["imu"][k], use_accel=use_accel, q_pos=q_pos, q_vel=q_vel)
        accepted, score = ekf.update_gps_with_gate(data["gps"][k])

        if use_barometer:
            ekf.update_barometer(data["barometer"][k])

        estimates[k, :] = ekf.x
        gps_used[k] = accepted
        gate_scores[k] = score

    return estimates, gps_used, gate_scores

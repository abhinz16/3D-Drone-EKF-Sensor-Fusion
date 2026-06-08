import numpy as np
from tqdm import tqdm


class DroneEKF:
    """Constant-acceleration EKF for a small 3D drone example.

    The IMU acceleration is used during prediction. GPS updates x/y/z, and the
    barometer gives another measurement of z. The model is linear here, so this
    is technically close to a Kalman Filter, but the structure is the same one I
    would use before swapping in a nonlinear drone model.
    """

    def __init__(self, dt, gps_std, baro_std, gps_gate_threshold):
        self.dt = float(dt)
        self.gps_gate_threshold = float(gps_gate_threshold)

        self.x = np.zeros(6, dtype=float)
        self.P = np.eye(6) * 8.0

        q_pos = 0.08
        q_vel = 0.35
        self.Q = np.diag([q_pos, q_pos, q_pos, q_vel, q_vel, q_vel])

        self.H_gps = np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
        ], dtype=float)
        self.R_gps = np.eye(3) * gps_std ** 2

        self.H_baro = np.array([[0, 0, 1, 0, 0, 0]], dtype=float)
        self.R_baro = np.array([[baro_std ** 2]], dtype=float)

    def initialize(self, first_gps, first_baro):
        self.x[0:3] = first_gps
        self.x[2] = 0.5 * (first_gps[2] + first_baro)
        self.x[3:6] = 0.0

    def predict(self, accel):
        dt = self.dt
        F = np.array([
            [1, 0, 0, dt, 0, 0],
            [0, 1, 0, 0, dt, 0],
            [0, 0, 1, 0, 0, dt],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
        ], dtype=float)

        B = np.array([
            [0.5 * dt * dt, 0, 0],
            [0, 0.5 * dt * dt, 0],
            [0, 0, 0.5 * dt * dt],
            [dt, 0, 0],
            [0, dt, 0],
            [0, 0, dt],
        ], dtype=float)

        self.x = F @ self.x + B @ accel
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z, H, R):
        z = np.atleast_1d(z).astype(float)
        y = z - H @ self.x
        S = H @ self.P @ H.T + R
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        I = np.eye(self.P.shape[0])
        self.P = (I - K @ H) @ self.P

    def update_barometer(self, z_altitude):
        self.update(np.array([z_altitude]), self.H_baro, self.R_baro)

    def update_gps_with_gate(self, z_gps):
        z_gps = np.asarray(z_gps, dtype=float)
        residual = z_gps - self.H_gps @ self.x
        S = self.H_gps @ self.P @ self.H_gps.T + self.R_gps
        score = float(residual.T @ np.linalg.inv(S) @ residual)

        if score > self.gps_gate_threshold:
            return False, score

        self.update(z_gps, self.H_gps, self.R_gps)
        return True, score


def run_filter(data, dt, gps_std, baro_std, gps_gate_threshold, show_progress=True):
    ekf = DroneEKF(dt, gps_std, baro_std, gps_gate_threshold)
    ekf.initialize(data["gps"][0], data["barometer"][0])

    n_steps = len(data["time"])
    estimates = np.zeros((n_steps, 6), dtype=float)
    gps_used = np.zeros(n_steps, dtype=bool)
    gate_scores = np.zeros(n_steps, dtype=float)

    step_iter = range(n_steps)
    if show_progress:
        step_iter = tqdm(step_iter, desc="EKF update", unit="step")

    for k in step_iter:
        ekf.predict(data["imu"][k])
        ekf.update_barometer(data["barometer"][k])
        accepted, score = ekf.update_gps_with_gate(data["gps"][k])

        estimates[k, :] = ekf.x
        gps_used[k] = accepted
        gate_scores[k] = score

    return estimates, gps_used, gate_scores

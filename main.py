"""Run the full 3D drone EKF project from one file.

This is the only file you need to run in Spyder. It creates the simulated data,
runs the EKF, saves plots and CSV files, and finishes with a small Monte Carlo
check using multiple CPU cores.
"""

import os
import csv
import numpy as np
from tqdm import tqdm

from config import (
    N_STEPS, DT, SEED, GPS_STD, IMU_STD, BARO_STD,
    GPS_OUTLIER_PROB, GPS_OUTLIER_SCALE, GPS_GATE_THRESHOLD,
    MONTE_CARLO_RUNS,
)
from src.simulate_drone import make_dataset
from src.ekf import run_filter
from src.metrics import position_rmse, gps_rmse, axis_rmse
from src.plots import (
    ensure_dir, plot_3d,
    plot_position_errors, plot_altitude,
)
from src.monte_carlo import run_monte_carlo


def save_main_csv(path, data, estimate, gps_used, gate_scores):
    headers = [
        "time", "true_x", "true_y", "true_z", "true_vx", "true_vy", "true_vz",
        "gps_x", "gps_y", "gps_z", "barometer_z",
        "ekf_x", "ekf_y", "ekf_z", "ekf_vx", "ekf_vy", "ekf_vz",
        "gps_used", "gps_gate_score", "gps_was_injected_outlier",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        rows = range(len(data["time"]))
        for k in tqdm(rows, desc="Saving flight CSV", unit="row"):
            writer.writerow([
                data["time"][k],
                *data["truth"][k].tolist(),
                *data["gps"][k].tolist(),
                data["barometer"][k],
                *estimate[k].tolist(),
                int(gps_used[k]),
                gate_scores[k],
                int(data["gps_is_outlier"][k]),
            ])


def save_monte_carlo_csv(path, results):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["seed", "gps_rmse", "ekf_rmse", "rejected_gps"])
        writer.writeheader()
        writer.writerows(results)


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    out_dir = os.path.join(base_dir, "outputs")
    fig_dir = os.path.join(base_dir, "figures")
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    print("Starting 3D drone EKF demo...")
    print(f"Samples: {N_STEPS}, dt: {DT} s")

    print("Generating simulated GPS, IMU, and barometer data...")
    data = make_dataset(
        n_steps=N_STEPS,
        dt=DT,
        seed=SEED,
        gps_std=GPS_STD,
        imu_std=IMU_STD,
        baro_std=BARO_STD,
        gps_outlier_prob=GPS_OUTLIER_PROB,
        gps_outlier_scale=GPS_OUTLIER_SCALE,
    )

    estimate, gps_used, gate_scores = run_filter(
        data=data,
        dt=DT,
        gps_std=GPS_STD,
        baro_std=BARO_STD,
        gps_gate_threshold=GPS_GATE_THRESHOLD,
        show_progress=True,
    )

    raw_gps_rmse = gps_rmse(data["truth"], data["gps"])
    ekf_rmse = position_rmse(data["truth"], estimate)
    per_axis = axis_rmse(data["truth"], estimate)

    print("\nMain run results")
    print("----------------")
    print(f"Raw GPS RMSE: {raw_gps_rmse:.3f} m")
    print(f"EKF RMSE:     {ekf_rmse:.3f} m")
    print(f"EKF RMSE by axis: x={per_axis['x']:.3f}, y={per_axis['y']:.3f}, z={per_axis['z']:.3f} m")
    print(f"Injected GPS outliers: {int(data['gps_is_outlier'].sum())}")
    print(f"GPS measurements rejected by gate: {int((~gps_used).sum())}")

    save_main_csv(os.path.join(out_dir, "flight_estimation_results.csv"), data, estimate, gps_used, gate_scores)

    plot_jobs = [
        ("3D trajectory", lambda: plot_3d(data["time"], data["truth"], data["gps"], estimate, gps_used,
                                          os.path.join(fig_dir, "trajectory_3d.png"))),
        ("position error", lambda: plot_position_errors(data["time"], data["truth"], data["gps"], estimate,
                                                        os.path.join(fig_dir, "position_error.png"))),
        ("altitude plot", lambda: plot_altitude(data["time"], data["truth"], data["barometer"], estimate,
                                                os.path.join(fig_dir, "altitude_estimate.png"))),
    ]
    for name, job in tqdm(plot_jobs, desc="Saving plots", unit="plot"):
        job()

    print("\nRunning small Monte Carlo check on CPU cores...")
    mc_results, workers = run_monte_carlo(MONTE_CARLO_RUNS)
    save_monte_carlo_csv(os.path.join(out_dir, "monte_carlo_results.csv"), mc_results)

    avg_gps = float(np.mean([r["gps_rmse"] for r in mc_results]))
    avg_ekf = float(np.mean([r["ekf_rmse"] for r in mc_results]))
    print(f"Monte Carlo runs: {MONTE_CARLO_RUNS} using {workers} worker process(es)")
    print(f"Average GPS RMSE: {avg_gps:.3f} m")
    print(f"Average EKF RMSE: {avg_ekf:.3f} m")

    print("\nSaved files:")
    print(" - figures/trajectory_3d.png")
    print(" - figures/position_error.png")
    print(" - figures/altitude_estimate.png")
    print(" - outputs/flight_estimation_results.csv")
    print(" - outputs/monte_carlo_results.csv")
    print("\nDone.")


if __name__ == "__main__":
    main()

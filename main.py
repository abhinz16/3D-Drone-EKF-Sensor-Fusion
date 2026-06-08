"""Run the full 3D drone EKF project from one file.

This is the file I run in Spyder. It first looks for a real PX4 .ulg flight log
in data/ or data/downloaded/. If it finds one, it uses real flight data. If not,
it tries to download a small batch of public PX4 logs and uses the first one
that contains readable GPS/global-position data. It does not silently switch to
simulation unless ALLOW_SIMULATION_FALLBACK is set to True in config.py.
"""

import csv
import glob
import os

import numpy as np
from tqdm import tqdm

from config import (
    N_STEPS, DT, SEED, GPS_STD, IMU_STD, BARO_STD,
    GPS_OUTLIER_PROB, GPS_OUTLIER_SCALE, GPS_GATE_THRESHOLD,
    MONTE_CARLO_RUNS, USE_PX4_LOGS_IF_AVAILABLE, AUTO_DOWNLOAD_PX4_LOG,
    PX4_DOWNLOAD_FOLDER, PX4_DOWNLOAD_MAX_LOGS, PX4_MAX_SAMPLES, PX4_TARGET_DT,
    ALLOW_SIMULATION_FALLBACK,
)
from src.simulate_drone import make_dataset
from src.ekf import run_filter
from src.metrics import position_rmse, gps_rmse, axis_rmse
from src.plots import ensure_dir, plot_3d, plot_position_errors, plot_altitude
from src.monte_carlo import run_monte_carlo
from src.px4_downloader import download_public_logs
from src.px4_reader import PX4LogError, load_px4_dataset, list_topics


def find_px4_logs(base_dir):
    patterns = [
        os.path.join(base_dir, "data", "*.ulg"),
        os.path.join(base_dir, "data", "downloaded", "*.ulg"),
    ]
    logs = []
    for pattern in patterns:
        logs.extend(glob.glob(pattern))
    return sorted(set(logs))


def get_dataset(base_dir):
    data_dir = os.path.join(base_dir, "data")
    download_dir = os.path.join(base_dir, PX4_DOWNLOAD_FOLDER)
    ensure_dir(data_dir)
    ensure_dir(download_dir)

    if USE_PX4_LOGS_IF_AVAILABLE:
        logs = find_px4_logs(base_dir)

        if not logs and AUTO_DOWNLOAD_PX4_LOG:
            try:
                print("No local PX4 log found. Trying to download public PX4 logs...")
                download_public_logs(
                    download_folder=download_dir,
                    max_num=PX4_DOWNLOAD_MAX_LOGS,
                    mav_type=("Quadrotor",),
                    rating=("Good",),
                )
                logs = find_px4_logs(base_dir)
            except Exception as exc:
                print(f"PX4 download skipped: {exc}")

        if logs:
            for log_path in logs:
                try:
                    data = load_px4_dataset(
                        log_path,
                        target_dt=PX4_TARGET_DT,
                        max_samples=PX4_MAX_SAMPLES,
                    )
                    print("Using real PX4 flight log data.")
                    print(f"Reference source: {data.get('reference_source', 'unknown')}")
                    print(f"IMU source:       {data.get('imu_source', 'unknown')}")
                    print(f"GPS source:       {data.get('gps_source', 'unknown')}")
                    print(f"Barometer source: {data.get('barometer_source', 'unknown')}")
                    return data
                except PX4LogError as exc:
                    print(f"Could not use {os.path.basename(log_path)}: {exc}")
                    try:
                        topics = list_topics(log_path)
                        preview = ", ".join(topics[:18])
                        print(f"  Topic preview: {preview}")
                    except Exception:
                        pass
                except Exception as exc:
                    print(f"Unexpected PX4 read error for {os.path.basename(log_path)}: {exc}")

    if not ALLOW_SIMULATION_FALLBACK:
        raise RuntimeError(
            "No usable PX4 log was found. The program did not switch to simulation.\n"
            "Try running: python download_px4_logs.py --max-num 10 --mav-type Quadrotor --rating Good\n"
            "Then run main.py again. You can also place any .ulg file in data/ manually."
        )

    print("Using simulated drone data instead because ALLOW_SIMULATION_FALLBACK=True.")
    return make_dataset(
        n_steps=N_STEPS,
        dt=DT,
        seed=SEED,
        gps_std=GPS_STD,
        imu_std=IMU_STD,
        baro_std=BARO_STD,
        gps_outlier_prob=GPS_OUTLIER_PROB,
        gps_outlier_scale=GPS_OUTLIER_SCALE,
    ) | {"source": "simulation"}


def save_main_csv(path, data, estimate, gps_used, gate_scores):
    headers = [
        "time", "ref_x", "ref_y", "ref_z", "ref_vx", "ref_vy", "ref_vz",
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
                int(data.get("gps_is_outlier", np.zeros(len(data["time"]), dtype=bool))[k]),
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

    print("Starting 3D drone EKF sensor-fusion project...")
    data = get_dataset(base_dir)

    dt = float(np.median(np.diff(data["time"]))) if len(data["time"]) > 1 else DT
    print(f"Data source: {data.get('source', 'unknown')}")
    if data.get("source_file"):
        print(f"Source file: {data['source_file']}")
    print(f"Samples: {len(data['time'])}, estimated dt: {dt:.4f} s")

    estimate, gps_used, gate_scores = run_filter(
        data=data,
        dt=dt,
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
    print(f"Raw GPS RMSE vs reference: {raw_gps_rmse:.3f} m")
    print(f"EKF RMSE vs reference:     {ekf_rmse:.3f} m")
    print(f"EKF RMSE by axis: x={per_axis['x']:.3f}, y={per_axis['y']:.3f}, z={per_axis['z']:.3f} m")
    if data.get("source") == "simulation":
        print(f"Injected GPS outliers: {int(data['gps_is_outlier'].sum())}")
    print(f"GPS measurements rejected by gate: {int((~gps_used).sum())}")

    save_main_csv(os.path.join(out_dir, "flight_estimation_results.csv"), data, estimate, gps_used, gate_scores)

    # The graphical abstract is a static PNG stored in figures/. The code no
    # longer regenerates it, so GitHub always shows the same polished image.
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

    print("\nRunning small Monte Carlo check on simulated data using CPU cores...")
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

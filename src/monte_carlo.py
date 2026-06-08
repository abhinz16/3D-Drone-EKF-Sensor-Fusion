from concurrent.futures import ProcessPoolExecutor
import os
from tqdm import tqdm

from config import N_STEPS, DT, GPS_STD, IMU_STD, BARO_STD, GPS_OUTLIER_PROB, GPS_OUTLIER_SCALE, GPS_GATE_THRESHOLD
from src.simulate_drone import make_dataset
from src.ekf import run_filter
from src.metrics import position_rmse, gps_rmse


def _single_run(seed):
    data = make_dataset(
        n_steps=N_STEPS,
        dt=DT,
        seed=seed,
        gps_std=GPS_STD,
        imu_std=IMU_STD,
        baro_std=BARO_STD,
        gps_outlier_prob=GPS_OUTLIER_PROB,
        gps_outlier_scale=GPS_OUTLIER_SCALE,
    )
    estimate, gps_used, _ = run_filter(data, DT, GPS_STD, BARO_STD, GPS_GATE_THRESHOLD, show_progress=False)
    return {
        "seed": seed,
        "gps_rmse": gps_rmse(data["truth"], data["gps"]),
        "ekf_rmse": position_rmse(data["truth"], estimate),
        "rejected_gps": int((~gps_used).sum()),
    }


def run_monte_carlo(n_runs):
    # Use most cores, but leave the machine usable.
    cpu_count = os.cpu_count() or 1
    workers = max(1, min(cpu_count - 1, n_runs))

    seeds = list(range(100, 100 + n_runs))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        mapped = pool.map(_single_run, seeds)
        results = list(tqdm(mapped, total=n_runs, desc="Monte Carlo", unit="run"))

    return results, workers

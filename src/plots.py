import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def plot_3d(time, truth, gps, estimate, gps_used, save_path):
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(truth[:, 0], truth[:, 1], truth[:, 2], label="true path")
    ax.plot(estimate[:, 0], estimate[:, 1], estimate[:, 2], label="EKF estimate")

    # Plot every few GPS points to keep the image readable.
    idx = np.arange(0, len(time), 8)
    ax.scatter(gps[idx, 0], gps[idx, 1], gps[idx, 2], s=9, label="GPS samples")

    rejected = ~gps_used
    if np.any(rejected):
        ax.scatter(gps[rejected, 0], gps[rejected, 1], gps[rejected, 2],
                   marker="x", s=35, label="rejected GPS")

    ax.set_xlabel("x position (m)")
    ax.set_ylabel("y position (m)")
    ax.set_zlabel("altitude (m)")
    ax.set_title("3D drone trajectory estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_position_errors(time, truth, gps, estimate, save_path):
    err_ekf = estimate[:, 0:3] - truth[:, 0:3]
    err_gps = gps[:, 0:3] - truth[:, 0:3]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(time, np.linalg.norm(err_gps, axis=1), label="raw GPS error")
    ax.plot(time, np.linalg.norm(err_ekf, axis=1), label="EKF error")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("3D position error (m)")
    ax.set_title("Position error comparison")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)


def plot_altitude(time, truth, barometer, estimate, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(time, truth[:, 2], label="true altitude")
    ax.plot(time, estimate[:, 2], label="EKF altitude")
    ax.plot(time, barometer, linewidth=0.9, label="barometer")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("altitude (m)")
    ax.set_title("Altitude estimation")
    ax.grid(True, alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=160)
    plt.close(fig)

import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_graphical_abstract(save_path):
    """Draw the README diagram.

    I kept this as code instead of a separate design file so the image can be
    regenerated whenever the project is run. The style is intentionally a bit
    more futuristic than the plots used for the actual results.
    """
    fig, ax = plt.subplots(figsize=(13.5, 7.2))
    fig.patch.set_facecolor("#050712")
    ax.set_facecolor("#050712")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Soft background grid.
    for x in np.linspace(0.03, 0.97, 18):
        ax.plot([x, x], [0.06, 0.94], color="#18324a", alpha=0.22, lw=0.8)
    for y in np.linspace(0.08, 0.92, 11):
        ax.plot([0.03, 0.97], [y, y], color="#18324a", alpha=0.22, lw=0.8)

    # A curved flight path to make the abstract feel like a robotics project,
    # not just a flowchart.
    t = np.linspace(0, 1, 350)
    x_path = 0.08 + 0.84 * t
    y_path = 0.28 + 0.17 * np.sin(2.2 * np.pi * t) + 0.26 * t
    ax.plot(x_path, y_path, color="#00f5ff", lw=4.0, alpha=0.30)
    ax.plot(x_path, y_path, color="#62ff8e", lw=1.6, alpha=0.95)

    # Noisy GPS samples and a few obvious outliers.
    rng = np.random.default_rng(7)
    sample_idx = np.linspace(20, 330, 22).astype(int)
    gps_x = x_path[sample_idx] + rng.normal(0, 0.015, len(sample_idx))
    gps_y = y_path[sample_idx] + rng.normal(0, 0.025, len(sample_idx))
    ax.scatter(gps_x, gps_y, s=34, color="#ffcf5a", edgecolor="#fff5cf", lw=0.6,
               alpha=0.95, label="GPS")
    out_x = np.array([0.34, 0.64, 0.79])
    out_y = np.array([0.80, 0.22, 0.74])
    ax.scatter(out_x, out_y, s=80, marker="x", color="#ff3b8d", lw=2.4,
               label="rejected GPS")

    # Simple drone icon.
    drone_x, drone_y = 0.14, 0.67
    ax.scatter([drone_x], [drone_y], s=190, color="#e9f7ff", edgecolor="#00f5ff", lw=1.8, zorder=5)
    arm_color = "#b7f7ff"
    rotor_color = "#8b5cff"
    for dx, dy in [(0.055, 0.035), (-0.055, 0.035), (0.055, -0.035), (-0.055, -0.035)]:
        ax.plot([drone_x, drone_x + dx], [drone_y, drone_y + dy], color=arm_color, lw=2.0, zorder=4)
        circ = plt.Circle((drone_x + dx, drone_y + dy), 0.022, fc="none", ec=rotor_color, lw=2.0, alpha=0.95)
        ax.add_patch(circ)

    def neon_box(cx, cy, w, h, text, edge, fill="#0b1023"):
        # Draw a small glow behind the main box.
        glow = plt.Rectangle((cx - w/2 - 0.006, cy - h/2 - 0.006), w + 0.012, h + 0.012,
                             fc=edge, ec="none", alpha=0.10, zorder=1)
        ax.add_patch(glow)
        box = plt.Rectangle((cx - w/2, cy - h/2), w, h, fc=fill, ec=edge, lw=2.0, zorder=2)
        ax.add_patch(box)
        ax.text(cx, cy, text, ha="center", va="center", color="#f2fbff",
                fontsize=12.5, fontweight="bold", zorder=3)

    neon_box(0.23, 0.40, 0.18, 0.12, "IMU\nacceleration", "#00f5ff")
    neon_box(0.23, 0.20, 0.18, 0.12, "GPS +\nbarometer", "#ffcf5a")
    neon_box(0.50, 0.31, 0.20, 0.15, "3D EKF\nsensor fusion", "#8b5cff")
    neon_box(0.78, 0.34, 0.20, 0.13, "clean 3D\ntrajectory", "#62ff8e")

    arrow_style = dict(arrowstyle="-|>", lw=2.3, color="#e9f7ff", shrinkA=6, shrinkB=6)
    ax.annotate("", xy=(0.41, 0.34), xytext=(0.32, 0.40), arrowprops=arrow_style)
    ax.annotate("", xy=(0.41, 0.28), xytext=(0.32, 0.20), arrowprops=arrow_style)
    ax.annotate("", xy=(0.67, 0.33), xytext=(0.60, 0.31), arrowprops=arrow_style)

    ax.text(0.50, 0.88, "3D Drone State Estimation with EKF Sensor Fusion",
            ha="center", va="center", color="#f8fbff", fontsize=22, fontweight="bold")
    ax.text(0.50, 0.815, "noisy GPS + IMU + barometer  →  outlier rejection  →  smoother position estimate",
            ha="center", va="center", color="#a8d8ff", fontsize=12.5)

    # Bottom metric strip.
    ax.text(0.50, 0.075, "Outputs: 3D trajectory plot  •  RMSE comparison  •  altitude estimate  •  Monte Carlo results",
            ha="center", va="center", color="#d9f8ff", fontsize=11.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#071a2d", ec="#00f5ff", lw=1.2, alpha=0.9))

    fig.savefig(save_path, dpi=180, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

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

# 3D Drone EKF Sensor Fusion

![Graphical abstract](figures/graphical_abstract.png)

This project estimates a drone's 3D position using an Extended Kalman Filter. It can run in two ways:

1. Use a real public PX4 `.ulg` flight log if one is available.
2. Fall back to a simulated drone flight if no real log can be downloaded or parsed.

The goal is to keep the project easy to run while still showing realistic robotics state-estimation work.

## What the project does

- Reads GPS, IMU, and barometer-style measurements
- Runs a 3D EKF with GPS outlier rejection
- Compares raw GPS against the EKF estimate
- Saves trajectory plots, error plots, and CSV results
- Runs a small Monte Carlo test using multiple CPU cores
- Uses `tqdm` progress bars for longer steps

## Folder layout

```text
3d_drone_ekf_sensor_fusion/
├── main.py
├── config.py
├── download_px4_logs.py
├── requirements.txt
├── README.md
├── data/
│   └── downloaded/
├── figures/
│   └── graphical_abstract.png
├── outputs/
└── src/
    ├── ekf.py
    ├── metrics.py
    ├── monte_carlo.py
    ├── plots.py
    ├── px4_downloader.py
    ├── px4_reader.py
    └── simulate_drone.py
```

## Quick start

Install the packages:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

You can also open `main.py` in Spyder and run it from there.

## Real PX4 data

The project looks for PX4 `.ulg` files in:

```text
data/
data/downloaded/
```

If no `.ulg` file is found, `main.py` tries to download one public PX4 quadrotor log from PX4 Flight Review. It only downloads one log by default because the Flight Review server is a shared public service.

To manually download more logs:

```bash
python download_px4_logs.py --max-num 5 --rating Good --mav-type Quadrotor
```

To only print matching entries:

```bash
python download_px4_logs.py --print
```

To place your own log manually, download a `.ulg` file and put it here:

```text
data/flight.ulg
```

Then run:

```bash
python main.py
```

## Outputs

After a run, the project saves:

```text
figures/trajectory_3d.png
figures/position_error.png
figures/altitude_estimate.png
outputs/flight_estimation_results.csv
outputs/monte_carlo_results.csv
```

## Notes on the real PX4 log mode

PX4 logs are not always identical. Some logs have GPS, IMU, barometer, and local-position topics. Some do not. The reader is written to be flexible, but if a log is missing important topics, the code will skip it and try another log or fall back to simulation.

For the real-log mode, `vehicle_local_position` is used as the reference trajectory when it exists. That is not perfect ground truth, but it is useful for comparing the custom EKF output against a PX4 estimator-style reference.

The IMU handling is intentionally kept simple for this project. A full inertial navigation system would rotate body-frame acceleration into the world frame using attitude estimates. This project focuses on the EKF pipeline, data handling, outlier rejection, and result visualization.

## Resume bullet

Built a 3D drone state-estimation project using an Extended Kalman Filter to fuse GPS, IMU, and barometer data from simulated and public PX4 flight logs, with GPS outlier rejection, RMSE evaluation, Monte Carlo testing, and trajectory visualization.


## PX4 real logs

`main.py` looks for `.ulg` files in `data/` and `data/downloaded/`. If none are found, it downloads a small number of public PX4 logs from Flight Review using the same public endpoints used by the official downloader script. Some public logs do not contain GPS/global-position topics, so the code tries multiple logs and uses the first readable one.

To manually download more logs:

```bash
python download_px4_logs.py --max-num 10 --mav-type Quadrotor --rating Good
```

Then run:

```bash
python main.py
```

The reader checks common PX4 topic names including `vehicle_gps_position`, `sensor_gps`, `vehicle_global_position`, and `estimator_global_position`.

## Notes on real PX4 logs

PX4 topics do not always use the same coordinate origin. This version aligns the GPS, barometer, and PX4 local-position reference to the same starting frame before running the EKF. It also allows the first GPS samples through before applying outlier rejection, which prevents the filter from rejecting every GPS point during startup.

Raw PX4 accelerometer data is body-frame data. A full inertial navigation system would rotate that acceleration into the world frame using attitude estimates. To keep this project focused and stable, the real-log reader uses a world-frame acceleration estimate from the reference velocity while still using real GPS and barometer measurements for the EKF measurement updates.

### Note on real PX4 logs

For real PX4 `.ulg` logs, the EKF uses GPS as the main position measurement.
The barometer is still loaded and plotted, but it is not forced into the main
state update because PX4 barometer altitude can be in a different relative frame
than the local position reference. IMU acceleration is also not used directly in
real-log mode because it is usually body-frame acceleration and needs attitude
rotation before it becomes world-frame acceleration. This keeps the demo honest
and prevents the filter from drifting away from the real flight path.

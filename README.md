# 3D Drone EKF Sensor Fusion

This is a small robotics/state-estimation project that estimates a drone's 3D position from noisy simulated sensors. The drone gets IMU acceleration, GPS position, and barometer altitude readings. An Extended Kalman Filter combines those readings and rejects GPS points that look too far away from the predicted state.

![Graphical abstract](figures/graphical_abstract.png)

## What the project does

The script simulates a drone flying through a smooth 3D path. Then it adds realistic-ish sensor noise:

- GPS gives noisy x, y, z position measurements.
- IMU gives noisy acceleration in x, y, z.
- Barometer gives noisy altitude.
- Some GPS points are intentionally corrupted to act like bad readings.

The EKF uses the IMU for prediction, then updates the estimate using GPS and barometer data. A simple gate checks the GPS residual before accepting a GPS update. This helps prevent large GPS spikes from pulling the estimate away from the real trajectory.

## Why I built it

I wanted a project that was more useful than a basic Kalman filter example. This version has a few things that show up in real robotics work: sensor noise, bad measurements, state prediction, measurement updates, and performance checks over multiple runs.

## Project structure

```text
3d_drone_ekf_sensor_fusion/
├── main.py
├── config.py
├── requirements.txt
├── README.md
├── src/
│   ├── ekf.py
│   ├── metrics.py
│   ├── monte_carlo.py
│   ├── plots.py
│   └── simulate_drone.py
├── figures/
└── outputs/
```

## How to run

Install the requirements:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python main.py
```

You can also open `main.py` in Spyder and run it directly. It runs the whole project from one file.

## Outputs

After running, the project saves:

```text
figures/graphical_abstract.png
figures/trajectory_3d.png
figures/position_error.png
figures/altitude_estimate.png
outputs/flight_estimation_results.csv
outputs/monte_carlo_results.csv
```

## Main results shown by the script

The terminal prints:

- raw GPS RMSE
- EKF position RMSE
- RMSE for x, y, and z
- number of injected GPS outliers
- number of GPS measurements rejected by the filter
- Monte Carlo average RMSE values

## Notes

This is a simulation project, so the flight path and sensors are generated inside the code. The model is simple on purpose. It keeps the project easy to inspect while still showing the main idea behind sensor fusion.

Some ideas for future upgrades:

- add attitude estimation with roll, pitch, and yaw
- use a nonlinear drone motion model
- load real drone logs from CSV
- add a live animation of the estimated path
- compare EKF with a complementary filter or UKF

## Resume line

Built a 3D drone sensor-fusion project using an Extended Kalman Filter to combine noisy GPS, IMU, and barometer data, with GPS outlier rejection, RMSE evaluation, and Monte Carlo testing.

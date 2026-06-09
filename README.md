# 3D Drone State Estimation using EKF Sensor Fusion

![Graphical Abstract](figures/graphical_abstract.png)

## Overview

This project implements a 3D state estimation pipeline for drone flight data using an Extended Kalman Filter (EKF).

The estimator combines measurements from multiple sensors and produces a filtered estimate of the vehicle position and velocity over time. The project supports both real PX4 flight logs and a synthetic simulation environment for testing.

The implementation includes:

* GPS position measurements
* IMU acceleration data
* Barometer altitude measurements
* GPS outlier rejection
* Extended Kalman Filter state estimation
* Performance evaluation and visualization

---

## Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Project

Run the complete workflow:

```bash
python main.py
```

The script will:

1. Search for PX4 flight logs
2. Download public PX4 logs if enabled
3. Load sensor measurements
4. Run the EKF
5. Generate plots
6. Export CSV results

---

## Outputs

Generated figures:

* 3D trajectory estimate
* Position error comparison
* Altitude estimate

Generated files:

```text
outputs/
├── flight_estimation_results.csv
└── monte_carlo_results.csv
```

---

## Data Sources

### PX4 Flight Logs

Public flight logs can be downloaded from:

https://review.px4.io

The project automatically searches for compatible `.ulg` files inside the data directory.

### Simulation Mode

If enabled in `config.py`, the project can generate synthetic GPS, IMU, and barometer measurements for testing.

---

## Method

State vector:

```text
[x, y, z, vx, vy, vz]
```

Prediction:

* Constant-velocity motion model

Measurements:

* GPS position
* Barometer altitude

Outlier rejection:

* Mahalanobis-distance gating

---

## License

This repository is provided for educational and research purposes.

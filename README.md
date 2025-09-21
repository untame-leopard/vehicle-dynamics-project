# Vehicle Dynamics Project: Lap Time Simulation & Real Telemetry

This project explores **vehicle dynamics and lap time simulation**, starting from longitudinal models and building up to a **quasi-steady-state (QSS) lap time simulator**.
The latest milestone is the integration of **real-world telemetry from AiM/RaceStudio** into the workflow, enabling validation of models against kart data.

---

## Project Overview

* **0–200–0 simulation** with KPIs (0–100, 0–200, top speed, 100–0 braking distance).
* **Aerodynamic effects**: downforce (Cl·A) and drag trade-offs.
* **Track-based models**: ingesting curvature/arc length for lap-time prediction.
* **Telemetry integration**:

  * Added a robust AiM CSV loader (`src/vdyn/telemtry.py`)
  * Normalises channels (`time_s`, `speed_mps`, `distance_m`, `rpm`)
  * Unit-tested with real AiM exports
* **Goal:** compare simulation outputs directly against measured telemetry for driver/kart development.

---

## Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/untame-leopard/vehicle-dynamics-project.git
cd vehicle-dynamics-project
pip install -e .
```

This makes the package importable from anywhere:

```python
from vdyn.models.longitudinal_1d import Vehicle, accel_brake_run, compute_kpis
```

---

## Usage

### 0–200–0 & Aero Effects

Run the notebook:

```bash
jupyter notebook notebooks/longitudinal_1d.ipynb
```

Generates speed, distance, and acceleration plots, along with key performance indicators.

---

### Telemetry (AiM CSV)

Telemetry can be loaded from a RaceStudio CSV export:

```python
from src.vdyn.telemetry import load_aim_csv

df = load_aim_csv("data/[file].csv") # Replace '[file]' with the name of your specific CSV file (e.g., '2.csv').
print(df[["time_s","distance_m","speed_mps","rpm"]].head())
```

Quick plot:

```python
import matplotlib.pyplot as plt

plt.plot(df["distance_m"], df["speed_mps"])
plt.xlabel("Distance [m]")
plt.ylabel("Speed [m/s]")
plt.title("AiM GPS Speed vs Distance")
plt.show()
```

---

## Repo Structure

```
src/vdyn/           # Vehicle dynamics code
  models/           # Longitudinal, aero, etc.
  tracks/           # Track curvature data (planned)
  utils/            # Utility modules for vehicle dynamics computations
  controllers/      # Control algorithms for trajectory following

scripts/            # One-off runnable scripts
notebooks/          # Jupyter notebooks with results
tests/              # Pytest unit tests
data/               # Example telemetry (local, gitignored)
```

---

## Tests

Run with:

```bash
pytest -q
```

Tests cover:

* Physics sanity checks (speed monotonicity, drag/top speed, braking with μ and downforce).
* Telemetry parsing (`tests/test_telemetry.py`) ensures CSV loader returns clean `time_s`, `speed_mps`, `distance_m`.

---

## Roadmap

### Foundations (done)

* 0–200–0 baseline (KPIs, plots, tests)
* Aerodynamic downforce (CL·A) + drag/braking trade‑off
* Track ingestion (centerline CSV → arc length `s` and curvature `κ`) + Frenet frame (beta)
* AiM telemetry loader with unit tests
* First telemetry visuals (GPS speed vs distance, RPM trace)
* Speed‑profile generator (forward/backward pass) with friction circle & simple tyre model

### In progress (next up)

* Lap‑time estimation from telemetry (∫ ds / v) and equal/track‑defined sectors
* Sim ↔ telemetry overlay aligned on distance, with error metrics (Δv(s), time loss)

### Stretch goals

* Driver delta analysis (brake points, min‑corner speeds, throttle traces)
* MATLAB parity (core longitudinal + QSS)
* CI, profiling, and documentation polish

---

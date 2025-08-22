# Vehicle Dynamics Project: Longitudinal 0–200–0 & Aero Effects

This project explores **vehicle longitudinal dynamics** with a focus on acceleration, braking, and aerodynamics.
The end goal is to build up towards a **quasi-steady-state (QSS) lap time simulator**, connecting physics, code quality, and engineering workflow.

---

## Project Overview

* Baseline 0–200–0 simulation with KPIs (0–100, 0–200, top speed, 100–0 braking distance).
* Aerodynamic downforce (Cl·A) added, showing the trade-off between drag and braking performance.
* Roadmap includes lap-time simulation, interactive visualisation, and MATLAB parity.

The repo is structured with a `src/` package (`vdyn`) containing reusable models, and notebooks for results/plots.

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
from vdyn.models.longitudinal import Vehicle, accel_brake_run, compute_kpis
```

---

## Usage

### Baseline 0–200–0

Run the notebook:

```bash
jupyter notebook notebooks/longitudinal_1d.ipynb
```

It produces speed, distance, and acceleration plots, along with key performance indicators.


### Aero Effects

The aero comparison notebook (`aero_effects.ipynb`) shows how downforce shortens braking distance but can slow top speed. (planned)


---

## Repo Structure

```
src/vdyn/          # Vehicle dynamics code
  models/          # Longitudinal, aero, etc.
  tracks/          # Track curvature data (planned)

notebooks/         # Jupyter notebooks with results
tests/             # Pytest unit tests
```

---

## Tests

Run the physics sanity checks with:

```bash
pytest -q
```

Tests cover:

* Speed monotonicity in 0–200–0
* Drag ↓ ⇒ higher top speed
* Brake μ ↑ ⇒ shorter 100–0 distance
* Downforce ↑ ⇒ shorter 100–0 distance

---

## Dependencies

* numpy
* matplotlib
* pandas (optional, for tables in notebooks)
* pytest (for testing)

---

## Roadmap

- [x] Longitudinal 0–200–0 baseline (KPIs, plots, tests)
- [x] Aerodynamic downforce (CL·A)
- [ ] Track ingestion (CSV of s, κ; toy track generator)
- [ ] QSS lap-time simulation (forward/backward pass, friction circle)
- [ ] Validation tests (lap time sensitivity to aero & μ)
- [ ] Interactive visualisation (Streamlit/Jupyter widgets + telemetry export)
- [ ] MATLAB parity (core longitudinal + QSS)
- [ ] CI integration, profiling, documentation polish
---

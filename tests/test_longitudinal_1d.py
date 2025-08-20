import numpy as np
from src.vdyn.models.longitudinal_1d import Vehicle, accel_brake_run, compute_kpis

def test_accel_brake_monotone_to_peak_and_back():
    car = Vehicle()
    T, V, S, A = accel_brake_run(car, dt=0.01)
    i_peak = int(np.argmax(V))
    # non-decreasing to peak (allow tiny numerical wiggle)
    assert np.all(np.diff(V[:i_peak+1]) >= -1e-9)
    # non-increasing after peak
    assert np.all(np.diff(V[i_peak:]) <= 1e-9)
    # sanity speed bounds
    assert V.min() >= 0.0 and V.max() < 200.0

def test_drag_reduction_increases_top_speed():
    car_hi_drag = Vehicle(CdA=1.10)
    car_lo_drag = Vehicle(CdA=0.80)  # lower drag area
    _, V1, _, _ = accel_brake_run(car_hi_drag, dt=0.01)
    _, V2, _, _ = accel_brake_run(car_lo_drag, dt=0.01)
    assert V2.max() > V1.max()

def test_more_brake_mu_shorter_100_to_0_distance():
    lo_mu = Vehicle(mu_brake=0.8)
    hi_mu = Vehicle(mu_brake=1.6)
    T1, V1, S1, _ = accel_brake_run(lo_mu, dt=0.01)
    T2, V2, S2, _ = accel_brake_run(hi_mu, dt=0.01)
    from src.vdyn.models.longitudinal_1d import kpi_100_to_0_brake_distance
    d1 = kpi_100_to_0_brake_distance(T1, V1, S1)
    d2 = kpi_100_to_0_brake_distance(T2, V2, S2)
    assert np.isfinite(d1) and np.isfinite(d2)
    assert d2 < d1  # higher grip -> shorter distance


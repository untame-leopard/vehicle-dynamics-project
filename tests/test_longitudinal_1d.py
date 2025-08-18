import numpy as np
from src.vdyn.longitudinal_1d import Vehicle, accel_brake_run

def test_accel_brake_run():
    car = Vehicle()
    T, V, S, A = accel_brake_run(car, dt=0.01)
    # peak speed reached once
    i_peak = int(np.argmax(V))
    assert np.all(np.diff(V[:i_peak+1]) >= -1e-9)  # non-decreasing to peak
    assert np.all(np.diff(V[i_peak:]) <= 1e-9)     # non-increasing after peak
    # speeds always in [0, ~1000] m/s
    assert V.min() >= 0.0 and V.max() < 200.0

import numpy as np
from vdyn.models.longitudinal_1d import Vehicle, accel_brake_run, kpi_0_to_100_kmh, kpi_100_to_0_brake_distance, kpi_0_to_200_kmh
from vdyn.models.powertrain import Powertrain, Gearbox, default_highrev_curve

def test_accel_brake_monotone_to_peak_and_back():
    car = Vehicle()
    T, V, S, A, _, _  = accel_brake_run(car, dt=0.01)
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
    _, V1, _, _, _, _ = accel_brake_run(car_hi_drag, dt=0.01)
    _, V2, _, _, _, _ = accel_brake_run(car_lo_drag, dt=0.01)
    assert V2.max() > V1.max()

def test_more_brake_mu_shorter_100_to_0_distance():
    lo_mu = Vehicle(mu_brake=0.8)
    hi_mu = Vehicle(mu_brake=1.6)
    T1, V1, S1, _, _, _ = accel_brake_run(lo_mu, dt=0.01)
    T2, V2, S2, _, _, _ = accel_brake_run(hi_mu, dt=0.01)
    d1 = kpi_100_to_0_brake_distance(T1, V1, S1)
    d2 = kpi_100_to_0_brake_distance(T2, V2, S2)
    assert np.isfinite(d1) and np.isfinite(d2)
    assert d2 < d1  # higher grip -> shorter distance

def test_more_downforce_shorter_100_to_0_distance():
    # Hold CdA constant to isolate CL effect
    base = dict(CdA=0.90)
    car_no_df = Vehicle(ClA=0.0, **base)
    car_hi_df = Vehicle(ClA=2.5, **base)  # strong downforce (F1-ish scale)
    T1, V1, S1, _, _, _ = accel_brake_run(car_no_df, dt=0.01)
    T2, V2, S2, _, _, _ = accel_brake_run(car_hi_df, dt=0.01)
    d1 = kpi_100_to_0_brake_distance(T1, V1, S1)
    d2 = kpi_100_to_0_brake_distance(T2, V2, S2)
    assert d2 < d1

def test_downforce_shortens_braking_even_with_ld_coupling():
    base = dict(CdA=0.90)
    no_df = Vehicle(ClA=0.0, dCdA_per_ClA=0.1, **base)   # L/D ≈ 10
    hi_df = Vehicle(ClA=2.5, dCdA_per_ClA=0.1, **base)
    T0,V0,S0,_, _, _ = accel_brake_run(no_df, dt=0.01)
    T1,V1,S1,_, _, _ = accel_brake_run(hi_df, dt=0.01)
    assert kpi_100_to_0_brake_distance(T1,V1,S1) < kpi_100_to_0_brake_distance(T0,V0,S0)

def test_ld_coupling_penalizes_0_to_200_time():
    base = dict(CdA=0.90)
    no_df = Vehicle(ClA=0.0, dCdA_per_ClA=0.1, **base)
    hi_df = Vehicle(ClA=2.5, dCdA_per_ClA=0.1, **base)
    T0,V0,_,_, _, _ = accel_brake_run(no_df, dt=0.01)
    T1,V1,_,_, _, _ = accel_brake_run(hi_df, dt=0.01)
    assert kpi_0_to_200_kmh(T1,V1) >= kpi_0_to_200_kmh(T0,V0)

def test_shorter_final_drive_quicker_0_100_until_traction_limited():
    base = Vehicle()
    gb_a = Gearbox([3.1,2.2,1.7,1.35,1.12,0.95], 3.4, 0.33)
    gb_b = Gearbox([3.1,2.2,1.7,1.35,1.12,0.95], 3.8, 0.33) 
    pt_a = Powertrain(default_highrev_curve(), gb_a)
    pt_b = Powertrain(default_highrev_curve(), gb_b)
    Ta,Va,Sa,_, _, _ = accel_brake_run(base, dt=0.01, powertrain=pt_a)
    Tb,Vb,Sb,_, _, _ = accel_brake_run(base, dt=0.01, powertrain=pt_b)

    assert kpi_0_to_100_kmh(Tb,Vb) <= kpi_0_to_100_kmh(Ta,Va) + 0.05 
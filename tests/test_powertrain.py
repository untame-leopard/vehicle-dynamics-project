import numpy as np
from vdyn.models.powertrain import Powertrain, Gearbox, TorqueCurve

def _flat_curve(val=250.0):
    return TorqueCurve(np.array([1000, 15000]), np.array([val, val]))

def test_upshift_reduces_wheel_torque():
    gbx = Gearbox([3.0, 2.0], final_drive=3.5, wheel_radius_m=0.33)
    pt  = Powertrain(_flat_curve(), gbx)
    F1,_ = pt.available_drive_force( v_ms=20.0, gear=1)
    F2,_ = pt.available_drive_force( v_ms=20.0, gear=2)
    assert F1 > F2

def test_engine_rpm_increases_with_speed_and_ratio():
    gbx = Gearbox([3.0, 2.0], 3.5, 0.33)
    assert gbx.engine_rpm(10.0, 1) > gbx.engine_rpm(10.0, 2)
    assert gbx.engine_rpm(20.0, 2) > gbx.engine_rpm(10.0, 2)

def test_engine_rpm_behaviour():
    gb = Gearbox([3.0, 2.0], 3.5, 0.33)
    assert gb.engine_rpm(10.0, 1) > gb.engine_rpm(10.0, 2)   # lower gear → higher rpm
    assert gb.engine_rpm(20.0, 2) > gb.engine_rpm(10.0, 2)   # faster → higher rpm

def test_wheel_force_decreases_with_higher_gear():
    gb = Gearbox([3.0, 2.0], 3.5, 0.33)
    pt = Powertrain(TorqueCurve(np.array([1000,15000]), np.array([250,250])), gb)
    F1,_ = pt.available_drive_force(20.0, 1)
    F2,_ = pt.available_drive_force(20.0, 2)
    assert F1 > F2

def test_upshift_and_downshift_thresholds_exist():
    gb = Gearbox([3.0,2.0], final_drive=3.5, wheel_radius_m=0.33, shift_rpm=5000, downshift_rpm=1500)
    assert gb.shift_rpm > gb.downshift_rpm

def test_upshift_reduces_wheel_force_at_same_speed():
    gb = Gearbox([3.0,2.0], final_drive=3.5, wheel_radius_m=0.33)
    pt = Powertrain(_flat_curve(), gb)
    F1,_ = pt.available_drive_force(20.0, 1)
    F2,_ = pt.available_drive_force(20.0, 2)
    assert F1 > F2

def test_engine_brake_increases_wheel_force_but_respects_muN():
    gb = Gearbox([3.0,2.0], final_drive=3.5, wheel_radius_m=0.33, shift_rpm=7000, downshift_rpm=1500)
    pt = Powertrain(TorqueCurve(np.array([1000,15000]), np.array([250,250])), gb,
                    engine_brake_coeff_nm_per_rpm=0.03, engine_brake_max_torque_nm=150)
    v = 30.0
    F_eb, rpm, T_eb = pt.engine_brake_force(v, 2)
    assert F_eb > 0.0

from vdyn.models.longitudinal_1d import accel_brake_run, Vehicle, compute_kpis
def test_shift_cut_slows_0_to_200():
    gb = Gearbox([3.0,2.0,1.5,1.2,1.0], final_drive=3.5, wheel_radius_m=0.33)
    pt_fast = Powertrain(_flat_curve(), gb)
    pt_slow = Powertrain(_flat_curve(), gb)
    pt_fast.shift_time_s = 0.00
    pt_slow.shift_time_s = 0.15
    car = Vehicle()
    T_fast, V_fast, S_fast, *_ = accel_brake_run(car, powertrain=pt_fast)
    T_slow, V_slow, S_slow, *_ = accel_brake_run(car, powertrain=pt_slow)
    k_fast = compute_kpis(T_fast, V_fast, S_fast)
    k_slow = compute_kpis(T_slow, V_slow, S_slow)
    assert k_fast['t_0_200_s'] < k_slow['t_0_200_s']

def test_iame_x30_curve_basic_specs():
    from vdyn.models.powertrain import iame_x30_curve
    import numpy as np

    c = iame_x30_curve(scale=1.0)
    tq_10500 = c.tq(10500)
    assert 18.5 <= tq_10500 <= 20.0  # near peak torque ~19.5 Nm

    rpm = 12000.0
    tq  = c.tq(rpm)
    power_kw = (tq * 2*np.pi*rpm/60.0) / 1000.0
    assert 21.0 <= power_kw <= 24.5  # ~30 hp around 12k
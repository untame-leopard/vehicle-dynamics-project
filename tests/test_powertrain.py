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
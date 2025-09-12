import numpy as np
from vdyn.limits.params_kart import KartParams
from vdyn.limits.lat_lim import ay_max, v_from_kappa
from vdyn.limits.long_lim import a_x_max, a_x_min

def test_lateral_cap_on_circle_matches_formula():
    p = KartParams(mu_lat=1.8)
    kappa = np.full(100, 1/25.0)  # radius=25 m
    v_k = v_from_kappa(kappa, p, v_cap=200.0)
    # expected v = sqrt(mu*g/|k|) everywhere
    expected = np.sqrt(p.mu_lat*9.81 / abs(kappa[0]))
    assert np.allclose(v_k, expected, rtol=1e-3, atol=1e-6)

def test_ay_max_is_mu_g_without_aero():
    p = KartParams(mu_lat=1.7, ClA=0.0)
    v = np.array([0.0, 10.0, 30.0])
    ay = ay_max(v, p)
    assert np.allclose(ay, p.mu_lat*9.81)

def test_longitudinal_signs_and_trends_power_fallback():
    p = KartParams(P_max=24000.0, CdA=0.7, Crr=0.012)
    v = np.array([1.0, 10.0, 30.0])
    ax_fwd = a_x_max(v, p)    # drive accel
    ax_brk = a_x_min(v, p)    # braking (negative)
    # forward accel should be positive but decay with v (drag and P/v)
    assert np.all(ax_fwd > 0.0)
    assert ax_fwd[0] >= ax_fwd[1] >= ax_fwd[2]
    # braking should be negative and magnitude grows a bit with v (drag term)
    assert np.all(ax_brk < 0.0)
    assert abs(ax_brk[0]) <= abs(ax_brk[1]) <= abs(ax_brk[2])

def test_drive_traction_cap_applies():
    p = KartParams(mu_long_drive=1.2, CdA=0.0, Crr=0.0, P_max=1e9)  # huge power, traction-limited
    v = np.array([5.0])
    ax = float(a_x_max(v, p))  # should clamp to mu_long_drive * g
    assert np.isclose(ax, p.mu_long_drive*9.81, rtol=1e-3, atol=1e-3)

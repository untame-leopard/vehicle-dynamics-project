from typing import Callable, Optional, Union
import numpy as np
from vdyn.limits.params_kart import KartParams

Number = Union[float, np.ndarray]
g = 9.81

def aero_forces(v: Number, p: KartParams):
    """
    Compute aerodynamic drag, normal load (with optional downforce), and rolling resistance.

    Parameters
    ----------
    v : float or ndarray
        Speed [m/s].
    p : KartParams

    Returns
    -------
    (Fd, Frr, N) : tuple of float or ndarray
        Fd: aerodynamic drag force [N]
        Frr: rolling resistance force [N]
        N: normal load [N]
    """
    v = np.asarray(v, float)
    Fd = 0.5 * p.rho * p.CdA * v * v
    N  = p.m * g + 0.5 * p.rho * p.ClA * v * v
    Frr = p.Crr * p.m * g
    return Fd, Frr, N

def drive_force_single_gear_from_torque(v: Number, p: KartParams, Te_fn: Callable[[Number], Number]) -> np.ndarray:
    """
    Wheel drive force for a single-gear kart using an engine torque curve.

    Parameters
    ----------
    v : float or ndarray
        Speed [m/s].
    p : KartParams
        Includes ratio, eta, Rw.
    Te_fn : callable
        Engine torque curve T_e(omega) [N·m] as a function of engine speed rad/s.

    Returns
    -------
    ndarray
        Available wheel drive force before traction/drag limits [N].
    """
    v = np.asarray(v, float)
    omega = (p.ratio * v) / max(p.Rw, 1e-6)  # rad/s (wheel -> engine)
    Te = np.asarray(Te_fn(omega), float)
    return (p.eta * p.ratio * Te) / max(p.Rw, 1e-6)

def drive_force_power_fallback(v: Number, p: KartParams) -> np.ndarray:
    """
    Power-based wheel force fallback when no torque curve is available.

    F = P_max / v clipped at a large value for very low speeds.
    """
    v = np.asarray(v, float)
    return np.minimum(p.P_max / np.maximum(v, 1e-3), 1e9)

def a_x_max(v: Number, p: KartParams, Te_fn: Optional[Callable[[Number], Number]] = None) -> np.ndarray:
    """
    Maximum forward acceleration a_x,max(v) accounting for traction, aero and rolling resistance.

    If Te_fn is provided, uses torque-based single-gear model; otherwise uses power fallback.

    Parameters
    ----------
    v : float or ndarray
        Speed [m/s].
    p : KartParams
    Te_fn : callable or None
        Engine torque curve T_e(omega) [N·m]. If None, uses p.P_max fallback.

    Returns
    -------
    ndarray
        a_x,max [m/s^2]
    """
    v = np.asarray(v, float)
    Fd, Frr, N = aero_forces(v, p)
    F_drv_raw = drive_force_single_gear_from_torque(v, p, Te_fn) if Te_fn else drive_force_power_fallback(v, p)
    # Traction limit under drive
    F_drv = np.minimum(F_drv_raw, p.mu_long_drive * N)
    return (F_drv - Fd - Frr) / p.m

def a_x_min(v: Number, p: KartParams) -> np.ndarray:
    """
    Maximum braking deceleration (negative), including aero and rolling resistance.

    Parameters
    ----------
    v : float or ndarray
        Speed [m/s].
    p : KartParams

    Returns
    -------
    ndarray
        a_x,min [m/s^2] (negative).
    """
    v = np.asarray(v, float)
    Fd, Frr, N = aero_forces(v, p)
    F_brk = p.mu_long_brake * N
    return -(F_brk + Fd + Frr) / p.m

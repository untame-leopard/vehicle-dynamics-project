from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, Tuple
import numpy as np

# ---------- Parameters & tires ----------

@dataclass
class DynParams:
    m: float = 800.0       # mass [kg]
    Iz: float = 1200.0     # yaw inertia [kg m^2]
    Lf: float = 1.3        # CoG -> front axle [m]
    Lr: float = 1.5        # CoG -> rear axle [m]  (wheelbase = Lf + Lr)
    Cf: float = 90000.0    # front cornering stiffness [N/rad]
    Cr: float = 110000.0   # rear cornering stiffness [N/rad]
    delta_max: float = np.deg2rad(25.0)
    u_min: float = 0.5     # speed floor to avoid div-by-zero

def linear_tire_forces(u: float, v: float, r: float, delta: float, p: DynParams) -> Tuple[float, float]:
    """
    Returns (Fy_front, Fy_rear) using linear tires. Positive Fy is to the left.
    """
    u_eff = max(abs(u), p.u_min) * np.sign(u if u != 0.0 else 1.0)
    # front & rear slip angles (small-angle bicycle)
    alpha_f = delta - (v + p.Lf * r) / u_eff
    alpha_r = - (v - p.Lr * r) / u_eff
    Fy_f = -p.Cf * alpha_f
    Fy_r = -p.Cr * alpha_r
    return Fy_f, Fy_r
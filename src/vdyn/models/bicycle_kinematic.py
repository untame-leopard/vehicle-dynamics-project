from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Tuple

@dataclass
class KinematicBicycle:
    """
    Simple kinematic bicycle for planar motion.
    State: [x, y, psi, v].  Inputs: [ax, delta].
    Relation: psi_dot = v/L * tan(delta).
    """
    wheelbase_m: float = 2.8

    def step(self, x: float, y: float, psi: float, v: float,
             ax: float, delta: float, dt: float) -> Tuple[float, float, float, float]:
        # Kinematic update (explicit Euler is fine here)
        beta = 0.0  # no slip in kinematic model
        v_next   = max(0.0, v + ax * dt)
        psi_dot  = v / self.wheelbase_m * np.tan(delta)
        psi_next = psi + psi_dot * dt
        x_next   = x + v * np.cos(psi + beta) * dt
        y_next   = y + v * np.sin(psi + beta) * dt
        return x_next, y_next, psi_next, v_next

    def delta_for_curvature(self, kappa: float) -> float:
        # for small slip angles: kappa ≈ tan(delta)/L  -> delta ≈ arctan(L*kappa)
        return np.arctan(self.wheelbase_m * kappa)
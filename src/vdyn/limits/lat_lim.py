from typing import Union
import numpy as np
from vdyn.limits.params_kart import KartParams

Number = Union[float, np.ndarray]
g = 9.81

def ay_max(v: Number, p: KartParams) -> Number:
    """
    Maximum lateral acceleration as a function of speed.

    Notes
    -----
    For karts we start with a speed-independent μ*g cap. If you later have measurable
    downforce, extend as: μ_lat*g + 0.5*rho*ClA/m * v^2.

    Parameters
    ----------
    v : float or array
        Speed [m/s] (currently unused in the baseline model).
    p : KartParams
        Parameters including mu_lat.

    Returns
    -------
    float or ndarray
        Lateral acceleration cap [m/s^2].
    """
    if isinstance(v, np.ndarray):
        return np.full_like(v, p.mu_lat * g, dtype=float)
    return p.mu_lat * g

def v_from_kappa(kappa: np.ndarray, p: KartParams, v_cap: float) -> np.ndarray:
    """
    Curvature-limited corner speed v_kappa(s) := sqrt(ay_max/|kappa|), with a global speed cap.

    Parameters
    ----------
    kappa : ndarray
        Track curvature κ(s) [1/m].
    p : KartParams
        Parameters (μ_lat).
    v_cap : float
        Global speed cap to prevent blow-up on straights.

    Returns
    -------
    ndarray
        Corner-limited speed profile [m/s] over s-grid of κ.
    """
    k = np.abs(np.asarray(kappa, dtype=float))
    aycap = ay_max(0.0, p)
    with np.errstate(divide="ignore", invalid="ignore"):
        v = np.sqrt(np.where(k > 1e-9, aycap / k, np.inf))
    return np.minimum(v, float(v_cap))

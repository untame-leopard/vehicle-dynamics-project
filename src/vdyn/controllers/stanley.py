import numpy as np

def stanley_delta(e_y, e_psi, v, k_gain=1.0, ks=1e-2, delta_ff=0.0, limits=None):
    """
    Stanley steering: delta = e_psi + atan(k_gain * e_y / (v + ks)) + delta_ff
    - e_y   : cross-track error [m]
    - e_psi : heading error [rad]
    - v     : speed [m/s]
    - k_gain: lateral gain
    - ks    : low-speed softening
    - delta_ff: feed-forward (e.g., atan(L*kappa))
    - limits: optional (min,max) radians clamp
    """
    steer_fb = np.arctan2(k_gain * e_y, max(v, 0.0) + ks)
    delta = e_psi + steer_fb + delta_ff
    if limits is not None:
        delta = np.clip(delta, limits[0], limits[1])
    return float(delta)

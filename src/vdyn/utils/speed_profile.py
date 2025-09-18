import numpy as np
from typing import Callable

def curvature_speed_profile(trk, a_lat_max=8.0, v_cap=60.0, n=4000):
    """
    Returns (S, v_ref_s) where v_ref_s is the curvature-limited speed along s.
    v_max(s) = sqrt(a_lat_max / (|kappa(s)| + eps)), capped by v_cap.
    """
    S = np.linspace(0.0, trk.L, n, endpoint=False)
    K = trk.sample_kappa(S)
    v_k = np.sqrt(np.maximum(a_lat_max, 0.0) / (np.abs(K) + 1e-6))
    v_ref_s = np.minimum(v_k, v_cap)
    return S, v_ref_s

def v_ref_lookup(s, S, v_ref_s, L):
    """ Interpolate periodic v_ref(s) at (possibly unwrapped) s. """
    sm = np.mod(s, L)
    return float(np.interp(sm, S, v_ref_s))

def forward_backward_speed_profile(
    s: np.ndarray,
    v_cap_kappa: np.ndarray,
    ds: float,
    axmax_fn: Callable[[float], float],
    axmin_fn: Callable[[float], float],
) -> np.ndarray:
    ''' Compute the limit lap speed v_opt(s) via a forward–backward pass.'''
    v = np.array(v_cap_kappa, dtype=float)

    # Forward (accel-limited)
    for i in range(len(s) - 1):
        a = float(axmax_fn(v[i]))
        v[i + 1] = min(v[i + 1], np.sqrt(max(0.0, v[i] * v[i] + 2.0 * a * ds)))

    # Backward (brake-limited)
    for i in range(len(s) - 2, -1, -1):
        a = abs(float(axmin_fn(v[i + 1])))
        v[i] = min(v[i], np.sqrt(max(0.0, v[i + 1] * v[i + 1] + 2.0 * a * ds)))

    return v

def soften_kappa_seam(kappa, W=12):
    """Blend first/last W samples so curvature is continuous at the seam."""
    k = kappa.copy()
    if len(k) < 2*W:  # tiny guard
        return k
    a = np.linspace(0, 1, W)
    k_start = k[:W].copy()
    k_end   = k[-W:].copy()
    # crossfade: last -> first
    k[:W]  = (1 - a) * k_start + a * k_end
    k[-W:] = (1 - a[::-1]) * k_end + a[::-1] * k_start
    return k

def limit_speed_profile_cyclic(S, v_kappa, ds, axmax, axmin, v_cap, iters=3):
    i0   = int(np.argmin(v_kappa))                # move seam to tightest corner
    vkap = np.roll(v_kappa, -i0)

    v = np.minimum(vkap, v_cap).astype(float)
    v[0] = max(1.0, v[0])                        

    for _ in range(iters):
        # forward (accel)
        for i in range(len(v) - 1):
            a = max(0.0, axmax(v[i]))
            v[i+1] = min(v[i+1], np.sqrt(max(0.0, v[i]*v[i] + 2.0*a*ds)))
        # backward (brake)  <-- NOTE the '+' sign
        for i in range(len(v) - 2, -1, -1):
            a = abs(axmin(v[i+1]))                # magnitude of decel
            v[i] = min(v[i], np.sqrt(max(0.0, v[i+1]*v[i+1] + 2.0*a*ds)))
        v = np.minimum(v, vkap)                 

    return np.roll(v, i0)

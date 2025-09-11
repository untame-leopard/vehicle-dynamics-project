import numpy as np

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

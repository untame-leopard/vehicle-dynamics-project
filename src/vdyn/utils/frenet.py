import numpy as np

def wrap_to_pi(a):
    return (a + np.pi) % (2*np.pi) - np.pi

def track_errors(x, y, psi, trk):
    """
    Signed cross-track error e_y, heading error e_psi, and nearest arclength s*.
    """
    s_star, xc, yc, _, _ = trk.project_to_centerline(x, y)
    psi_c = trk.sample_psi(s_star)

    # vector from centerline point to vehicle
    rx = x - xc
    ry = y - yc

    # tangent at centerline
    tx = np.cos(psi_c)
    ty = np.sin(psi_c)

    # signed lateral error: positive if to the left of tangent
    sign = np.sign(tx*ry - ty*rx)
    e_y = sign * np.hypot(rx, ry)

    e_psi = wrap_to_pi(psi - psi_c)
    return e_y, e_psi, s_star

from __future__ import annotations
import numpy as np
from scipy.interpolate import CubicSpline

import numpy as np

def enforce_periodic_endpoints(x, y, psi, kappa):
    """
    Make first and last samples identical so CubicSpline(..., bc_type='periodic') is valid.
    Also wraps heading to avoid a 2pi jump.
    """
    x = np.asarray(x, float).copy()
    y = np.asarray(y, float).copy()
    psi = np.asarray(psi, float).copy()
    kappa = np.asarray(kappa, float).copy()

    # positions: exact match
    x[-1], y[-1] = x[0], y[0]
    # heading: wrap difference to [-pi, pi] then force match
    psi[-1] = psi[0]
    # curvature: match
    kappa[-1] = kappa[0]
    return x, y, psi, kappa


class CenterlineTrack:
    """
    Periodic centerline model x(s), y(s) with smooth cubic splines and Frenet utilities.

    Parameters
    ----------
    s : array-like [m]
        Monotone increasing arclength samples along the closed track (0 .. L).
    x, y : array-like [m]
        Global coordinates at each s.
    psi : array-like [rad], optional
        Heading. If None, computed from spline tangents.
    kappa : array-like [1/m], optional
        Curvature. If None, computed from spline derivatives.
    """

    def __init__(self, s, x, y, psi=None, kappa=None):
        s = np.asarray(s, float)
        x = np.asarray(x, float)
        y = np.asarray(y, float)

        # sort by s and ensure strictly increasing
        order = np.argsort(s)
        s, x, y = s[order], x[order], y[order]

        # remove duplicated terminal sample if present (periodic splines require it)
        if len(s) >= 2 and np.isclose(s[-1] - s[0], 0.0) and np.allclose([x[0], y[0]], [x[-1], y[-1]]):
            s, x, y = s[:-1], x[:-1], y[:-1]

        # force strictly increasing s
        ds = np.diff(s) #Prints array of the difference between s values. If one s value is smaller than the one before it does'nt work (negative ds)
        if not np.all(ds > 0):
            raise ValueError("s must be strictly increasing for spline construction.")

        self.s = s
        self.x = x
        self.y = y
        self.L = float(s[-1])

        # periodic splines for x(s), y(s)
        self._x_spline = CubicSpline(self.s, self.x, bc_type="periodic")
        self._y_spline = CubicSpline(self.s, self.y, bc_type="periodic")

        # heading from tangent if not provided
        if psi is None:
            dx = self._x_spline.derivative()(self.s)
            dy = self._y_spline.derivative()(self.s)
            self.psi = np.arctan2(dy, dx)
        else:
            psi = np.asarray(psi, float)
            if psi.shape != self.s.shape:
                raise ValueError("psi must have same shape as s.")
            self.psi = psi[order]

        # curvature from x(s), y(s) if not provided
        if kappa is None:
            dx  = self._x_spline.derivative()(self.s)
            dy  = self._y_spline.derivative()(self.s)
            ddx = self._x_spline.derivative(2)(self.s)
            ddy = self._y_spline.derivative(2)(self.s)
            denom = np.clip((dx*dx + dy*dy)**1.5, 1e-9, None)
            self.kappa = (dx*ddy - dy*ddx) / denom
        else:
            kappa = np.asarray(kappa, float)
            if kappa.shape != self.s.shape:
                raise ValueError("kappa must have same shape as s.")
            self.kappa = kappa[order]

        self._kappa_spline = CubicSpline(self.s, self.kappa, bc_type="periodic")

    @classmethod
    def from_dataframe(cls, df):
        """Construct from a DataFrame containing columns s,x,y[,psi,kappa]."""
        return cls(
            df["s"].to_numpy(),
            df["x"].to_numpy(),
            df["y"].to_numpy(),
            df["psi"].to_numpy() if "psi" in df.columns else None,
            df["kappa"].to_numpy() if "kappa" in df.columns else None,
        )

    # ---------- vectorised samplers (for plotting / grids) ----------
    def sample_xy(self, s_query: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized sample of centerline positions at arclength(s) s_query [m]."""
        s = np.asarray(s_query, float)
        sm = np.mod(s, self.L)
        x = self._x_spline(sm)
        y = self._y_spline(sm)
        return (float(x), float(y)) if np.ndim(x) == 0 else (x, y)

    def sample_psi(self, s_query: np.ndarray) -> np.ndarray:
        """Vectorized heading ψ(s) [rad] at arclength(s) s_query."""
        s = np.asarray(s_query, float)
        sm = np.mod(s, self.L)
        dxds = self._x_spline.derivative()(sm)
        dyds = self._y_spline.derivative()(sm)
        psi = np.arctan2(dyds, dxds)
        return float(psi) if np.ndim(psi) == 0 else psi

    def sample_kappa(self, s_query: np.ndarray) -> np.ndarray:
        """Vectorized curvature κ(s) [1/m] at arclength(s) s_query."""
        s = np.asarray(s_query, float)
        sm = np.mod(s, self.L)
        k = self._kappa_spline(sm)
        return float(k) if np.ndim(k) == 0 else k

    # ---------- single-point evaluators (used in projection/Newton) ----------
    def x_at(self, s):   return self._x_spline(np.mod(s, self.L))
    def y_at(self, s):   return self._y_spline(np.mod(s, self.L))
    def dx(self, s):     return self._x_spline.derivative()(np.mod(s, self.L))
    def dy(self, s):     return self._y_spline.derivative()(np.mod(s, self.L))
    def ddx(self, s):    return self._x_spline.derivative(2)(np.mod(s, self.L))
    def ddy(self, s):    return self._y_spline.derivative(2)(np.mod(s, self.L))
    def psi_c(self, s):  return float(np.arctan2(self.dy(s), self.dx(s)))

    # ---------- utilities ----------
    def nearest_s(self, xq: float, yq: float, grid: int = 400) -> float:
        """
        Coarse nearest arclength by scanning a uniform grid (fast hint for Newton).
        """
        s_grid = np.linspace(0.0, self.L, grid, endpoint=False)
        xg, yg = self.sample_xy(s_grid)
        i = int(np.argmin((xg - xq)**2 + (yg - yq)**2))
        return float(s_grid[i])

    # ---------- projection / Frenet ----------
    def project_to_centerline(self, px: float, py: float, s_hint: float | None = None):
        """
        Project (px,py) to the closest point on the centerline.

        Returns
        -------
        s_star : float [m]
            Arclength of the closest point.
        x_c, y_c : float [m]
            Coordinates of the closest point.
        idx : int
            Placeholder for KD-tree index (-1: none).
        dist : float [m]
            Euclidean distance to the centerline at s_star.
        """
        S = self.L
        s = self.nearest_s(px, py) if s_hint is None else float(np.mod(s_hint, S))

        # Newton refinement on 0.5 ||r(s)||^2
        for _ in range(10):
            cx, cy = float(self.x_at(s)), float(self.y_at(s))
            dx, dy = float(self.dx(s)), float(self.dy(s))
            ddx, ddy = float(self.ddx(s)), float(self.ddy(s))
            rx, ry = px - cx, py - cy
            f  = -(rx*dx + ry*dy)
            fp = (dx*dx + dy*dy) - (rx*ddx + ry*ddy)
            step = f / (fp if abs(fp) > 1e-9 else 1e-9)
            s = (s - step) % S
            if abs(step) < 1e-6:
                break

        cx, cy = float(self.x_at(s)), float(self.y_at(s))
        dist = float(np.hypot(px - cx, py - cy))
        return float(s), cx, cy, -1, dist

    def global_to_frenet(self, xq: float, yq: float, s_hint: float | None = None) -> tuple[float, float]:
        """
        Map a global point (xq,yq) to Frenet coordinates (s, n) relative to the centerline.
        n is signed lateral offset using the local left-normal.

        Returns
        -------
        s : float [m]
        n : float [m]
        """
        s_star, xc, yc, _, _ = self.project_to_centerline(float(xq), float(yq), s_hint)

        # tangent from spline derivatives (avoid angle wrapping)
        tx, ty = float(self.dx(s_star)), float(self.dy(s_star))
        norm = (tx*tx + ty*ty)**0.5
        if norm < 1e-12:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / norm, ty / norm

        # left normal
        nx, ny = -ty, tx
        n = (xq - xc)*nx + (yq - yc)*ny 
        return float(s_star), float(n)

    def frenet_to_global(self, s_q: float, n_q: float) -> tuple[float, float]:
        """
        Map Frenet (s, n) back to global (x, y) using the spline-derived normal.
        """
        s_q = float(np.mod(s_q, self.L))
        tx, ty = float(self.dx(s_q)), float(self.dy(s_q))
        norm = (tx*tx + ty*ty)**0.5
        if norm < 1e-12:
            tx, ty = 1.0, 0.0
        else:
            tx, ty = tx / norm, ty / norm
        nx, ny = -ty, tx
        xc, yc = float(self.x_at(s_q)), float(self.y_at(s_q))
        return xc + n_q*nx, yc + n_q*ny

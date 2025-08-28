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
def f_dyn(state: np.ndarray, u_cmd: Tuple[float, float], p: DynParams,
          tire_model: Callable[[float,float,float,float,DynParams], Tuple[float,float]] = linear_tire_forces
          ) -> np.ndarray:
    """
    Calculates the derivatives of the vehicle's state variables based on planar
    dynamic bicycle model equations of motion (EOM).

    This function represents the core physics of the vehicle, determining how its
    state (position, velocity, and yaw) changes in a single moment in time.
    Args:
        state (np.ndarray): Current state vector [x, y, psi, u, v, r].
            - x: Global x position [m]
            - y: Global y position [m]
            - psi: Heading angle [rad]
            - u: Longitudinal velocity in body frame [m/s]
            - v: Lateral velocity in body frame [m/s]
            - r: Yaw rate [rad/s]
        u_cmd (Tuple[float, float]): Control inputs (delta, ax).
            - delta: Steering angle [rad]
            - ax: Longitudinal acceleration command [m/s^2]
        p (DynParams): Vehicle parameters.
        tire_model (Callable): Function to compute lateral tire forces.
            Defaults to linear tire model.

    Returns:
        np.ndarray: A vector of the state derivatives [dx, dy, dpsi, du, dv, dr].
    """
    # Unpack the state and control inputs for readability
    x, y, psi, u, v, r = state
    delta, ax = u_cmd
    
    # Clip the steering angle to prevent unrealistic inputs
    delta = float(np.clip(delta, -p.delta_max, p.delta_max))

    # Calculate lateral forces from the tire model
    Fy_f, Fy_r = tire_model(u, v, r, delta, p)

    # Vehicle Equations of Motion (EOM) for planar motion
    # Change in longitudinal velocity (du) due to commanded acceleration and centrifugal force
    du = ax + r * v
    
    # Change in lateral velocity (dv) due to tire forces and centrifugal force
    # The sum of lateral forces (F_y) divided by mass (m) gives lateral acceleration.
    dv = (Fy_f + Fy_r) / p.m - r * u
    
    # Change in yaw rate (dr) due to the yaw moment from tire forces
    # Moment = Force * Lever Arm. A positive yaw moment turns the car left.
    dr = (p.Lf * Fy_f - p.Lr * Fy_r) / p.Iz
    
    # Kinematic equations to update global position and heading
    dpsi = r  # Change in heading is equal to the yaw rate
    dx = u * np.cos(psi) - v * np.sin(psi)
    dy = u * np.sin(psi) + v * np.cos(psi)

    # Return the derivatives in a single array
    return np.array([dx, dy, dpsi, du, dv, dr])

def step_rk4(state: np.ndarray, u_cmd: Tuple[float, float], p: DynParams, dt: float,
             tire_model: Callable[[float,float,float,float,DynParams], Tuple[float,float]] = linear_tire_forces
             ) -> np.ndarray:
    """
    Performs one time step of the simulation using the Fourth-Order Runge-Kutta (RK4) method.

    RK4 is a robust numerical integration method that calculates the next state
    by taking four weighted samples of the state derivatives over the time step.
    This provides a very accurate approximation for the system's evolution.

    Args:
        state (np.ndarray): The current state vector [x, y, psi, u, v, r].
        u_cmd (Tuple[float, float]): The control inputs (delta, ax).
        p (DynParams): Vehicle physical parameters.
        dt (float): The time step size in seconds.
        tire_model (Callable): Function to calculate tire forces.

    Returns:
        np.ndarray: The new state vector after one time step.
    """
    # Calculate four samples (k1, k2, k3, k4) of the state derivatives
    k1 = f_dyn(state, u_cmd, p, tire_model)
    k2 = f_dyn(state + 0.5 * dt * k1, u_cmd, p, tire_model)
    k3 = f_dyn(state + 0.5 * dt * k2, u_cmd, p, tire_model)
    k4 = f_dyn(state + dt * k3, u_cmd, p, tire_model)

    # Calculate the next state using the weighted average of the four samples
    s_next = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    
    # Keep yaw angle (psi) within the range of [-pi, pi] for consistency
    s_next[2] = (s_next[2] + np.pi) % (2 * np.pi) - np.pi
    
    return s_next
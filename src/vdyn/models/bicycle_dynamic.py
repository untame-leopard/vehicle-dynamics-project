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

def simulate_step_steer(T: float = 5.0, dt: float = 0.001,
                        delta_step_deg: float = 3.0,
                        p: DynParams = DynParams(),
                        u0: float = 20.0) -> Dict[str, np.ndarray]:
    """
    Simulates a step-steer maneuver to test the vehicle's transient response.

    The vehicle drives at a constant longitudinal speed (u0) and then a small
    steering input (delta_step_deg) is abruptly applied. The function tracks
    how the lateral velocity (v) and yaw rate (r) respond over time.

    Args:
        T (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        delta_step_deg (float): The magnitude of the steering step in degrees.
        p (DynParams): Vehicle physical parameters.
        u0 (float): The initial and constant longitudinal speed in m/s.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the simulation data over time.
    """
    n = int(T / dt)
    # Initialize the state vector: [x, y, psi, u, v, r]
    s = np.array([0.0, 0.0, 0.0, u0, 0.0, 0.0], float)
    delta_step = np.deg2rad(delta_step_deg)
    
    # Dictionary to store the output data for plotting
    out = {"t": [], "u": [], "v": [], "r": [], "psi": [], "delta": []}
    
    for i in range(n):
        # Apply the step steer after 0.5 seconds
        delta = delta_step if i * dt > 0.5 else 0.0
        ax = 0.0  # Hold longitudinal acceleration at zero to maintain constant speed
        
        # Take a single simulation step using the RK4 integrator
        s = step_rk4(s, (delta, ax), p, dt)
        
        # Append the current state to the output dictionary
        out["t"].append(i * dt)
        out["u"].append(s[3])
        out["v"].append(s[4])
        out["r"].append(s[5])
        out["psi"].append(s[2])
        out["delta"].append(delta)
        
    return {k: np.array(v) for k, v in out.items()}

def simulate_skidpad(radius: float = 40.0, T: float = 15.0, dt: float = 0.001,
                     p: DynParams = DynParams()) -> Dict[str, np.ndarray]:
    """
    Simulates a vehicle on a skidpad at a constant radius, increasing speed over time.

    This test validates the model's steady-state cornering behavior and demonstrates
    the relationship between speed, steering angle, and lateral acceleration.

    Args:
        radius (float): The target radius of the skidpad in meters.
        T (float): Total simulation time in seconds.
        dt (float): Time step in seconds.
        p (DynParams): Vehicle physical parameters.

    Returns:
        Dict[str, np.ndarray]: A dictionary containing the simulation data.
    """
    n = int(T / dt)
    # Initial state: [x, y, psi, u, v, r]. Start on the right side of the pad, heading up.
    s = np.array([radius, 0.0, np.pi / 2, 5.0, 0.0, 0.0], float)
    
    # Calculate the feedforward steering angle for the given radius
    L = p.Lf + p.Lr
    delta_ff = np.arctan(L / radius)
    
    # Dictionary to store the output data
    out = {"t": [], "u": [], "ay": [], "r": [], "delta": []}
    
    for i in range(n):
        # Crude speed ramp to reach steady state. This can be improved later with a proper controller.
        ax = 0.5
        delta = delta_ff
        
        # Take a single simulation step
        s = step_rk4(s, (delta, ax), p, dt)
        
        # Calculate lateral acceleration from the tire forces for easy plotting
        Fy_f, Fy_r = linear_tire_forces(s[3], s[4], s[5], delta, p)
        ay = (Fy_f + Fy_r) / p.m
        
        # Append data to the output dictionary
        out["t"].append(i * dt)
        out["u"].append(s[3])
        out["ay"].append(ay)
        out["r"].append(s[5])
        out["delta"].append(delta)
        
    return {k: np.array(v) for k, v in out.items()}
import numpy as np

g = 9.81

class Vehicle:
    def __init__(self, m: float = 800.0, power: float = 300e3, CdA: float = 0.90, 
                 rho: float = 1.225, Crr: float = 0.015, mu_drive: float = 1.2, 
                 mu_brake: float = 1.2, ita_drive: float = 0.90, 
                 v_target: float = 200/3.6):
        """
        Initialises the Vehicle class with its physical properties.

        Args:
            m: Mass of the vehicle in kg
            power: Engine power in W
            CdA: Drag area in m^2
            rho: Air density in kg/m^3
            Crr: Rolling resistance coefficient
            mu_drive: Tyre-road friction coefficient for acceleration
            mu_brake: Tyre-road friction coefficient for braking
            ita_drive: Drivetrain efficiency
            v_target: Target speed in m/s
        """
        self.m = m
        self.power = power
        self.CdA = CdA
        self.rho = rho
        self.Crr = Crr
        self.mu_drive = mu_drive
        self.mu_brake = mu_brake
        self.ita_drive = ita_drive
        self.v_target = v_target

def accel_brake_run(car: Vehicle, dt: float = 0.01):
    """
    Simulates vehicle acceleration to a target speed, followed by braking to a stop.

    Args:
        car: The Vehicle object containing physical properties.
        dt: The time step for the simulation in seconds.

    Returns:
        tuple: A tuple of numpy arrays for time, velocity, distance, and acceleration.
    """
    v, s, t = 0.0, 0.0, 0.0
    T, V, S, A = [], [], [], []

    # --- Acceleration (0 -> v_target)
    while v < car.v_target:
        # Calculate forces acting on the vehicle
        F_drag = 0.5 * car.rho * car.CdA * v*v
        F_rr   = car.Crr * car.m * g
        F_trac = car.mu_drive * car.m * g
        F_power= (car.ita_drive * car.power) / max(v, 1e-6)  # Avoid division by zero at v=0
        F_drive= min(F_trac, F_power)

        # Calculate acceleration and update state variables
        a = (F_drive - F_drag - F_rr) / car.m
        if a < 0: a = 0.0  # Don't slow down in the accel phase

        v += a * dt
        s += v * dt
        t += dt
        T.append(t); V.append(v); S.append(s); A.append(a)
        if t > 120: break  # Safety break

    # --- Braking (v_target -> 0)
    while v > 0:
        # Calculate forces for braking
        F_drag = 0.5 * car.rho * car.CdA * v*v
        F_rr   = car.Crr * car.m * g
        F_brake= car.mu_brake * car.m * g
        a = -(F_brake + F_drag + F_rr) / car.m

        # Update state variables
        v = max(0.0, v + a * dt)
        s += v * dt
        t += dt
        T.append(t); V.append(v); S.append(s); A.append(a)
        if t > 240: break  # Safety break

    return np.array(T), np.array(V), np.array(S), np.array(A)

def to_kmh(v_ms: np.ndarray) -> np.ndarray:
    """Converts velocity from meters per second to kilometers per hour."""
    return v_ms * 3.6

def time_to_speed(T, V, v_target_ms):
    """Finds the time it takes to reach a target velocity."""
    idx = np.where(V >= v_target_ms)[0]
    return float(T[idx[0]]) if idx.size else np.nan

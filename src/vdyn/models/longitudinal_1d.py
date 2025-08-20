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


# --- Small config helper (to serialise Vehicle later)
def vehicle_to_dict(car: Vehicle) -> dict:
    return {
        "m": car.m, "power": car.power, "CdA": car.CdA, "rho": car.rho,
        "Crr": car.Crr, "mu_drive": car.mu_drive, "mu_brake": car.mu_brake,
        "ita_drive": car.ita_drive, "v_target": car.v_target
    }

# --- KPI helpers with light interpolation for precision
def _t_to_reach_speed(T: np.ndarray, V: np.ndarray, v_target_ms: float) -> float:
    idx = np.where(V >= v_target_ms)[0]
    if idx.size == 0:
        return float("nan")
    i = int(idx[0])
    if i == 0:
        return float(T[0])
    v0, v1 = V[i-1], V[i]
    t0, t1 = T[i-1], T[i]
    if v1 == v0:
        return float(t1)
    frac = (v_target_ms - v0) / (v1 - v0)
    return float(t0 + frac * (t1 - t0))

def kpi_0_to_100_kmh(T: np.ndarray, V: np.ndarray) -> float:
    return _t_to_reach_speed(T, V, 27.7777777778)

def kpi_0_to_200_kmh(T: np.ndarray, V: np.ndarray) -> float:
    return _t_to_reach_speed(T, V, 55.5555555556)

def kpi_top_speed_ms(V: np.ndarray) -> float:
    return float(np.max(V))

def kpi_top_speed_kmh(V: np.ndarray) -> float:
    return float(np.max(V) * 3.6)

def kpi_100_to_0_brake_distance(T: np.ndarray, V: np.ndarray, S: np.ndarray) -> float:
    """Distance from first time crossing down through 100 km/h on the *braking* side until stop."""
    v100 = 27.7777777778
    i_peak = int(np.argmax(V))
    if i_peak >= len(V) - 2:
        return float("nan")
    Vb = V[i_peak:]      # braking segment
    Sb = S[i_peak:]
    # find first index where we cross from > v100 to <= v100
    cross = np.where((Vb[:-1] > v100) & (Vb[1:] <= v100))[0]
    if cross.size == 0:
        return float("nan")
    j = int(cross[0] + 1)  # index in Vb/Sb
    v0, v1 = Vb[j-1], Vb[j]
    s0, s1 = Sb[j-1], Sb[j]
    if v1 == v0:
        s_start = s1
    else:
        frac = (v100 - v0) / (v1 - v0)
        s_start = s0 + frac * (s1 - s0)
    s_end = float(S[-1])   # final distance when v reaches 0
    return float(s_end - s_start)

def compute_kpis(T: np.ndarray, V: np.ndarray, S: np.ndarray) -> dict:
    return {
        "t_0_100_s": kpi_0_to_100_kmh(T, V),
        "t_0_200_s": kpi_0_to_200_kmh(T, V),
        "vmax_kmh": kpi_top_speed_kmh(V),
        "brake_100_0_m": kpi_100_to_0_brake_distance(T, V, S),
    }

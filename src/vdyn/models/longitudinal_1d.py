import numpy as np
from dataclasses import dataclass
from vdyn.models.powertrain import Powertrain
g = 9.81

@dataclass
class Vehicle:
    """
    Initialises the Vehicle class with its physical properties.
    """
    m: float = 800.0          # Mass of the vehicle in kg
    power: float = 300e3      # Engine power in W
    CdA: float = 0.90         # Drag area in m^2
    rho: float = 1.225        # Air density in kg/m^3
    Crr: float = 0.015        # Rolling resistance coefficient
    mu_drive: float = 1.2     # Tyre-road friction coefficient for acceleration
    mu_brake: float = 1.2     # Tyre-road friction coefficient for braking
    ita_drive: float = 0.90   # Drivetrain efficiency
    v_target: float = 200/3.6 # Target speed in m/s
    ClA: float = 0.0          # Lift area in m^2 (negative for downforce)
    dCdA_per_ClA: float = 0.0 # Change in drag area per unit lift area (for aerodynamic coupling)

def accel_brake_run(car: Vehicle, dt: float = 0.01, powertrain: Powertrain | None = None):
    """
    Simulates vehicle acceleration to a target speed, followed by braking to a stop.

    Args:
        car: The Vehicle object containing physical properties.
        dt: The time step for the simulation in seconds.

    Returns:
        tuple: A tuple of numpy arrays for time, velocity, distance, and acceleration.
    """
    v, s, t = 0.0, 0.0, 0.0
    T, V, S, A, G, R = [], [], [], [], [], []

    # --- Acceleration (0 -> v_target)

    gear = powertrain.box.launch_gear if powertrain else 0
    while v < car.v_target:
        # Calculate forces acting on the vehicle
        CdA_eff = car.CdA + car.dCdA_per_ClA * car.ClA
        F_drag = 0.5 * car.rho * CdA_eff * v*v
        F_rr   = car.Crr * car.m * g
        N = car.m * g + 0.5 * car.rho * car.ClA * v*v  # Normal force with downforce
        
        if powertrain:
            F_eng, rpm = powertrain.available_drive_force(v, gear)
            if rpm > powertrain.box.shift_rpm and gear < len(powertrain.box.ratios):
                gear += 1
                F_eng, rpm = powertrain.available_drive_force(v, gear)
            elif rpm < powertrain.box.downshift_rpm and gear > 1:
                gear -= 1
                F_eng, rpm = powertrain.available_drive_force(v, gear)
        else:
            rpm = 0.0
            F_eng = (car.ita_drive * car.power) / max(v, 1e-6)

        F_trac = car.mu_drive * N
        F_drive= min(F_trac, F_eng)

        # Calculate acceleration and update state variables
        a = (F_drive - F_drag - F_rr) / car.m
        if a < 0: 
            a = 0.0
            break

        v_prev = v
        v += a * dt
        s += v_prev * dt + 0.5 * a * dt * dt
        t += dt
        T.append(t); V.append(v); S.append(s); A.append(a); G.append(gear); R.append(rpm)
        if t > 120: break  # Safety break

    # --- Braking (v_target -> 0)
    while v > 0:
        # Calculate forces for braking
        CdA_eff = car.CdA + car.dCdA_per_ClA * car.ClA
        F_drag = 0.5 * car.rho * CdA_eff * v*v
        F_rr   = car.Crr * car.m * g
        N = car.m * g + 0.5 * car.rho * car.ClA * v*v
        F_brake= car.mu_brake * N

        a = -(F_brake + F_drag + F_rr) / car.m

        # Update state variables
        v_prev = v
        v = max(0.0, v + a * dt)
        s += v_prev * dt + 0.5 * a * dt * dt
        t += dt

        if powertrain:
        # visual downshift-only trace (no physics change)
            rpm = powertrain.box.engine_rpm(v, gear)
            if gear > 1:
                rpm_down = powertrain.box.engine_rpm(v, gear-1)
                if (rpm < powertrain.box.downshift_rpm) and (rpm_down <= powertrain.box.shift_rpm * 1.02):
                    gear -= 1
                    rpm = rpm_down
        else:
            gear = 0
            rpm  = 0.0

        T.append(t); V.append(v); S.append(s); A.append(a); G.append(gear); R.append(rpm)
        if t > 240: break  # Safety break

    return np.array(T), np.array(V), np.array(S), np.array(A), np.array(G), np.array(R)

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
        "ita_drive": car.ita_drive, "v_target": car.v_target, "ClA": car.ClA, "dCdA_per_ClA": car.dCdA_per_ClA,
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

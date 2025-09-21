from dataclasses import dataclass
import numpy as np

TWOPI = 2.0 * np.pi

@dataclass
class TorqueCurve:
    """
    Represents an engine torque curve as a function of RPM.
    """
    rpm: np.ndarray
    torque_nm: np.ndarray  

    def tq(self, rpm: float) -> float:
        """
        Returns the engine torque at a given RPM using linear interpolation.
        Clamps RPM to the valid range.
        """
        rpm = float(np.clip(rpm, self.rpm[0], self.rpm[-1]))
        return float(np.interp(rpm, self.rpm, self.torque_nm))

@dataclass
class Gearbox:
    """
    Models a vehicle gearbox and driveline.
    """
    ratios: list[float]  # Gear ratios for each gear
    final_drive: float   # Final drive ratio         
    wheel_radius_m: float  # Wheel radius in meters       
    driveline_eff: float = 0.92 # Driveline efficiency (0-1)
    shift_rpm: float = 12000.0  # RPM to upshift
    downshift_rpm: float = 3000.0  # RPM to downshift
    launch_gear: int = 1  # Starting gear

    def wheel_speed_rps(self, v_ms: float) -> float:
        """
        Calculates wheel rotational speed in radians per second from vehicle speed.
        """
        return v_ms / max(self.wheel_radius_m, 1e-6)

    def engine_rpm(self, v_ms: float, gear: int) -> float:
        """
        Calculates engine RPM for a given vehicle speed and gear.
        """
        if gear < 1: gear = 1
        if gear > len(self.ratios): gear = len(self.ratios)
        r = self.ratios[gear-1] * self.final_drive
        return self.wheel_speed_rps(v_ms) * r * 60.0 / TWOPI

    def wheel_torque_from_engine(self, eng_tq_nm: float, gear: int) -> float:
        """
        Converts engine torque to wheel torque for a given gear.
        """
        r = self.ratios[gear-1] * self.final_drive
        return eng_tq_nm * r * self.driveline_eff

    def drive_force_from_engine(self, eng_tq_nm: float, gear: int) -> float:
        """
        Calculates the drive force at the wheels from engine torque and gear.
        """
        # F = T_wheel / Rwheel
        return self.wheel_torque_from_engine(eng_tq_nm, gear) / max(self.wheel_radius_m, 1e-6)

@dataclass
class Powertrain:
    """
    Combines an engine torque curve with a gearbox to model a vehicle powertrain.
    """
    curve: TorqueCurve
    box: Gearbox

    engine_brake_coeff_nm_per_rpm: float = 0.03
    engine_brake_max_torque_nm: float = 400.0
    shift_time_s: float = 0.00

    def available_drive_force(self, v_ms: float, gear: int) -> tuple[float, float]:
        """
        Returns the available drive force and engine RPM for a given speed and gear.
        """
        rpm = self.box.engine_rpm(v_ms, gear)
        tq  = self.curve.tq(rpm)
        F   = self.box.drive_force_from_engine(tq, gear)
        return F, rpm
    
    def engine_brake_force(self, v_ms: float, gear: int) -> tuple[float, float, float]:
        """
        Returns the engine braking force, engine RPM, and engine brake torque for a given speed and gear.
        """
        if gear < 1:
            return 0.0, 0.0, 0.0
        rpm = self.box.engine_rpm(v_ms, gear)
        T_eb = min(self.engine_brake_coeff_nm_per_rpm * rpm, self.engine_brake_max_torque_nm)
        F_eb = self.box.drive_force_from_engine(T_eb, gear)
        return F_eb, rpm, T_eb
    
def default_highrev_curve() -> TorqueCurve:
    """
    Returns a default high-revving engine torque curve for simulation.
    """
    rpm = np.array([1000, 4000, 8000, 11000, 13000, 14000], dtype=float)
    tq  = np.array([120,   220,   260,    250,    230,    210], dtype=float)
    return TorqueCurve(rpm, tq)

def iame_x30_curve(scale: float = 1.0) -> TorqueCurve:
    """
    Approximate IAME X30 (EU) torque curve (no power valve).
    Broad, smooth plateau: T_max ≈ 19.5 Nm near 10.5-11k rpm;
    ~30 hp (~22.37 kW) around ~12k rpm -> T ≈ 17.8 Nm. Rev limit 16k.
    data taken from: https://iameengines.com/product/x30-spec-eu/

    Parameters
    ----------
    scale : float
        Global torque scale for quick telemetry fitting (1.0 = spec-like).

    Returns
    -------
    TorqueCurve
        Discrete (rpm, Nm) curve; your TorqueCurve interpolates between points.
    """
    rpm = np.array([ 4000,  6000,  8000,  9000, 10000, 10500, 11000, 12000, 13000, 14000, 15000, 16000 ], float)
    tq  = np.array([  8.0,  12.0,  17.0,  18.5,  19.2,  19.5,  19.0,  17.8,  16.5,  14.5,  12.5,  10.5 ], float)
    return TorqueCurve(rpm, scale * tq)

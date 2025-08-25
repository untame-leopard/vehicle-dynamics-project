from dataclasses import dataclass
import numpy as np

TWOPI = 2.0 * np.pi

@dataclass
class TorqueCurve:
    rpm: np.ndarray       
    torque_nm: np.ndarray  

    def tq(self, rpm: float) -> float:
        # clamp + linear interp
        rpm = float(np.clip(rpm, self.rpm[0], self.rpm[-1]))
        return float(np.interp(rpm, self.rpm, self.torque_nm))

@dataclass
class Gearbox:
    ratios: list[float]       
    final_drive: float            
    wheel_radius_m: float         
    driveline_eff: float = 0.92
    shift_rpm: float = 12000.0
    launch_gear: int = 1

    def wheel_speed_rps(self, v_ms: float) -> float:
        return v_ms / max(self.wheel_radius_m, 1e-6)

    def engine_rpm(self, v_ms: float, gear: int) -> float:
        if gear < 1: gear = 1
        if gear > len(self.ratios): gear = len(self.ratios)
        r = self.ratios[gear-1] * self.final_drive
        return self.wheel_speed_rps(v_ms) * r * 60.0 / TWOPI

    def wheel_torque_from_engine(self, eng_tq_nm: float, gear: int) -> float:
        r = self.ratios[gear-1] * self.final_drive
        return eng_tq_nm * r * self.driveline_eff

    def drive_force_from_engine(self, eng_tq_nm: float, gear: int) -> float:
        # F = T_wheel / Rwheel
        return self.wheel_torque_from_engine(eng_tq_nm, gear) / max(self.wheel_radius_m, 1e-6)

@dataclass
class Powertrain:
    curve: TorqueCurve
    box: Gearbox

    def available_drive_force(self, v_ms: float, gear: int) -> tuple[float, float]:
        rpm = self.box.engine_rpm(v_ms, gear)
        tq  = self.curve.tq(rpm)
        F   = self.box.drive_force_from_engine(tq, gear)
        return F, rpm
    
def default_highrev_curve() -> TorqueCurve:
    rpm = np.array([1000, 4000, 8000, 11000, 13000, 14000], dtype=float)
    tq  = np.array([120,   220,   260,    250,    230,    210], dtype=float)
    return TorqueCurve(rpm, tq)

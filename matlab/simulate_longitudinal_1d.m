function out = simulate_longitudinal(car, dt)
% car: struct with fields m, power, CdA, rho, Crr, mu_drive, mu_brake,
%      ita_drive, v_target, ClA, dCdA_per_ClA
g = 9.81; v = 0; s = 0; t = 0;
T=[]; V=[]; S=[]; A=[];

% --- Accel to target
while v < car.v_target
    CdA_eff = car.CdA + car.dCdA_per_ClA * car.ClA;
    F_drag = 0.5 * car.rho * CdA_eff * v^2;
    F_rr   = car.Crr * car.m * g;
    N      = car.m * g + 0.5 * car.rho * car.ClA * v^2;
    F_trac_cap  = car.mu_drive * N;
    F_power_cap = (car.ita_drive * car.power) / max(v, 1e-6);
    F_drive = min(F_trac_cap, F_power_cap);

    a = (F_drive - F_drag - F_rr) / car.m;
    if a < 0, a = 0; break; end

    v_prev = v;
    v = v + a*dt;
    s = s + v_prev*dt + 0.5*a*dt*dt;
    t = t + dt;

    T(end+1)=t; V(end+1)=v; S(end+1)=s; A(end+1)=a; %#ok<AGROW>
    if t > 120, break; end
end

% --- Brake to stop
while v > 0
    CdA_eff = car.CdA + car.dCdA_per_ClA * car.ClA;
    F_drag = 0.5 * car.rho * CdA_eff * v^2;
    F_rr   = car.Crr * car.m * g;
    N      = car.m * g + 0.5 * car.rho * car.ClA * v^2;
    F_brake_cap = car.mu_brake * N;

    a = -(F_brake_cap + F_drag + F_rr) / car.m;

    v_prev = v;
    v = max(0.0, v + a*dt);
    s = s + v_prev*dt + 0.5*a*dt*dt;
    t = t + dt;

    T(end+1)=t; V(end+1)=v; S(end+1)=s; A(end+1)=a; %#ok<AGROW>
    if t > 240, break; end
end

out.T = T(:); out.V = V(:); out.S = S(:); out.A = A(:);
end

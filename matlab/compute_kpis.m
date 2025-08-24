function k = compute_kpis(T, V, S)
k.t_0_100_s = t_to_reach_speed(T, V, 27.7777777778);
k.t_0_200_s = t_to_reach_speed(T, V, 55.5555555556);
k.vmax_kmh  = max(V) * 3.6;
k.brake_100_0_m = brake_distance_100_to_0(T, V, S);
end

function t = t_to_reach_speed(T, V, v_target)
idx = find(V >= v_target, 1, 'first');
if isempty(idx), t = NaN; return; end
if idx == 1, t = T(1); return; end
v0=V(idx-1); v1=V(idx); t0=T(idx-1); t1=T(idx);
if v1==v0, t=t1; else, t = t0 + (v_target - v0) * (t1 - t0) / (v1 - v0); end
end

function d = brake_distance_100_to_0(T, V, S)
v100 = 27.7777777778;
[~, i_peak] = max(V);
Vb = V(i_peak:end); Sb = S(i_peak:end);
cross = find(Vb(1:end-1) > v100 & Vb(2:end) <= v100, 1, 'first');
if isempty(cross), d = NaN; return; end
j = cross + 1;
v0=Vb(j-1); v1=Vb(j); s0=Sb(j-1); s1=Sb(j);
if v1==v0, s_start=s1; else, s_start = s0 + (v100 - v0) * (s1 - s0) / (v1 - v0); end
d = S(end) - s_start;
end

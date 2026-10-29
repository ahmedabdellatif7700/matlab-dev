function optical = modulate(cfg,voltage)
%MODULATE Balanced single-arm field transfer, including common optical phase.
if isinf(cfg.mzm_bandwidth_hz)
    drive = voltage;
else
    drive = real(filter_envelope(voltage,cfg.internal_sample_rate_hz, ...
        cfg.filter_guard_samples,@(f) 1./(1+1j*f/cfg.mzm_bandwidth_hz)));
end
phase = pi*(cfg.mzm_bias_v+drive)/cfg.mzm_vpi_v;
field = (1+exp(1j*phase))/2;
optical = struct("modulated_field",field,"lo_field",1, ...
    "drive_voltage",drive,"power_units","normalized; not calibrated optical watts");
end

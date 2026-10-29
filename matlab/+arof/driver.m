function voltage = driver(cfg,input)
%DRIVER Real IF driver; compressed option is a declared identification model.
voltage = cfg.driver_voltage_gain*input;
if cfg.driver_model == "linear"
    return;
end
assert(any(cfg.driver_model == ["rapp" "rapp_envelope"]),"Unknown driver model.");
assert(isfinite(cfg.driver_saturation_v) && cfg.driver_saturation_v>0,"Invalid saturation.");
p = cfg.driver_smoothness;
assert(p>0,"Driver smoothness must be positive.");
if cfg.driver_model == "rapp_envelope"
    analytic = hilbert(voltage);
    voltage = real(analytic./(1+(abs(analytic)/cfg.driver_saturation_v).^(2*p)).^(1/(2*p)));
else
    voltage = voltage./(1+(abs(voltage)/cfg.driver_saturation_v).^(2*p)).^(1/(2*p));
end
end

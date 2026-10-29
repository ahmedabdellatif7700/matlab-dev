function detected = photodetect(cfg,optical)
%PHOTODETECT Analytic square-law beat term around the positive RF carrier.
% DC and optical-frequency components are excluded; no 60 GHz sampled tone.
beat = 2*optical.lo_field.*conj(optical.modulated_field);
if ~isinf(cfg.pd_bandwidth_hz)
    beat = arof.filter_envelope(beat,cfg.internal_sample_rate_hz, ...
        cfg.filter_guard_samples,@(f) 1./(1+1j*(cfg.rf_carrier_hz+f)/cfg.pd_bandwidth_hz));
end
detected = struct("rf_envelope",beat,"rf_carrier_hz",cfg.rf_carrier_hz);
end

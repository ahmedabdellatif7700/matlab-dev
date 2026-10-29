function detected = photodetect(cfg,optical)
%PHOTODETECT Analytic square-law beat term around the positive RF carrier.
% DC and optical-frequency components are excluded; no 60 GHz sampled tone.
power_dbm=NaN; responsivity=1;
% JSON round trips decode the disabled NaN setting as an empty value.
if isfield(cfg,'received_optical_power_dbm') && ~isempty(cfg.received_optical_power_dbm) && ...
        ~isnan(cfg.received_optical_power_dbm)
    % ROP is defined as total time-average two-tone optical power at the PD.
    % The optical-carrier cross term averages out on an optical power meter.
    assert(isscalar(cfg.received_optical_power_dbm) && isfinite(cfg.received_optical_power_dbm));
    total=mean(abs(optical.modulated_field).^2)+mean(abs(optical.lo_field).^2);
    watts=1e-3*10^(cfg.received_optical_power_dbm/10);
    scale=sqrt(watts/total);
    optical.modulated_field=scale*optical.modulated_field;
    optical.lo_field=scale*optical.lo_field;
    power_dbm=10*log10((mean(abs(optical.modulated_field).^2)+ ...
        mean(abs(optical.lo_field).^2))/1e-3);
end
if isfield(cfg,'pd_responsivity_a_w'), responsivity=cfg.pd_responsivity_a_w; end
assert(isscalar(responsivity) && isfinite(responsivity) && responsivity>0);
beat = 2*responsivity*optical.lo_field.*conj(optical.modulated_field);
if ~isinf(cfg.pd_bandwidth_hz)
    beat = arof.filter_envelope(beat,cfg.internal_sample_rate_hz, ...
        cfg.filter_guard_samples,@(f) 1./(1+1j*(cfg.rf_carrier_hz+f)/cfg.pd_bandwidth_hz));
end
detected = struct("rf_envelope",beat,"rf_carrier_hz",cfg.rf_carrier_hz, ...
    "received_optical_power_dbm",power_dbm);
end

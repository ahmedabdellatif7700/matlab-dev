function optical = propagate(cfg,optical)
%PROPAGATE Each optical band sees its own dispersion phase.
% Common beta0/beta1 removed. LO offset is relative to the modulated laser.
length_m = cfg.fiber_length_km*1000;
beta2 = -(cfg.dispersion_ps_nm_km*1e-6)*cfg.modulated_wavelength_m^2/(2*pi*299792458);
field_gain = 10^((-cfg.attenuation_db_km*cfg.fiber_length_km+cfg.ideal_edfa_gain_db)/20);
phase = @(f) exp(-1j*beta2*length_m*(2*pi*f).^2/2);
optical.modulated_field = field_gain*arof_history.filter_envelope(optical.modulated_field, ...
    cfg.internal_sample_rate_hz,cfg.filter_guard_samples,phase);
optical.lo_field = field_gain*optical.lo_field*phase(cfg.rf_carrier_hz);
optical.beta2_s2_m = beta2;
end

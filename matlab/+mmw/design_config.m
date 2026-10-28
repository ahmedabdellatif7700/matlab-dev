function cfg = design_config()
%DESIGN_CONFIG Four-family NN efficiency study, independent of old test data.
paths = mmw.scenario_config();
cfg = struct("root",paths.root,"python_exe",paths.python_exe);
cfg.rf_carrier_hz = 60e9;
cfg.symbol_rate_hz = 1e9;
cfg.samples_per_symbol = 8;
cfg.rrc_rolloff = 0.25;
cfg.rrc_span_symbols = 24;
cfg.pa_smoothness = 2;
cfg.optical_wavelength_m = 1550e-9;
cfg.dispersion_ps_nm_km = 17;
cfg.fft_guard_samples = 4096;
cfg.output_root = fullfile(cfg.root, "datasets", "design_study");
cfg.run_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
cfg.discovery_seed = 501;
cfg.confirmation_seeds = [1201 1202 1203];
cfg.discovery_counts = [10000 10000];
cfg.confirmation_counts = [18000 6000 50000];
cfg.tap_offsets = [0 -1]; cfg.equalizer_taps = numel(cfg.tap_offsets);
% Network, activation and receiver-search decisions: python/config.py.
cfg.visible = "off";
end

function cfg = suite_config(scenario)
%SUITE_CONFIG Research scenarios, with independent settings from hardware.
cfg = mmw.scenario_config();
cfg.scenario = scenario;
cfg.output_root = fullfile(cfg.root, "datasets", "scenarios", "scenario" + scenario);
cfg.train_symbols = 6000;
cfg.validation_symbols = 1800;
cfg.test_symbols = 15000;
cfg.nn_hidden_size = 24;
cfg.nn_epochs = 400;
cfg.nn_patience = 50;
cfg.nn_lbfgs_steps = 15;
cfg.alpha_learning_rate = 0.001;
cfg.polynomial_orders = [1 2 3 5];
cfg.max_polynomial_features = 1000;
cfg.activations = ["sigmoid" "multilevel"];
cfg.baseline_name = "Before EQ";
cfg.mzm_drive = [0.2 0.6 1.0 1.4 1.8];
cfg.mzm_snr_db = 30;
cfg.mzm_vpi = 4.4;
cfg.mzm_bias_over_vpi = 0.5;
cfg.report_condition = "qam16_mem0_drive100";
cfg.memory_coefficient = 0.2*exp(1j*pi/5);
cfg.memory_coefficient = [real(cfg.memory_coefficient) imag(cfg.memory_coefficient)];
cfg.tap_offsets = [0 -1];
cfg.optical_wavelength_m = 1550e-9;
cfg.dispersion_ps_nm_km = 17;
cfg.fft_guard_samples = 4096;
cfg.fiber_lengths_km = [0 0.5 0.9 1.02 1.15 1.5];
if scenario == 3
    cfg.tap_offsets = [0 1 -1]; cfg.equalizer_taps = 3;
    cfg.report_condition = "DSB_pa1_L090";
elseif scenario ~= 2
    error("This scenario is not implemented yet.");
end
end

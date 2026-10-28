function cfg = design_config()
%DESIGN_CONFIG Four-family NN efficiency study, independent of old test data.
cfg = mmw.suite_config(2);
cfg.output_root = fullfile(cfg.root, "datasets", "design_study");
cfg.run_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
cfg.discovery_seed = 501;
cfg.confirmation_seeds = [1201 1202 1203];
cfg.discovery_counts = [10000 10000];
cfg.confirmation_counts = [18000 6000 50000];
cfg.tap_offsets = [0 -1]; cfg.equalizer_taps = 2;
cfg.nn_epochs = 600; cfg.nn_patience = 70; cfg.nn_polish_steps = 12;
cfg.ber_targets = [1e-4 5e-4 1e-3 3.8e-3 1e-2];
cfg.required_cost_ratio = 1.5;
cfg.max_polynomial_features = 2048;
cfg.polynomial_orders = [1 2 3 5 7];
cfg.bit_polynomial_orders = [1 2 3 5];
cfg.kernel_centers = [16 64 256];
cfg.visible = "off";
end

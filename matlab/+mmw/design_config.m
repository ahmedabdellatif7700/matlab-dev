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
% Network, activation and receiver-search decisions: python/config.py.
cfg.visible = "off";
end

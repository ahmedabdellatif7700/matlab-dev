function cfg = scenario_config()
%SCENARIO_CONFIG Settings for the isolated RF driver study.
cfg.root = fileparts(fileparts(fileparts(mfilename("fullpath"))));
cfg.profile = "study";
cfg.visible = "off";
cfg.output_root = fullfile(cfg.root, "datasets", "scenarios", "pa_single_carrier");
cfg.run_id = ""; % Empty creates a timestamp; existing runs are refused.
cfg.python_exe = string(getenv("MMW_PYTHON"));
if strlength(cfg.python_exe) == 0
    candidate = fullfile(getenv("HOME"), "miniconda3", "envs", ...
        "machine-learning", "bin", "python");
    if isfile(candidate)
        cfg.python_exe = string(candidate);
    else
        cfg.python_exe = "python3";
    end
end
cfg.mod_order = 16;
cfg.rf_carrier_hz = 60e9; % Complex-envelope carrier label, not a sampled RF tone.
cfg.symbol_rate_hz = 1e9;
cfg.samples_per_symbol = 8;
cfg.rrc_rolloff = 0.25;
cfg.rrc_span_symbols = 24; % Reduce finite-filter ISI below the measured PA effects.
cfg.equalizer_taps = 2; % Current and previous sample for every model.
cfg.pa_smoothness = 2;
cfg.pa_phase_limit_deg = 20;
cfg.backoff_db = [-3 0 3 6 9 12];
% Ideal linear matched-filter SNR at zero-dB backoff. Noise stays fixed
% during a drive sweep; effective linear-link SNR is reference SNR - backoff.
cfg.reference_snr_db = [24 30 36];
cfg.seeds = [42 43 44];
cfg.train_symbols = 12000;
cfg.validation_symbols = 3000;
cfg.test_symbols = 30000;
cfg.nn_epochs = 250;
cfg.nn_patience = 35;
cfg.nn_batch_size = 512;
cfg.nn_learning_rate = 0.003;
cfg.nn_hidden_size = 12;
cfg.ridge_grid = [0 1e-8 1e-6 1e-4 1e-2 1];
cfg.ber_target = 3.8e-3; % Literature reference, not an implemented FEC decoder.
cfg.constellation_backoff_db = 0;
cfg.constellation_snr_db = 30;
end

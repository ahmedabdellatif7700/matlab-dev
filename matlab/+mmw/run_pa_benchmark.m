function run_dir = run_pa_benchmark(cfg)
%RUN_PA_BENCHMARK Export conditions, fit receivers, and plot held-out results.
[status, branch] = system(sprintf('git -C "%s" branch --show-current', cfg.root));
assert(status == 0 && strcmp(strtrim(branch), "codex/development"), ...
    "Run this study on codex/development only.");
if cfg.profile == "smoke"
    cfg.seeds = cfg.seeds(1);
    cfg.train_symbols = 1200;
    cfg.validation_symbols = 400;
    cfg.test_symbols = 2000;
    cfg.backoff_db = [0 6];
    cfg.reference_snr_db = 30;
    cfg.nn_epochs = 30;
    cfg.nn_patience = 10;
elseif cfg.profile ~= "study"
    error("Unknown profile: %s", cfg.profile);
end
assert(cfg.mod_order == 16, "Scenario 1 is configured for 16-QAM.");
assert(cfg.equalizer_taps == 2 && cfg.nn_hidden_size == 12, ...
    "Scenario 1 uses the shared two-symbol window and 4-12-2 NN.");
if strlength(cfg.run_id) == 0
    cfg.run_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
end
run_dir = fullfile(cfg.output_root, cfg.run_id);
assert(~isfolder(run_dir), "Run directory already exists: %s", run_dir);
mkdir(run_dir);
source_dir = fullfile(run_dir, "source");
mkdir(source_dir);
copyfile(fullfile(cfg.root, "matlab"), fullfile(source_dir, "matlab"));
copyfile(fullfile(cfg.root, "python", "scenario_benchmark.py"), source_dir);
[~, revision] = system(sprintf('git -C "%s" rev-parse HEAD', cfg.root));
[~, changes] = system(sprintf('git -C "%s" status --porcelain', cfg.root));
cfg.git_revision = string(strtrim(revision));
cfg.git_dirty = ~isempty(strtrim(changes));
save(fullfile(run_dir, "configuration.mat"), "cfg");
write_text(fullfile(run_dir, "configuration.json"), jsonencode(cfg, "PrettyPrint", true));
conditions = table([false; false], [0; 0], [Inf; 30], ...
    ["linear_noiseless"; "linear_awgn"], ...
    'VariableNames', {'PA', 'Backoff_dB', 'Reference_SNR_dB', 'Condition'});
for snr = cfg.reference_snr_db
    for backoff = cfg.backoff_db
        name = sprintf("pa_ibo_%+03d_snr_%02d", backoff, snr);
        conditions(end+1, :) = {true, backoff, snr, string(name)}; %#ok<AGROW>
    end
end
writetable(conditions, fullfile(run_dir, "conditions.csv"));
script = fullfile(cfg.root, "python", "scenario_benchmark.py");
for seed = cfg.seeds
    for condition = 1:height(conditions)
        row = conditions(condition, :);
        condition_dir = fullfile(run_dir, sprintf("seed_%d", seed), row.Condition);
        mkdir(condition_dir);
        data = mmw.make_dataset(cfg, seed, row.PA, row.Backoff_dB, row.Reference_SNR_dB);
        data.configuration_json = jsonencode(cfg);
        dataset_path = fullfile(condition_dir, "dataset.mat");
        save(dataset_path, "-struct", "data", "-v7");
        fprintf("\nSeed %d | %s | %d/%d conditions\n", seed, row.Condition, ...
            condition, height(conditions));
        command = sprintf(['env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ' ...
            '"%s" -B "%s" --dataset "%s" --output "%s"'], ...
            cfg.python_exe, script, dataset_path, condition_dir);
        [status, output] = system(command, "-echo");
        write_text(fullfile(condition_dir, "console.txt"), output);
        assert(status == 0, "Receiver benchmark failed; see %s", condition_dir);
    end
end
mmw.plot_results(run_dir, cfg);
write_text(fullfile(run_dir, "COMPLETE"), "All conditions completed successfully.");
end

function write_text(path, content)
[fid, message] = fopen(path, "w");
assert(fid >= 0, "Cannot open %s: %s", path, message);
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%s\n", content);
end

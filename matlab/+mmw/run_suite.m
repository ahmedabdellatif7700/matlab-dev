function run_dir = run_suite(cfg)
%RUN_SUITE Generate, fit, verify, plot, and archive one research scenario.
[status, branch] = system(sprintf('git -C "%s" branch --show-current', cfg.root));
assert(status == 0 && strcmp(strtrim(branch), "codex/development"), "Use codex/development only.");
if cfg.profile == "smoke"
    cfg.seeds = cfg.seeds(1); cfg.train_symbols = 500;
    cfg.validation_symbols = 200; cfg.test_symbols = 600;
    cfg.nn_epochs = 20; cfg.nn_patience = 10; cfg.nn_lbfgs_steps = 2;
elseif cfg.profile ~= "study"
    error("Unknown study profile.");
end
conditions = mmw.suite_conditions(cfg);
if cfg.profile == "smoke"
    conditions = conditions([1 height(conditions)], :);
end
if strlength(cfg.run_id) == 0
    cfg.run_id = string(datetime("now", "Format", "yyyyMMdd_HHmmss_SSS"));
end
run_dir = fullfile(cfg.output_root, cfg.run_id);
assert(~isfolder(run_dir), "Refusing to overwrite %s", run_dir);
mkdir(run_dir); source = fullfile(run_dir, "source"); mkdir(source);
copyfile(fullfile(cfg.root, "matlab", "+mmw"), fullfile(source, "+mmw"));
copyfile(fullfile(cfg.root, "python", "research_suite.py"), source);
copyfile(fullfile(cfg.root, "python", "scenario_benchmark.py"), source);
copyfile(fullfile(cfg.root, "python", "config.py"), source);
copyfile(fullfile(cfg.root, "python", "activations"), fullfile(source, "activations"));
copyfile(fullfile(cfg.root, "python", "hardware"), fullfile(source, "hardware"));
[~, revision] = system(sprintf('git -C "%s" rev-parse HEAD', cfg.root));
cfg.git_revision = string(strtrim(revision)); cfg.validated_before_commit = true;
save(fullfile(run_dir, "configuration.mat"), "cfg");
write_text(fullfile(run_dir, "configuration.json"), jsonencode(cfg, "PrettyPrint", true));
writetable(conditions, fullfile(run_dir, "conditions.csv"));
for seed = cfg.seeds
    for c = 1:height(conditions)
        condition = table2struct(conditions(c, :));
        folder = fullfile(run_dir, sprintf("seed_%d", seed), condition.Condition);
        mkdir(folder);
        data = mmw.make_suite_dataset(cfg, seed, condition);
        data.configuration_json = jsonencode(cfg); data.condition_json = jsonencode(condition);
        dataset = fullfile(folder, "dataset.mat"); save(dataset, "-struct", "data", "-v7");
        fprintf("\nScenario %d | seed %d | %s | %d/%d\n", cfg.scenario, seed, condition.Condition, c, height(conditions));
        command = sprintf(['env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 ' ...
            '"%s" -B "%s" --dataset "%s" --output "%s"'], cfg.python_exe, ...
            fullfile(source, "research_suite.py"), dataset, folder);
        [status, output] = system(command, "-echo");
        write_text(fullfile(folder, "console.txt"), output);
        assert(status == 0, "Receiver failed; see %s", folder);
    end
end
mmw.plot_suite_results(run_dir, cfg);
if cfg.scenario == 3, mmw.plot_optical_response(cfg, run_dir); end
if cfg.scenario == 4, mmw.plot_multiuser_spectrum(cfg, run_dir); end
if cfg.scenario == 5 && cfg.profile == "study", mmw.plot_ofdm_diagnostics(cfg, run_dir); end
write_text(fullfile(run_dir, "COMPLETE"), "All operating points and independent metric checks passed.");
end

function write_text(path, content)
[fid, message] = fopen(path, "w"); assert(fid >= 0, "%s", message);
cleanup = onCleanup(@() fclose(fid)); fprintf(fid, "%s\n", content);
end

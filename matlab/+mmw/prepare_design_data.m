function run_dir = prepare_design_data(cfg, cases, phase, run_dir)
%PREPARE_DESIGN_DATA Discovery files contain no test data at all.
if nargin < 4
    run_dir = fullfile(cfg.output_root, cfg.run_id);
    assert(~isfolder(run_dir), "Refusing to overwrite %s", run_dir);
    mkdir(run_dir); mkdir(fullfile(run_dir, "source"));
    copyfile(fullfile(cfg.root, "python", "tuning"), fullfile(run_dir, "source", "tuning"));
    copyfile(fullfile(cfg.root, "python", "tune_equalizers.py"), fullfile(run_dir, "source"));
    copyfile(fullfile(cfg.root, "matlab", "+mmw"), fullfile(run_dir, "source", "+mmw"));
    copyfile(fullfile(cfg.root, "python", "extend_design_search.py"), fullfile(run_dir, "source"));
    copyfile(fullfile(cfg.root, "matlab", "main.m"), fullfile(run_dir, "source"));
    writetable(cases, fullfile(run_dir, "candidates.csv"));
    save(fullfile(run_dir, "configuration.mat"), "cfg");
    write_text(fullfile(run_dir, "configuration.json"), jsonencode(cfg, "PrettyPrint", true));
end
if phase == "discovery"
    seeds = cfg.discovery_seed; counts = cfg.discovery_counts;
    names = ["train" "validation"];
else
    seeds = cfg.confirmation_seeds; counts = cfg.confirmation_counts;
    names = ["train" "validation" "test"];
end
for seed = seeds
    for c = 1:height(cases)
        condition = table2struct(cases(c, :)); data = struct();
        for p = 1:numel(names)
            [rx, tx] = mmw.design_burst(cfg, condition, counts(p), seed+100000*p);
            offsets = cfg.tap_offsets;
            keep = (1-min(offsets):counts(p)-max(offsets)).';
            x = zeros(numel(keep), 2*numel(offsets), "single");
            for tap = 1:numel(offsets)
                value = rx(keep+offsets(tap));
                x(:, 2*tap-1) = single(real(value)); x(:, 2*tap) = single(imag(value));
            end
            data.("X_"+names(p)) = x;
            data.("Y_"+names(p)) = single([real(tx(keep)) imag(tx(keep))]);
        end
        data.ideal_symbols = qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray");
        data.ideal_bits = de2bi((0:condition.ModOrder-1).', log2(condition.ModOrder), "left-msb");
        data.condition_json = jsonencode(condition); data.configuration_json = jsonencode(cfg);
        folder = fullfile(run_dir, phase, sprintf("seed_%d",seed), condition.ID);
        assert(~isfolder(folder), "Dataset already exists: %s", folder); mkdir(folder);
        save(fullfile(folder,"dataset.mat"), "-struct", "data", "-v7");
    end
end
end

function write_text(path, content)
[fid, msg] = fopen(path, "w"); assert(fid >= 0, "%s", msg);
cleanup = onCleanup(@() fclose(fid)); fprintf(fid, "%s\n", content);
end

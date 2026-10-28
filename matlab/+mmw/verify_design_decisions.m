function count = verify_design_decisions(run_dir)
%VERIFY_DESIGN_DECISIONS Independently recompute Python BER in MATLAB.
metrics = readtable(fullfile(run_dir,"confirmation_metrics.csv"), "TextType","string");
files = dir(fullfile(run_dir,"confirmation","seed_*","*","decisions.mat"));
count = 0;
for f = 1:numel(files)
    data = load(fullfile(files(f).folder,files(f).name));
    [parent, case_id] = fileparts(files(f).folder);
    [~, seed_id] = fileparts(parent); seed = str2double(extractAfter(seed_id,"seed_"));
    for j = 1:numel(data.names)
        decisions = squeeze(data.predictions(j,:,:));
        ber = nnz(decisions ~= data.truth)/numel(data.truth);
        row = metrics.case == string(case_id) & metrics.seed == seed ...
            & metrics.method == string(data.names{j});
        assert(nnz(row) == 1, "Expected unique metric row.");
        assert(abs(ber-metrics.ber(row)) < 1e-12, "MATLAB/Python BER mismatch.");
        count = count+1;
    end
end
assert(count == height(metrics), "Missing decision files or receiver rows.");
fprintf("Independently verified %d BER rows in MATLAB.\n",count);
end

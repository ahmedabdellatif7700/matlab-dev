function pooled = plot_suite_results(run_dir, cfg)
%PLOT_SUITE_RESULTS Verify all Python predictions, pool replicates, and plot.
conditions = readtable(fullfile(run_dir, "conditions.csv"), "TextType", "string");
all = table();
for seed = cfg.seeds
    for c = 1:height(conditions)
        folder = fullfile(run_dir, sprintf("seed_%d", seed), conditions.Condition(c));
        rows = readtable(fullfile(folder, "metrics.csv"), "TextType", "string");
        result = load(fullfile(folder, "results.mat"));
        source = load(fullfile(folder, "dataset.mat"), "Y_test");
        reference = complex(double(source.Y_test(:, 1:2:end)), double(source.Y_test(:, 2:2:end)));
        assert(isequal(reference, result.reference), "Test reference mismatch.");
        names = string(result.method_names(:));
        for r = 1:height(rows)
            m = find(names == rows.method(r)); assert(isscalar(m), "Ambiguous receiver.");
            prediction = result.predictions(:, :, m); target = reference;
            if rows.user(r) > 0
                prediction = prediction(:, rows.user(r)); target = target(:, rows.user(r));
            end
            bits = qamdemod(target(:), conditions.ModOrder(c), "gray", "OutputType", "bit");
            detected = qamdemod(prediction(:), conditions.ModOrder(c), "gray", "OutputType", "bit");
            assert(nnz(bits ~= detected) == rows.bit_errors(r), "MATLAB/Python BER mismatch.");
            assert(numel(bits) == rows.bit_count(r), "Bit count mismatch.");
            evm = 100*norm(prediction(:)-target(:))/norm(target(:));
            assert(abs(evm-rows.evm_percent(r)) < 1e-8, "MATLAB/Python EVM mismatch.");
        end
        rows.Seed = repmat(seed, height(rows), 1);
        for name = ["Condition" "Group" "Axis" "XLabel" "ModOrder"]
            rows.(name) = repmat(conditions.(name)(c), height(rows), 1);
        end
        all = [all; rows]; %#ok<AGROW>
    end
end
writetable(all, fullfile(run_dir, "all_metrics.csv"));
[g, condition, method, user] = findgroups(all.Condition, all.method, all.user);
pooled = table(condition, method, user, 'VariableNames', {'Condition', 'method', 'user'});
for name = ["Group" "Axis" "XLabel" "ModOrder" "parameters" "real_multiplications" "activation_evaluations"]
    pooled.(name) = splitapply(@(x) x(1), all.(name), g);
end
for name = ["bit_errors" "bit_count" "error_energy" "reference_energy"]
    pooled.(name) = splitapply(@sum, all.(name), g);
end
pooled.evm_percent = 100*sqrt(pooled.error_energy./pooled.reference_energy);
pooled.ber = pooled.bit_errors./pooled.bit_count;
pooled.ber_lower_95 = zeros(height(pooled), 1); pooled.ber_upper_95 = ones(height(pooled), 1);
for r = 1:height(pooled)
    e = pooled.bit_errors(r); n = pooled.bit_count(r);
    if e > 0, pooled.ber_lower_95(r) = betaincinv(0.025, e, n-e+1); end
    if e < n, pooled.ber_upper_95(r) = betaincinv(0.975, e+1, n-e); end
end
pooled.target_status = repmat("inconclusive", height(pooled), 1);
pooled.target_status(pooled.ber_upper_95 < cfg.ber_target) = "meets reference";
pooled.target_status(pooled.ber_lower_95 > cfg.ber_target) = "above reference";
writetable(pooled, fullfile(run_dir, "pooled_metrics.csv"));
users = unique(pooled.user); display_user = min(users);
display = pooled(pooled.user == display_user, :);
groups = unique(display.Group, "stable"); groups(groups == "Control") = [];
methods = unique(all.method, "stable"); colors = lines(numel(methods));
folder = fullfile(run_dir, "figures"); if ~isfolder(folder), mkdir(folder); end
for metric = ["evm_percent" "ber"]
    fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1400 850]);
    tiles = tiledlayout(fig, "flow", "TileSpacing", "compact");
    for group = groups.'
        ax = nexttile(tiles); hold(ax, "on");
        for m = 1:numel(methods)
            rows = sortrows(display(display.Group == group & display.method == methods(m), :), "Axis");
            if isempty(rows), continue; end
            draw(ax, rows, metric, colors(m, :), methods(m));
        end
        xlabel(ax, display.XLabel(find(display.Group == group, 1))); title(ax, group); grid(ax, "on");
        if metric == "ber"
            set(ax, "YScale", "log"); ylabel(ax, "Held-out BER"); ylim(ax, [1e-6 1]);
            yline(ax, cfg.ber_target, "--k", "Reference", "HandleVisibility", "off");
        else
            ylabel(ax, "Held-out RMS EVM (%)");
        end
        legend(ax, "Location", "best", "FontSize", 7);
    end
    title(tiles, sprintf("Scenario %d: pooled independent test bursts", cfg.scenario));
    if metric == "ber"
        subtitle(tiles, "Triangles: zero observed errors at nominal 95% upper bound; no FEC decoding");
    end
    export(fig, folder, metric + "_vs_sweep");
end

selected = conditions.Condition(find(conditions.Group ~= "Control", 1));
if isfield(cfg, "report_condition") && any(conditions.Condition == cfg.report_condition)
    selected = cfg.report_condition;
end
selected_folder = fullfile(run_dir, sprintf("seed_%d", cfg.seeds(1)), selected);
data = load(fullfile(selected_folder, "results.mat"));
order = conditions.ModOrder(conditions.Condition == selected);
names = string(data.method_names(:));
fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1400 850]);
tiles = tiledlayout(fig, "flow", "TileSpacing", "compact");
show = 1:min(2500, size(data.predictions, 1));
ideal = qammod((0:order-1).', order, "gray");
lim = ceil(max([abs(real(data.predictions(show, :, :))), abs(imag(data.predictions(show, :, :)))], [], "all"));
for m = 1:numel(names)
    ax = nexttile(tiles); value = data.predictions(show, 1, m);
    scatter(ax, real(value), imag(value), 5, "filled", "MarkerFaceAlpha", 0.15);
    hold(ax, "on"); plot(ax, real(ideal), imag(ideal), "kx");
    axis(ax, "equal"); xlim(ax, [-lim lim]); ylim(ax, [-lim lim]);
    grid(ax, "on"); xlabel(ax, "I"); ylabel(ax, "Q"); title(ax, names(m));
end
title(tiles, "Same held-out symbols, user 1: " + selected, "Interpreter", "none");
export(fig, folder, "constellations");

histories = dir(fullfile(selected_folder, "history_*.csv"));
fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1100 450]);
tiles = tiledlayout(fig, 1, 2); ax = nexttile(tiles); hold(ax, "on");
for k = 1:numel(histories)
    h = readmatrix(fullfile(histories(k).folder, histories(k).name));
    label = erase(string(histories(k).name), ["history_" ".csv"]);
    semilogy(ax, h(:, 1), h(:, 3), "DisplayName", replace(label, "_", " "));
end
set(ax, "YScale", "log"); grid(ax, "on"); legend(ax, "Location", "best");
xlabel(ax, "Optimization iteration (Adam epoch / L-BFGS step)"); ylabel(ax, "Validation MSE");
title(ax, "Checkpoint selection uses validation only");
ax = nexttile(tiles); hold(ax, "on");
for k = 1:numel(histories)
    h = readmatrix(fullfile(histories(k).folder, histories(k).name));
    if any(h(:, 4) ~= 0)
        label = erase(string(histories(k).name), ["history_" ".csv"]);
        plot(ax, h(:, 1), h(:, 4), "DisplayName", replace(label, "_", " "));
    end
end
grid(ax, "on"); legend(ax, "Location", "best"); xlabel(ax, "Optimization iteration"); ylabel(ax, "Alpha");
title(ax, "Slope coefficients (multilevel uses c1=6)");
export(fig, folder, "validation_and_alpha");

if max(users) > 1
    fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1300 750]);
    tiles = tiledlayout(fig, "flow");
    selected_group = conditions.Group(conditions.Condition == selected);
    for u = users(users > 0).'
        ax = nexttile(tiles); hold(ax, "on");
        for m = 1:numel(methods)
            rows = sortrows(pooled(pooled.user == u & pooled.Group == selected_group & pooled.method == methods(m), :), "Axis");
            if ~isempty(rows), draw(ax, rows, "ber", colors(m, :), methods(m)); end
        end
        set(ax, "YScale", "log"); grid(ax, "on"); ylim(ax, [1e-6 1]);
        title(ax, sprintf("User %d", u));
        xlabel(ax, pooled.XLabel(find(pooled.Group == selected_group, 1))); ylabel(ax, "BER");
    end
    legend(ax, "Location", "eastoutside"); export(fig, folder, "per_user_ber");
end
fprintf("All %d Python metric rows independently verified in MATLAB.\n", height(all));
end

function draw(ax, rows, metric, color, label)
if metric == "ber"
    zero = rows.bit_errors == 0; values = rows.ber; values(zero) = NaN;
    semilogy(ax, rows.Axis, values, "-o", "Color", color, "DisplayName", label);
    if any(zero)
        semilogy(ax, rows.Axis(zero), rows.ber_upper_95(zero), "v", "Color", color, "HandleVisibility", "off");
    end
else
    plot(ax, rows.Axis, rows.(metric), "-o", "Color", color, "DisplayName", label);
end
end

function export(fig, folder, name)
exportgraphics(fig, fullfile(folder, name+".png"), "Resolution", 170);
exportgraphics(fig, fullfile(folder, name+".pdf"), "ContentType", "vector");
savefig(fig, fullfile(folder, name+".fig"));
if fig.Visible == "off", close(fig); end
end

function pooled = plot_results(run_dir, cfg)
%PLOT_RESULTS Pool independent replicates; plot only held-out predictions.
conditions = readtable(fullfile(run_dir, "conditions.csv"), "TextType", "string");
summary = table();
for seed = cfg.seeds
    for c = 1:height(conditions)
        folder = fullfile(run_dir, sprintf("seed_%d", seed), conditions.Condition(c));
        assert(isfile(fullfile(folder, "results.mat")), "Incomplete condition: %s", folder);
        rows = readtable(fullfile(folder, "metrics.csv"), "TextType", "string");
        validate_results(folder, rows, cfg);
        rows.Seed = repmat(seed, height(rows), 1);
        rows.Condition = repmat(conditions.Condition(c), height(rows), 1);
        rows.PA = repmat(logical(conditions.PA(c)), height(rows), 1);
        rows.Backoff_dB = repmat(conditions.Backoff_dB(c), height(rows), 1);
        rows.Reference_SNR_dB = repmat(conditions.Reference_SNR_dB(c), height(rows), 1);
        summary = [summary; rows]; %#ok<AGROW>
    end
end
writetable(summary, fullfile(run_dir, "all_metrics.csv"));
[groups, condition, method] = findgroups(summary.Condition, summary.method);
pooled = table(condition, method, 'VariableNames', {'Condition', 'method'});
for name = ["PA" "Backoff_dB" "Reference_SNR_dB" "parameters" ...
        "real_multiplications" "sigmoid_evaluations"]
    pooled.(name) = splitapply(@(x) x(1), summary.(name), groups);
end
for name = ["bit_errors" "bit_count" "error_energy" "reference_energy"]
    pooled.(name) = splitapply(@sum, summary.(name), groups);
end
pooled.evm_percent = 100*sqrt(pooled.error_energy./pooled.reference_energy);
pooled.evm_std_across_seeds = splitapply(@std, summary.evm_percent, groups);
pooled.ber = pooled.bit_errors./pooled.bit_count;
pooled.ber_lower_95 = zeros(height(pooled), 1);
pooled.ber_upper_95 = ones(height(pooled), 1);
for row = 1:height(pooled)
    e = pooled.bit_errors(row); n = pooled.bit_count(row);
    if e > 0
        pooled.ber_lower_95(row) = betaincinv(0.025, e, n-e+1);
    end
    if e < n
        pooled.ber_upper_95(row) = betaincinv(0.975, e+1, n-e);
    end
end
pooled.target_status = repmat("inconclusive", height(pooled), 1);
pooled.target_status(pooled.ber_upper_95 < cfg.ber_target) = "meets reference";
pooled.target_status(pooled.ber_lower_95 > cfg.ber_target) = "above reference";
writetable(pooled, fullfile(run_dir, "pooled_metrics.csv"));

methods = unique(summary.method, "stable");
colors = lines(numel(methods));
figure_dir = fullfile(run_dir, "figures");
if ~isfolder(figure_dir), mkdir(figure_dir); end
for metric = ["evm_percent" "ber"]
    fig = figure("Visible", cfg.visible, "Color", "w", "Position", [80 80 1300 440]);
    tiles = tiledlayout(fig, 1, numel(cfg.reference_snr_db), "TileSpacing", "compact");
    for snr = cfg.reference_snr_db
        ax = nexttile(tiles); hold(ax, "on");
        for m = 1:numel(methods)
            rows = pooled(pooled.PA & pooled.Reference_SNR_dB == snr & ...
                pooled.method == methods(m), :);
            rows = sortrows(rows, "Backoff_dB");
            draw_curve(ax, rows.Backoff_dB, rows, metric, colors(m, :), methods(m));
        end
        xlabel(ax, "Input backoff (dB)");
        title(ax, sprintf("Reference SNR = %g dB", snr));
        decorate(ax, metric, cfg);
    end
    legend(ax, "Location", "eastoutside");
    if metric == "ber"
        title(tiles, "Held-out BER; triangles: zero errors, nominal 95% upper bound");
    else
        title(tiles, "Reference-normalized held-out RMS EVM");
    end
    export_figure(fig, figure_dir, metric + "_vs_backoff");
end

fig = figure("Visible", cfg.visible, "Color", "w", "Position", [80 80 1250 750]);
tiles = tiledlayout(fig, "flow", "TileSpacing", "compact");
for backoff = cfg.backoff_db
    ax = nexttile(tiles); hold(ax, "on");
    for m = 1:numel(methods)
        rows = pooled(pooled.PA & pooled.Backoff_dB == backoff & pooled.method == methods(m), :);
        rows = sortrows(rows, "Reference_SNR_dB");
        draw_curve(ax, rows.Reference_SNR_dB, rows, "ber", colors(m, :), methods(m));
    end
    xlabel(ax, "Reference SNR at zero-dB backoff (dB)");
    title(ax, sprintf("Backoff = %g dB", backoff));
    decorate(ax, "ber", cfg);
end
legend(ax, "Location", "eastoutside");
title(tiles, "Held-out BER versus receiver reference SNR; triangles are upper bounds");
export_figure(fig, figure_dir, "ber_vs_reference_snr");

selected = sprintf("pa_ibo_%+03d_snr_%02d", cfg.constellation_backoff_db, cfg.constellation_snr_db);
selected_folder = fullfile(run_dir, sprintf("seed_%d", cfg.seeds(1)), selected);
if isfolder(selected_folder)
    data = load(fullfile(selected_folder, "results.mat"));
    names = string(data.method_names(:));
    fig = figure("Visible", cfg.visible, "Color", "w", "Position", [80 80 1250 700]);
    tiles = tiledlayout(fig, 2, 4, "TileSpacing", "compact");
    show = 1:min(3000, size(data.predictions, 1));
    lim = max(4, ceil(max(abs([real(data.predictions(show, :)); ...
        imag(data.predictions(show, :))]), [], "all")));
    ideal = qammod((0:cfg.mod_order-1).', cfg.mod_order, "gray");
    for m = 1:numel(names)
        ax = nexttile(tiles);
        scatter(ax, real(data.predictions(show, m)), imag(data.predictions(show, m)), ...
            5, "filled", "MarkerFaceAlpha", 0.15);
        hold(ax, "on"); plot(ax, real(ideal), imag(ideal), "kx", "LineWidth", 1.2);
        axis(ax, "equal"); xlim(ax, [-lim lim]); ylim(ax, [-lim lim]); grid(ax, "on");
        title(ax, names(m)); xlabel(ax, "I"); ylabel(ax, "Q");
    end
    title(tiles, sprintf("Identical test symbols: backoff %g dB, reference SNR %g dB, seed %d", ...
        cfg.constellation_backoff_db, cfg.constellation_snr_db, cfg.seeds(1)));
    export_figure(fig, figure_dir, "constellations");

    fig = figure("Visible", cfg.visible, "Color", "w");
    ax = axes(fig); hold(ax, "on");
    for seed = cfg.seeds
        history = readmatrix(fullfile(run_dir, sprintf("seed_%d", seed), selected, "nn_history.csv"));
        semilogy(ax, history(:, 1), history(:, 2), "--", ...
            "DisplayName", sprintf("Train, seed %d", seed));
        semilogy(ax, history(:, 1), history(:, 3), "-", ...
            "DisplayName", sprintf("Validation, seed %d", seed));
    end
    set(ax, "YScale", "log"); grid(ax, "on"); legend(ax, "Location", "best");
    xlabel(ax, "Epoch"); ylabel(ax, "MSE per real output coordinate");
    title(ax, "NN convergence; checkpoint selected using validation MSE");
    export_figure(fig, figure_dir, "nn_convergence");

    rows = pooled(pooled.Condition == selected & pooled.real_multiplications > 0, :);
    fig = figure("Visible", cfg.visible, "Color", "w");
    ax = axes(fig); hold(ax, "on");
    for row = 1:height(rows)
        scatter(ax, rows.real_multiplications(row), rows.evm_percent(row), 50, "filled");
        text(ax, rows.real_multiplications(row)*1.03, rows.evm_percent(row), rows.method(row));
    end
    set(ax, "XScale", "log"); grid(ax, "on");
    xlabel(ax, "Real multiplications per symbol (direct evaluation)");
    ylabel(ax, "Pooled held-out RMS EVM (%)");
    title(ax, "Complexity comparison; NN also requires 12 sigmoid evaluations");
    export_figure(fig, figure_dir, "evm_vs_complexity");
end

controls = pooled(~pooled.PA, ["Condition" "method" "evm_percent" "ber" "bit_errors" "bit_count"]);
disp(controls);
fprintf("Pooled metrics: %s\n", fullfile(run_dir, "pooled_metrics.csv"));
end

function validate_results(folder, rows, cfg)
% Independently check Python's BER/EVM against MATLAB's QAM demapper.
data = load(fullfile(folder, "results.mat"));
source = load(fullfile(folder, "dataset.mat"), "Y_test");
reference = complex(double(source.Y_test(:, 1)), double(source.Y_test(:, 2)));
assert(isequal(reference, data.reference(:)), "Test reference mismatch: %s", folder);
assert(isequal(string(data.method_names(:)), rows.method), "Method order mismatch: %s", folder);
assert(isequal(size(data.predictions), [numel(reference) height(rows)]), ...
    "Prediction shape mismatch: %s", folder);
bits = qamdemod(reference, cfg.mod_order, "gray", "OutputType", "bit");
for m = 1:height(rows)
    prediction = data.predictions(:, m);
    detected = qamdemod(prediction, cfg.mod_order, "gray", "OutputType", "bit");
    evm = 100*norm(prediction-reference)/norm(reference);
    assert(nnz(detected ~= bits) == rows.bit_errors(m), "BER mismatch: %s", folder);
    assert(numel(bits) == rows.bit_count(m), "Bit count mismatch: %s", folder);
    assert(abs(evm-rows.evm_percent(m)) < 1e-8, "EVM mismatch: %s", folder);
end
end

function draw_curve(ax, x, rows, metric, color, name)
if metric == "ber"
    zero = rows.bit_errors == 0;
    values = rows.ber; values(zero) = NaN;
    semilogy(ax, x, values, "-o", "Color", color, "DisplayName", name);
    if any(zero)
        semilogy(ax, x(zero), rows.ber_upper_95(zero), "v", "Color", color, ...
            "MarkerFaceColor", "white", "HandleVisibility", "off");
    end
else
    plot(ax, x, rows.(metric), "-o", "Color", color, "DisplayName", name);
end
end

function decorate(ax, metric, cfg)
grid(ax, "on");
if metric == "ber"
    set(ax, "YScale", "log");
    ylabel(ax, "BER");
    yline(ax, cfg.ber_target, "--k", "Literature reference", "HandleVisibility", "off");
    ylim(ax, [1e-6 1]);
else
    ylabel(ax, "RMS EVM (%)");
end
end

function export_figure(fig, folder, name)
exportgraphics(fig, fullfile(folder, name + ".png"), "Resolution", 180);
exportgraphics(fig, fullfile(folder, name + ".pdf"), "ContentType", "vector");
savefig(fig, fullfile(folder, name + ".fig"));
if fig.Visible == "off", close(fig); end
end

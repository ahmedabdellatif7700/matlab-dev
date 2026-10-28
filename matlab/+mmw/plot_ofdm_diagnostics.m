function plot_ofdm_diagnostics(cfg, run_dir)
%PLOT_OFDM_DIAGNOSTICS Pilot mismatch and matched-duration transmitter PAPR.
conditions = readtable(fullfile(run_dir, "conditions.csv"), "TextType", "string");
fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1100 450]);
tiles = tiledlayout(fig, 1, 2); ax = nexttile(tiles); hold(ax, "on");
for order = [16 64]
    selected = conditions(conditions.ModOrder == order & conditions.EstimateMode == "LS" & conditions.Group ~= "Control", :);
    values = zeros(height(selected), numel(cfg.seeds));
    for c = 1:height(selected)
        for s = 1:numel(cfg.seeds)
            path = fullfile(run_dir, sprintf("seed_%d", cfg.seeds(s)), selected.Condition(c), "dataset.mat");
            d = load(path, "diagnostic_json"); diagnostic = jsondecode(d.diagnostic_json);
            values(c, s) = diagnostic.pilot_shape_nmse;
        end
    end
    plot(ax, selected.Axis, 10*log10(mean(values, 2)), "-o", "DisplayName", sprintf("%d-QAM", order));
end
grid(ax, "on"); legend(ax, "Location", "best");
xlabel(ax, "Input backoff (dB)"); ylabel(ax, "Pilot channel-shape mismatch NMSE (dB)");
title(ax, "Pilot channel-shape mismatch");
ax = nexttile(tiles); hold(ax, "on");
for order = [16 64]
    selected = sprintf("qam%d_LS_ibo+00", order); ofdm = []; single = [];
    for seed = cfg.seeds
        d = load(fullfile(run_dir, sprintf("seed_%d", seed), selected, "dataset.mat"), ...
            "papr_ofdm_db", "papr_single_carrier_db");
        ofdm = [ofdm d.papr_ofdm_db]; single = [single d.papr_single_carrier_db]; %#ok<AGROW>
    end
    plot(ax, sort(ofdm), (1:numel(ofdm))/numel(ofdm), "DisplayName", sprintf("OFDM %d-QAM", order));
    plot(ax, sort(single), (1:numel(single))/numel(single), "--", "DisplayName", sprintf("RRC single-carrier %d-QAM", order));
end
grid(ax, "on"); legend(ax, "Location", "best");
xlabel(ax, "Payload-interval PAPR (dB)"); ylabel(ax, "Empirical cumulative fraction");
title(ax, "Transmitter PAPR: 1 GHz, approximately 52 ns");
title(tiles, "Pilot diagnostic removes common gain only; PAPR is measured before the PA");
folder = fullfile(run_dir, "figures");
exportgraphics(fig, fullfile(folder, "ofdm_pilots_and_papr.png"), "Resolution", 170);
exportgraphics(fig, fullfile(folder, "ofdm_pilots_and_papr.pdf"), "ContentType", "vector");
savefig(fig, fullfile(folder, "ofdm_pilots_and_papr.fig"));
if fig.Visible == "off", close(fig); end
end

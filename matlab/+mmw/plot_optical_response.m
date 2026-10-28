function plot_optical_response(cfg, run_dir)
%PLOT_OPTICAL_RESPONSE Electrical beating response over the occupied band.
fig = figure("Visible", cfg.visible, "Color", "w", "Position", [50 50 1100 450]);
tiles = tiledlayout(fig, 1, 2);
for kind = ["DSB" "SSB"]
    ax = nexttile(tiles); hold(ax, "on");
    for length_km = cfg.fiber_lengths_km
        [~, f, h] = mmw.optical_fading(zeros(4096, 1), cfg, kind, length_km, 4096);
        plot(ax, f/1e9, 20*log10(max(abs(h), 1e-6)), ...
            "DisplayName", sprintf("%.2f km", length_km));
    end
    width = cfg.symbol_rate_hz*(1+cfg.rrc_rolloff)/2/1e9;
    xlim(ax, [-width width]); ylim(ax, [-70 2]); grid(ax, "on");
    xlabel(ax, "Offset from 60 GHz (GHz)"); ylabel(ax, "RF magnitude response (dB)");
    title(ax, kind); legend(ax, "Location", "best");
end
title(tiles, "Small-signal optical beating response; SSB known phase/delay removed");
folder = fullfile(run_dir, "figures");
exportgraphics(fig, fullfile(folder, "optical_rf_response.png"), "Resolution", 170);
exportgraphics(fig, fullfile(folder, "optical_rf_response.pdf"), "ContentType", "vector");
savefig(fig, fullfile(folder, "optical_rf_response.fig"));
if fig.Visible == "off", close(fig); end
end

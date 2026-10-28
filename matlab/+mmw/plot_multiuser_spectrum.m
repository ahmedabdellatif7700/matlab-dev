function plot_multiuser_spectrum(cfg, run_dir)
folder = fullfile(run_dir, sprintf("seed_%d", cfg.seeds(1)), cfg.report_condition);
if ~isfolder(folder), return; end
d = load(fullfile(folder, "dataset.mat"), "spectrum_frequency_hz", "spectrum_before", "spectrum_after");
fig = figure("Visible", cfg.visible, "Color", "w");
reference = max(d.spectrum_before);
plot(d.spectrum_frequency_hz/1e9, 10*log10(max(d.spectrum_before/reference, 1e-10)), "DisplayName", "Before common MZM");
hold on;
plot(d.spectrum_frequency_hz/1e9, 10*log10(max(d.spectrum_after/reference, 1e-10)), "DisplayName", "After common MZM");
grid on; legend("Location", "best"); xlabel("Baseband frequency (GHz)");
ylabel("Mean FFT-bin power, common reference (dB)");
title("Aggregated four-user SC-FDMA spectrum, excluding receiver noise");
output = fullfile(run_dir, "figures");
exportgraphics(fig, fullfile(output, "multiuser_spectrum.png"), "Resolution", 170);
exportgraphics(fig, fullfile(output, "multiuser_spectrum.pdf"), "ContentType", "vector");
savefig(fig, fullfile(output, "multiuser_spectrum.fig"));
if fig.Visible == "off", close(fig); end
end

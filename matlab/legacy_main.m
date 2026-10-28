%% 60 GHz radio-over-fiber: chromatic dispersion and neural equalization

clear;
clc;
close all;
format short;

%% Configuration

rng(42, "twister");

mod_order   = 16;
num_symbols = 10000;
snr_db      = 15;

prbs_order = 15;

% Small-signal, chirp-free optical DSB intensity modulation/direct detection.
% Optical link gain is normalized at L=0; RF driver compression is optional.
rf_carrier_hz = 60e9;
optical_wavelength_m = 1550e-9;
fiber_length_km = 0.5;
dispersion_ps_nm_km = 17;
speed_of_light_m_s = 299792458;

symbol_rate_hz = 1e9;
samples_per_symbol = 8;
sample_rate_hz = symbol_rate_hz * samples_per_symbol;
rrc_rolloff = 0.25;
rrc_span_symbols = 12;
noise_enabled = false; % Enable for optional receiver AWGN.

% RF driver at 60 GHz, before the linearized optical modulator.
% Illustrative AM/AM and AM/PM parameters, not measured 60 GHz device data.
pa_enabled = true;
pa_input_backoff_db = 3; % Lower -> stronger compression; try 0, 3, 6, 12 dB.
pa_smoothness = 2;     % Higher -> sharper approach to saturation.
pa_phase_limit_deg = 20; % Asymptotic AM/PM rotation; 10 deg at |x|=A_sat.
% No phase noise or optical-modulator nonlinearity is included.
% Compare IBO = 0, 3, 6, 12 dB in separate runs; keep data/split identical.
% Set pa_enabled=false to isolate CD, fiber_length_km=0 to isolate RF distortion.
% snr_db defines the back-to-back matched-filter SNR, before CD fading.
fft_guard_samples = 4096; % Zero guard on each side of the optical-link waveform

equalizer_taps = 2;

python_exe = ...
    "/home/ahmedabdellatif/miniconda3/envs/machine-learning/bin/python";

% Standard workflow: regenerate the dataset and run Python training/inference.
% Set false only when replotting a previously verified experiment.
% FPGA execution is not required; optional FPGA plotting is disabled below.
run_python_equalizer = true;
plot_fpga_results = false; % Enable later to include the fourth constellation figure.

%% PRBS generation

bits_per_symbol = log2(mod_order);
num_bits = num_symbols * bits_per_symbol;

prbs_seed = mod((0:prbs_order-1).', 2);

if ~any(prbs_seed)
    prbs_seed(1) = 1;
end

switch prbs_order
    case 15
        taps = [15 14];

    case 31
        taps = [31 28];

    otherwise
        error("Unsupported PRBS order.");
end

state = logical(prbs_seed(:).');
tx_bits = false(num_bits, 1);

for n = 1:num_bits

    tx_bits(n) = state(end);

    feedback = false;

    for t = 1:numel(taps)
        feedback = xor( ...
            feedback, ...
            state(taps(t)) ...
        );
    end

    state = [
        feedback, ...
        state(1:end-1)
    ];

end

tx_bits = uint8(tx_bits);

%% QAM modulation

tx_indices = bi2de( ...
    reshape( ...
        tx_bits, ...
        bits_per_symbol, ...
        [] ...
    ).', ...
    "left-msb");

tx_symbols = qammod( ...
    tx_indices, ...
    mod_order, ...
    "gray");

tx_symbols = tx_symbols(:);

%% Pulse shaping
% Unit-energy root-raised-cosine Tx/Rx filters; model CD at waveform rate.
rrc_filter = rcosdesign(rrc_rolloff, rrc_span_symbols, ...
    samples_per_symbol, "sqrt").';
tx_upsampled = zeros(num_symbols * samples_per_symbol, 1);
tx_upsampled(1:samples_per_symbol:end) = tx_symbols;
tx_waveform = conv(tx_upsampled, rrc_filter, "full");

%% 60 GHz RF driver distortion (complex-envelope AM/AM and AM/PM)
% Apply after pulse shaping: even 4-QAM has a varying waveform envelope.
% P_ref = E_symbol/sps is the nominal uncompressed waveform power.
% A_sat^2/P_ref = 10^(IBO/10), with unity small-signal gain.
% The fixed saturation level is independent of test-set sample statistics.
pa_reference_power = mean(abs(qammod((0:mod_order-1).', ...
    mod_order, "gray")).^2) / samples_per_symbol;
pa_saturation_amplitude = sqrt(pa_reference_power * 10^(pa_input_backoff_db/10));
tx_pa_waveform = apply_rapp_pa(tx_waveform, pa_enabled, ...
    pa_saturation_amplitude, pa_smoothness, pa_phase_limit_deg);

% Debug transfer characteristic, including the linear response for reference.
pa_debug_amplitude = linspace(0, 2*pa_saturation_amplitude, 500).';
pa_debug_output = apply_rapp_pa(pa_debug_amplitude, pa_enabled, ...
    pa_saturation_amplitude, pa_smoothness, pa_phase_limit_deg);
pa_debug_fig = figure("Color", "w", "Name", "60 GHz RF driver distortion", ...
    "Position", [100 100 1000 420]);
tiledlayout(1, 2);
nexttile;
plot(pa_debug_amplitude/pa_saturation_amplitude, ...
    abs(pa_debug_output)/pa_saturation_amplitude, "LineWidth", 1.2);
hold on;
plot(pa_debug_amplitude/pa_saturation_amplitude, ...
    pa_debug_amplitude/pa_saturation_amplitude, "--");
xlabel("Input amplitude / saturation amplitude");
ylabel("Output amplitude / saturation amplitude");
legend("Configured RF driver", "Linear driver", "Location", "northwest");
title("AM/AM compression");
grid on;
nexttile;
plot(pa_debug_amplitude/pa_saturation_amplitude, ...
    angle(pa_debug_output)*180/pi, "LineWidth", 1.2);
xlabel("Input amplitude / saturation amplitude");
ylabel("Phase rotation (degrees)");
title("AM/PM conversion");
grid on;

%% Chromatic dispersion: optical DSB IM/DD equivalent RF response
% D [ps/(nm km)] -> [s/m^2]; L [km] -> [m].
% H_RF(f) = cos(pi*D*lambda^2*L*f^2/c).
% H_BB(f) = H_RF(fc+f), after removing bulk fiber propagation delay.
% This is electrical carrier/sideband beating after direct detection.
% It is NOT the optical-field all-pass dispersion transfer function.
D_si = dispersion_ps_nm_km * 1e-6;
L_m = fiber_length_km * 1e3;
cd_factor = pi * D_si * optical_wavelength_m^2 * L_m / speed_of_light_m_s;

[rx_cd_waveform, cd_frequency_hz, cd_response] = apply_cd_dsb( ...
    tx_pa_waveform, sample_rate_hz, rf_carrier_hz, cd_factor, ...
    fft_guard_samples);

%% Optional receiver noise (disabled by default)
% Fixed noise referenced to transmitted symbol energy; never normalize to
% the faded received power, which would conceal the CD-induced SNR penalty.
rx_waveform = rx_cd_waveform;
if noise_enabled
    noise_variance = mean(abs(tx_symbols).^2) / 10^(snr_db/10);
    rx_waveform = rx_waveform + sqrt(noise_variance/2) * ...
        (randn(size(rx_waveform)) + 1j*randn(size(rx_waveform)));
end

%% Matched filter and symbol sampling
% Tx + Rx RRC delay is span*sps. Bulk fiber delay was removed analytically.
% No correlation alignment or gain/phase fitting is applied.
matched_cd = conv(rx_cd_waveform, rrc_filter, "full");
matched_rx = conv(rx_waveform, rrc_filter, "full");
sample_indices = rrc_span_symbols * samples_per_symbol + 1 + ...
    (0:num_symbols-1).' * samples_per_symbol;
rx_linear = matched_cd(sample_indices);
rx_noisy = matched_rx(sample_indices);

% Exclude burst edges so zero padding and pulse-filter start/end transients
% do not dominate training. X and Y keep the original Python interface.
edge_symbols = rrc_span_symbols;
keep_indices = (edge_symbols+1:num_symbols-edge_symbols).';
rx_aligned = rx_noisy(keep_indices);
tx_aligned = tx_symbols(keep_indices);
delay = 0;

%% Debug: CD-induced RF power fading versus frequency
% D is held constant over the optical sidebands. This plots the electrical
% link response caused by CD, not a wavelength-dependent D(lambda) curve.
debug_frequency_hz = linspace(0, 100e9, 20001).';
debug_response = cos(cd_factor * debug_frequency_hz.^2);
occupied_half_bandwidth_hz = symbol_rate_hz*(1+rrc_rolloff)/2;
occupied_edges_hz = rf_carrier_hz + [-1 1]*occupied_half_bandwidth_hz;
null_frequencies_hz = [];
if cd_factor ~= 0
    null_orders = (0:floor(abs(cd_factor)*(100e9)^2/pi)).';
    null_frequencies_hz = sqrt((pi/2 + null_orders*pi)/abs(cd_factor));
    null_frequencies_hz = null_frequencies_hz(null_frequencies_hz <= 100e9);
end
cd_debug_fig = figure("Color", "w", ...
    "Name", "Chromatic-dispersion RF fading debug", ...
    "Position", [100 100 1100 440]);
tiledlayout(1, 2);
nexttile;
plot(debug_frequency_hz/1e9, ...
    20*log10(max(abs(debug_response), 1e-6)), "LineWidth", 1.2);
hold on;
xline(rf_carrier_hz/1e9, "--r", "60 GHz carrier");
for f_null = null_frequencies_hz.'
    xline(f_null/1e9, ":", "Color", [0.5 0.5 0.5]);
end
xlabel("RF frequency (GHz)");
ylabel("Normalized RF power response (dB)");
title(sprintf("DSB IM/DD: L = %.2f km, D = %.1f ps/(nm km)", ...
    fiber_length_km, dispersion_ps_nm_km));
xlim([0 100]); ylim([-60 1]); grid on;
nexttile;
plot((rf_carrier_hz+cd_frequency_hz)/1e9, ...
    20*log10(max(abs(cd_response), 1e-6)), "LineWidth", 1.2);
hold on;
xline(rf_carrier_hz/1e9, "--r", "Carrier");
xline(occupied_edges_hz(1)/1e9, "--k", "Band edge");
xline(occupied_edges_hz(2)/1e9, "--k", "Band edge");
for f_null = null_frequencies_hz.'
    if abs(f_null-rf_carrier_hz) <= 2*occupied_half_bandwidth_hz
        xline(f_null/1e9, ":", "Fading null");
    end
end
xlabel("RF frequency (GHz)");
ylabel("Normalized RF power response (dB)");
title("Response around the occupied QAM band");
xlim((rf_carrier_hz + [-2 2]*occupied_half_bandwidth_hz)/1e9);
ylim([-60 1]); grid on;

% Export before Python execution so this diagnostic is available independently.
cd_debug_dir = fullfile(fileparts(fileparts(mfilename("fullpath"))), ...
    "figures", "matlab", "report");
if ~exist(cd_debug_dir, "dir")
    mkdir(cd_debug_dir);
end
exportgraphics(pa_debug_fig, fullfile(cd_debug_dir, ...
    "rf_driver_amam_ampm.png"), "Resolution", 300);
exportgraphics(cd_debug_fig, fullfile(cd_debug_dir, ...
    "chromatic_dispersion_frequency_response.png"), "Resolution", 300);

%% Equalizer time-delay line

num_samples = ...
    length(rx_aligned) ...
    - equalizer_taps ...
    + 1;

X = zeros( ...
    num_samples, ...
    2 * equalizer_taps);

Y = zeros( ...
    num_samples, ...
    2);

for n = equalizer_taps:length(rx_aligned)

    k = ...
        n ...
        - equalizer_taps ...
        + 1;

    for tap = 1:equalizer_taps

        sample_index = ...
            n - tap + 1;

        X(k, 2*tap-1) = ...
            real( ...
                rx_aligned(sample_index) ...
            );

        X(k, 2*tap) = ...
            imag( ...
                rx_aligned(sample_index) ...
            );

    end

    Y(k, :) = [
        real(tx_aligned(n)), ...
        imag(tx_aligned(n))
    ];

end

%% Equalizer interface

equalizer_input_size = ...
    size(X, 2);

equalizer_output_size = ...
    size(Y, 2);



%% Project paths

matlab_dir = ...
    fileparts( ...
        mfilename("fullpath"));

project_root = ...
    fileparts(matlab_dir);

dataset_dir = ...
    fullfile( ...
        project_root, ...
        "datasets");

if ~exist(dataset_dir, "dir")
    mkdir(dataset_dir);
end

dataset_file = ...
    fullfile( ...
        dataset_dir, ...
        "training_data.mat");

%% Save equalizer dataset
% Do not overwrite the dataset used for an existing FPGA run merely
% to regenerate plots.
if run_python_equalizer
    save( ...
    dataset_file, ...
    "X", ...
    "Y", ...
    "tx_bits", ...
    "tx_indices", ...
    "tx_symbols", ...
    "rx_linear", ...
    "rf_carrier_hz", "optical_wavelength_m", "fiber_length_km", ...
    "dispersion_ps_nm_km", "symbol_rate_hz", "samples_per_symbol", ...
    "sample_rate_hz", "rrc_rolloff", "rrc_span_symbols", ...
    "noise_enabled", "fft_guard_samples", "keep_indices", ...
    "pa_enabled", "pa_input_backoff_db", "pa_smoothness", ...
    "pa_saturation_amplitude", "pa_reference_power", "pa_phase_limit_deg", ...
    "tx_waveform", "tx_pa_waveform", ...
    "cd_frequency_hz", "cd_response", "null_frequencies_hz", ...
    "rx_noisy", ...
    "rx_aligned", ...
    "tx_aligned", ...
    "delay", ...
    "snr_db", ...
    "mod_order", ...
    "bits_per_symbol", ...
    "equalizer_taps", ...
    "equalizer_input_size", ...
    "equalizer_output_size", ...
    "-v7");
end

%% Export Q8.8 inference inputs (MATLAB is the sole producer)
% The Python split is int(0.8 * N), with zero-based indexing. Quantize the
% held-out rows only, after casting X to single as Python does when loading
% training_data.mat. Preserve I[n], Q[n], I[n-1], Q[n-1] sample order.
input_mem_file = fullfile(project_root, "python", "artifacts", ...
    "inference", "test_inputs.mem");
test_start = floor(0.8 * size(X, 1));
if run_python_equalizer
    export_q8_8_inputs(input_mem_file, X(test_start+1:end, :));
    fprintf("MATLAB Q8.8 input file: %s (%d samples)\n", ...
        input_mem_file, size(X, 1) - test_start);
end

%% Linear FFE benchmark (same training/test split and temporal taps as NN)
ffe_script = fullfile(project_root, "python", "ffe_equalizer.py");
if run_python_equalizer
    fprintf("\nFitting linear FFE benchmark...\n");
    ffe_status = system(sprintf('"%s" "%s"', python_exe, ffe_script));
    if ffe_status ~= 0
        error("FFE benchmark failed.");
    end
end

%% Neural equalizer

python_file = ...
    fullfile( ...
        project_root, ...
        "python", ...
        "equalizer.py");

if run_python_equalizer

    python_cmd = sprintf( ...
        '"%s" "%s"', ...
        python_exe, ...
        python_file);

    fprintf( ...
        "\nRunning neural equalizer...\n\n");

    status = system( ...
        python_cmd);

    if status ~= 0
        error( ...
            "Neural equalizer failed.");
    end

else

    fprintf("\nUsing existing Python equalizer results (no retraining).\n");

end

%% Load separate Python inference results

ffe_file = fullfile(dataset_dir, "equalized_output_ffe.mat");
onnx_file = fullfile(dataset_dir, "equalized_output_onnx_fp32.mat");
software_file = fullfile(dataset_dir, "equalized_output_software_q8_8.mat");
fpga_file = fullfile(dataset_dir, "equalized_output_fpga.mat");

has_fpga_results = plot_fpga_results && isfile(fpga_file);

ffe_data = load(ffe_file);
rx_ffe = ffe_data.rx_equalized_ffe(:);
onnx_data = load(onnx_file);
software_data = load(software_file);
if has_fpga_results
    fpga_data = load(fpga_file);
end

rx_before = onnx_data.rx_before(:);
tx_reference = onnx_data.tx_reference(:);
rx_onnx_fp32 = onnx_data.rx_equalized_onnx_fp32(:);
rx_python_q8_8 = software_data.rx_equalized_software_q8_8(:);
rx_fpga_q8_8 = [];
if has_fpga_results
    rx_fpga_q8_8 = fpga_data.rx_equalized_fixed_point(:);
end

test_start = double(onnx_data.test_start(1));
test_count = numel(tx_reference);

%% Optional FPGA results
% Python training and both software inference paths remain required.
% FPGA inference is optional for MATLAB/Python-only experiments.

%% Minimal, directly comparable constellation plots
% Four software constellation figures, plus an optional FPGA figure:
%   1. Received samples before equalization.
%   2. ONNX Runtime inference with original FP32 parameters and operations.
%   3. Python Q8.8 fixed-point software inference.
%   4. Linear least-squares FFE with identical temporal taps.
%   5. KV260 FPGA Q8.8 hardware inference, when its MAT file exists.
% Optional FPGA results are plotted as supplied, without validation.
% No legends or figure titles; RMS EVM appears inside each plot.
% Reference-normalized RMS EVM uses no gain/phase fitting.

ideal_symbols = qammod((0:mod_order-1).', mod_order, "gray");
all_plot_symbols = [ ...
    rx_before; rx_ffe; rx_onnx_fp32; rx_python_q8_8; rx_fpga_q8_8; ideal_symbols ...
];
axis_limit = max(1, ceil(2 * 1.08 * max([ ...
    abs(real(all_plot_symbols)); abs(imag(all_plot_symbols)) ...
])) / 2);

% Standard MATLAB colors shared across all four constellation figures.
sample_color = [0.0000, 0.4470, 0.7410];  % MATLAB default blue
ideal_color = [0.8500, 0.3250, 0.0980];   % MATLAB default orange

save_constellation_figures = true;
figure_dir = fullfile(project_root, "figures", "matlab", "report");
% Keep the four-figure comparison in its own export directory.
constellation_dir = fullfile(figure_dir, "inference_comparison");
if ~exist(constellation_dir, "dir")
    mkdir(constellation_dir);
end

% All four plots use identical axis limits and marker sizes.
evm_before = plot_constellation( ...
    rx_before, tx_reference, ideal_symbols, ...
    "Before equalization", sample_color, ideal_color, axis_limit, ...
    save_constellation_figures, constellation_dir, "00_before_equalization.png");

evm_ffe = plot_constellation( ...
    rx_ffe, tx_reference, ideal_symbols, ...
    "Linear FFE (least squares)", sample_color, ideal_color, axis_limit, ...
    save_constellation_figures, constellation_dir, "04_ffe.png");

evm_onnx_fp32 = plot_constellation( ...
    rx_onnx_fp32, tx_reference, ideal_symbols, ...
    "ONNX FP32", sample_color, ideal_color, axis_limit, ...
    save_constellation_figures, constellation_dir, "01_onnx_fp32.png");

evm_python_q8_8 = plot_constellation( ...
    rx_python_q8_8, tx_reference, ideal_symbols, ...
    "Python Q8.8", sample_color, ideal_color, axis_limit, ...
    save_constellation_figures, constellation_dir, "02_software_q8_8.png");

if has_fpga_results
    evm_fpga = plot_constellation( ...
        rx_fpga_q8_8, tx_reference, ideal_symbols, ...
        "FPGA Q8.8", sample_color, ideal_color, axis_limit, ...
        save_constellation_figures, constellation_dir, "03_fpga_q8_8.png");
end

%% Held-out BER benchmark (same Gray QAM mapping as transmitter)
benchmark_names = ["Before EQ"; "Linear FFE"; "ONNX FP32"; "Python Q8.8"];
benchmark_signals = {rx_before; rx_ffe; rx_onnx_fp32; rx_python_q8_8};
reference_indices = qamdemod(tx_reference, mod_order, "gray");
reference_bits = de2bi(reference_indices, bits_per_symbol, "left-msb");
benchmark_ber = zeros(4, 1);
for method = 1:4
    detected_indices = qamdemod(benchmark_signals{method}, mod_order, "gray");
    detected_bits = de2bi(detected_indices, bits_per_symbol, "left-msb");
    benchmark_ber(method) = nnz(detected_bits ~= reference_bits)/numel(reference_bits);
end
benchmark_evm = [evm_before; evm_ffe; evm_onnx_fp32; evm_python_q8_8];
benchmark_table = table(benchmark_names(:), benchmark_evm(:), benchmark_ber(:), ...
    'VariableNames', {'Method', 'RMS_EVM_percent', 'BER'});
disp(benchmark_table);
writetable(benchmark_table, fullfile(constellation_dir, "equalizer_benchmark.csv"));

% Float-vs-fixed differences are reported numerically, not inferred from
% the visual similarity of constellation clusters.
fp32_q8_error = double(rx_onnx_fp32) - double(rx_python_q8_8);
fp32_q8_different = nnz(fp32_q8_error ~= 0);
fp32_q8_max_error = max(abs(fp32_q8_error));
fp32_q8_rmse = sqrt(mean(abs(fp32_q8_error).^2));
%% Summary
fprintf("RF driver compression: %d | IBO: %.1f dB | Rapp p: %.1f\n", ...
    pa_enabled, pa_input_backoff_db, pa_smoothness);
fprintf("AM/PM phase limit   : %.1f degrees\n", pa_phase_limit_deg);
fprintf("RF carrier          : %.1f GHz\n", rf_carrier_hz/1e9);
fprintf("Optical wavelength  : %.1f nm\n", optical_wavelength_m*1e9);
fprintf("Fiber length        : %.3f km\n", fiber_length_km);
fprintf("Dispersion          : %.3f ps/(nm km)\n", dispersion_ps_nm_km);
fprintf("CD carrier gain     : %.3f dB\n", ...
    20*log10(max(abs(cos(cd_factor*rf_carrier_hz^2)), realmin)));
fprintf("Receiver noise      : %d\n", noise_enabled);
fprintf("\n");
fprintf("============================================================\n");
fprintf("Equalizer workflow completed successfully\n");
fprintf("============================================================\n");

fprintf( ...
    "Modulation          : %d-QAM\n", ...
    mod_order);

fprintf( ...
    "Symbols             : %d\n", ...
    num_symbols);

fprintf( ...
    "Optional B2B SNR    : %.1f dB\n", ...
    snr_db);

fprintf( ...
    "Residual delay      : %d\n", ...
    delay);

fprintf( ...
    "Equalizer taps      : %d\n", ...
    equalizer_taps);

fprintf( ...
    "Equalizer inputs    : %d\n", ...
    equalizer_input_size);

fprintf( ...
    "Equalizer outputs   : %d\n", ...
    equalizer_output_size);

fprintf( ...
    "Dataset samples     : %d\n", ...
    size(X, 1));

fprintf( ...
    "Test samples        : %d\n", ...
    length(tx_reference));

fprintf("\nRMS EVM (delay aligned; no gain/phase correction)\n");
fprintf("  Estimated delay     : %d symbol(s)\n", delay);
fprintf("  Held-out test set   : %d symbol(s)\n", numel(tx_reference));
fprintf("  Before EQ           : %.3f %%\n", evm_before);
fprintf("  Linear FFE EQ      : %.3f %%\n", evm_ffe);
fprintf("  ONNX FP32 EQ        : %.3f %%\n", evm_onnx_fp32);
fprintf("  Python Q8.8 EQ      : %.3f %%\n", evm_python_q8_8);
if has_fpga_results
    fprintf("  KV260 FPGA Q8.8 EQ  : %.3f %%\n", evm_fpga);
end

fprintf("\nONNX FP32 vs Python Q8.8\n");
fprintf("  Different samples  : %d / %d\n", fp32_q8_different, test_count);
fprintf("  Maximum abs. error : %.8g\n", fp32_q8_max_error);
fprintf("  RMSE               : %.8g\n", fp32_q8_rmse);
fprintf("Dataset             : %s\n", dataset_file);
fprintf("ONNX results        : %s\n", onnx_file);
fprintf("Software Q8.8       : %s\n", software_file);
fprintf("FPGA results        : %s\n", fpga_file);
if save_constellation_figures
    fprintf("Constellation plots  : %s\n", constellation_dir);
end

% Append a compact numerical report to disk for experiment records.
log_file = fullfile(figure_dir, "equalizer_metrics.log");
log_id = fopen(log_file, "a");
log_cleanup = onCleanup(@() fclose(log_id));
fprintf(log_id, "\n=== Equalizer evaluation: %s ===\n", ...
    datestr(now, "yyyy-mm-dd HH:MM:SS"));
fprintf(log_id, "RF driver: enabled=%d | IBO=%.6g dB | Rapp p=%.6g\n", ...
    pa_enabled, pa_input_backoff_db, pa_smoothness);
fprintf(log_id, "AM/PM phase limit: %.6g degrees\n", pa_phase_limit_deg);
fprintf(log_id, "CD: fc=%.6g Hz | lambda=%.6g m | L=%.6g km | D=%.6g ps/(nm km) | noise=%d\n", ...
    rf_carrier_hz, optical_wavelength_m, fiber_length_km, ...
    dispersion_ps_nm_km, noise_enabled);
fprintf(log_id, "Modulation: %d-QAM | Symbols: %d | SNR: %.1f dB\n", ...
    mod_order, num_symbols, snr_db);
fprintf(log_id, "Residual delay: %d | Taps: %d | Test samples: %d | Test start: %d\n", ...
    delay, equalizer_taps, test_count, test_start);
fprintf(log_id, "RMS EVM (%%): before=%.6f, ONNX FP32=%.6f, Python Q8.8=%.6f\n", ...
    evm_before, evm_onnx_fp32, evm_python_q8_8);
fprintf(log_id, "Linear FFE RMS EVM (%%): %.6f | BER: %.8g\n", evm_ffe, benchmark_ber(2));
if has_fpga_results
    fprintf(log_id, "FPGA Q8.8 RMS EVM (%%): %.6f\n", evm_fpga);
end
fprintf(log_id, ...
    "ONNX FP32 vs Python Q8.8: differing=%d/%d, max_abs_error=%.8g, rmse=%.8g\n", ...
    fp32_q8_different, test_count, fp32_q8_max_error, fp32_q8_rmse);
fprintf(log_id, "ONNX MAT: %s\nSoftware Q8.8 MAT: %s\nFPGA MAT: %s\n", ...
    onnx_file, software_file, fpga_file);
clear log_cleanup; % Close the file before printing the completion message.
fprintf("Metrics log         : %s\n", log_file);
fprintf("============================================================\n");

%% Local functions

function evm_percent = rms_evm(signal, reference)

signal = signal(:);
reference = reference(:);
evm_percent = 100 * sqrt( ...
    sum(abs(signal - reference).^2) / sum(abs(reference).^2));

end

function evm_percent = plot_constellation( ...
    signal, reference, ideal_symbols, window_name, ...
    sample_color, ideal_color, axis_limit, ...
    save_figure, figure_dir, file_name)

signal = signal(:);
reference = reference(:);
evm_percent = rms_evm(signal, reference);

fig = figure( ...
    "Color", "w", ...
    "Name", window_name, ...
    "NumberTitle", "off", ...
    "Position", [120, 100, 520, 480]);
ax = axes("Parent", fig);
hold(ax, "on");

% Small, translucent sample points and simple ideal QAM markers.
scatter(ax, real(signal), imag(signal), 8, ...
    "filled", "MarkerFaceColor", sample_color, ...
    "MarkerEdgeColor", "none", "MarkerFaceAlpha", 0.25);
plot(ax, real(ideal_symbols), imag(ideal_symbols), "x", ...
    "LineStyle", "none", "Color", ideal_color, ...
    "MarkerSize", 9, "LineWidth", 1.5);

% Only axis labels and a small EVM annotation; no title or legend.
xlabel(ax, "In-phase (I)");
ylabel(ax, "Quadrature (Q)");
text(ax, 0.04, 0.96, sprintf("RMS EVM: %.3f%%", evm_percent), ...
    "Units", "normalized", "HorizontalAlignment", "left", ...
    "VerticalAlignment", "top", "FontSize", 10, ...
    "Color", [0.22, 0.25, 0.29]);

axis(ax, "equal");
xlim(ax, [-axis_limit, axis_limit]);
ylim(ax, [-axis_limit, axis_limit]);
xticks(ax, linspace(-axis_limit, axis_limit, 5));
yticks(ax, linspace(-axis_limit, axis_limit, 5));
grid(ax, "on");
box(ax, "off");
set(ax, "FontName", "Arial", "FontSize", 10, ...
    "LineWidth", 0.7, "GridAlpha", 0.10, "TickDir", "out");

if save_figure
    exportgraphics(fig, fullfile(figure_dir, file_name), ...
        "Resolution", 300, "BackgroundColor", "white");
end

end

function export_q8_8_inputs(path, test_inputs)
% Identical to Python: np.rint(np.asarray(float32_X, float64) * 256)
% followed by signed int16 saturation and row-major two's-complement hex.

scaled = double(single(test_inputs)) * 256;
rounded = round(scaled);
% MATLAB round() uses ties-away-from-zero, whereas NumPy rint uses
% ties-to-even. Correct only exact .5 ties, including negative values.
lower = floor(scaled);
ties = (scaled - lower) == 0.5;
rounded(ties) = lower(ties) + mod(lower(ties), 2);
rounded = min(max(rounded, -32768), 32767);

% A transpose followed by linearization preserves each 4-feature sample.
signed_words = int16(reshape(rounded.', [], 1));
unsigned_words = typecast(signed_words, "uint16");

parent = fileparts(path);
if ~exist(parent, "dir")
    mkdir(parent);
end
[fid, message] = fopen(path, "w");
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, "%04X\n", unsigned_words);
end

function [y, frequency_hz, H] = apply_cd_dsb(x, fs, fc, cd_factor, guard)
% Zero-guarded FFT filtering: isolated-burst approximation, not unpadded
% circular convolution. Increase guard to check convergence near deep notches
% or for substantially longer fibers / wider signal bandwidths.
x = x(:);
nfft = 2^nextpow2(numel(x) + 2*guard);
padded = zeros(nfft, 1);
padded(guard+(1:numel(x))) = x;
frequency_hz = (-nfft/2:nfft/2-1).' * fs/nfft;
H = cos(cd_factor * (fc + frequency_hz).^2);
filtered = ifft(fft(padded) .* ifftshift(H));
y = filtered(guard+(1:numel(x)));
end

function y = apply_rapp_pa(x, enabled, saturation_amplitude, smoothness, phase_limit_deg)
% Rapp AM/AM plus a bounded amplitude-dependent phase rotation.
% u=|x|/A_sat; gain=(1+u^(2*p))^(-1/(2*p));
% phi=phi_limit*u^2/(1+u^2). No phase change at vanishing input.
% Phenomenological driver model, not a calibrated 60 GHz device.
y = x;
if enabled
    u = abs(x)/saturation_amplitude;
    gain = (1 + u.^(2*smoothness)).^(-1/(2*smoothness));
    phase_rad = (phase_limit_deg*pi/180) * u.^2 ./ (1 + u.^2);
    y = x .* gain .* exp(1j*phase_rad);
end
end

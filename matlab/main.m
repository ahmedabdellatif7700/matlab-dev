%% Single-link 60 GHz analog radio-over-fiber baseline
clear; clc; close all;
matlab_dir = fileparts(mfilename("fullpath")); addpath(matlab_dir);

%% 1. Configuration
% Paper values and explicit assumptions: PHYSICAL_LINK.md.
cfg = arof.configuration();
% cfg.modulation_order = 16;       % Start with 16-QAM.
% cfg.drive_rms_v = 0.1;           % Low-drive, noiseless diagnostic.
% cfg.fiber_length_km = 15;

%% 2. Transmitted symbols and known calibration prefix
symbol_data = arof.generate_symbols(cfg);
tx_symbols = symbol_data.symbols;

%% 3. Transmit pulse shaping at the AWG sample rate
baseband_awg = arof.pulse_shape(cfg,tx_symbols);

%% 4. Electrical IF modulation, driver and analog reconstruction
% IF = 0.8 GHz; AWG = 4 GSa/s; internal numerical grid = 20 GSa/s.
tx = arof.upconvert(cfg,symbol_data,baseband_awg);
if_voltage = tx.if_voltage;

%% 5. Single-arm MZM optical-field modulation and second laser tone
optical_tx = arof.modulate(cfg,if_voltage);

%% 6. Optical fiber propagation and explicitly ideal noiseless amplification
optical_rx = arof.propagate(cfg,optical_tx);

%% 7. Photodetection: extract the optical beat near 60 GHz
% The optical/RF carriers are represented analytically, not undersampled.
detected = arof.photodetect(cfg,optical_rx);

%% 8. Millimeter-wave wireless hop
wireless_link = arof.wireless(cfg,detected);

%% 9. Envelope detection to IF and digital baseband conversion
receiver_baseband = arof.downconvert(cfg,wireless_link);

%% 10. Receive matched filter, timing/gain calibration and symbol sampling
% Calibration uses only the known prefix, not the held-out payload.
rx = arof.recover(cfg,receiver_baseband,tx);
rx_symbols = rx.sampled;

%% 11. Unequalized payload BER and continuous-output EVM
metrics = arof.measure(cfg,optical_tx,rx);
disp(metrics);

%% 12. Physical consistency controls and baseline figures (no equalizer)
result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
    "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
run_dir = arof.baseline_report(result);

%% 13. Physical-baseline review checkpoint
% After review: MATLAB dataset export -> FFE/Volterra -> supplied
% nn_training_model/ (Q8.24 MSE/online SGD) -> activation comparisons.
% No NN is invoked while establishing the physical baseline.

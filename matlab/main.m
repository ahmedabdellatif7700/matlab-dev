%% Single-link 60 GHz analog radio-over-fiber baseline
clear; clc; close all;
matlab_dir = fileparts(mfilename("fullpath")); addpath(matlab_dir);

%% 1. Configuration
% Paper values and explicit assumptions: PHYSICAL_LINK.md.
cfg = link_configuration();
% cfg.sweep_payload_symbols = 4096; % Short BER-versus-ROP run.
% cfg.sweep_seeds = 8701;
% cfg.modulation_order = 16;       % Start with 16-QAM.
% cfg.drive_rms_v = 0.1;           % Low-drive, noiseless diagnostic.
% cfg.fiber_length_km = 15;

%% 2. Transmitted symbols and known calibration prefix
symbol_data = arof.generate_symbols(cfg);
tx_symbols = symbol_data.symbols;

%% 3. Transmit pulse shaping at the AWG sample rate
baseband_awg = arof.pulse_shape(cfg,tx_symbols);

%% 4. Electrical IF modulation and analog reconstruction
% IF = 0.8 GHz; AWG = 4 GSa/s; internal numerical grid = 20 GSa/s.
tx = arof.upconvert(cfg,symbol_data,baseband_awg);
% Set the pre-driver peak-to-peak voltage using only the transmitted waveform.
cfg.drive_rms_v = cfg.drive_rms_v*cfg.input_vpp_v/(max(tx.if_before_driver)-min(tx.if_before_driver));
tx = arof.upconvert(cfg,symbol_data,baseband_awg);

%% 4b. Electrical driver before the MZM
tx.if_voltage = arof.driver(cfg,tx.if_before_driver);
if_voltage = tx.if_voltage;

%% 5. Single-arm MZM optical-field modulation and second laser tone
optical_tx = arof.modulate(cfg,if_voltage);

%% 6. Optical fiber propagation and explicitly ideal noiseless amplification
optical_rx = arof.propagate(cfg,optical_tx);

%% 7. Photodetection: extract the optical beat near 60 GHz
% The optical/RF carriers are represented analytically, not undersampled.
% Optional received_optical_power_dbm sets total two-tone power at the PD.
detected = arof.photodetect(cfg,optical_rx);

%% 8. Millimeter-wave wireless hop
wireless_link = arof.wireless(cfg,detected);

%% 9. Envelope detection to IF and digital baseband conversion
receiver_baseband = arof.downconvert(cfg,wireless_link);

%% 10. Receive matched filter, timing/gain calibration and symbol sampling
% Calibration uses only the known prefix, not the held-out payload.
rx = arof.recover(cfg,receiver_baseband,tx);
rx_symbols = rx.sampled;

%% 11. Unequalized payload BER and EVM
metrics = arof.measure(cfg,optical_tx,rx);
fprintf("Unequalized %d-QAM, %.0f mVpp, ROP %.1f dBm:\n", ...
    cfg.modulation_order,1000*cfg.input_vpp_v,cfg.received_optical_power_dbm);
disp(metrics);

%% 12. BER versus received optical power with the same fixed link parameters
power_run_dir = ber_vs_rop(cfg);

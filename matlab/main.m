%% Single-link 60 GHz analog radio-over-fiber baseline
clear; clc; close all;
matlab_dir = fileparts(mfilename("fullpath")); addpath(matlab_dir);

%% 1. Configuration
% Paper values and explicit assumptions: PHYSICAL_LINK.md.
cfg = arof.configuration();
% cfg.profile = "smoke";          % Short received-power run.
% cfg.baseline_report_enabled = true; % Optional physical/numerical report.
% cfg.modulation_order = 16;       % Start with 16-QAM.
% cfg.drive_rms_v = 0.1;           % Low-drive, noiseless diagnostic.
% cfg.fiber_length_km = 15;

%% 2. Transmitted symbols and known calibration prefix
symbol_data = arof.transmitter.generate_symbols(cfg);
tx_symbols = symbol_data.symbols;

%% 3. Transmit pulse shaping at the AWG sample rate
baseband_awg = arof.transmitter.pulse_shape(cfg,tx_symbols);

%% 4. Electrical IF modulation, driver and analog reconstruction
% IF = 0.8 GHz; AWG = 4 GSa/s; internal numerical grid = 20 GSa/s.
tx = arof.transmitter.upconvert(cfg,symbol_data,baseband_awg);
if_voltage = tx.if_voltage;

%% 5. Single-arm MZM optical-field modulation and second laser tone
optical_tx = arof.optical_link.modulate(cfg,if_voltage);

%% 6. Optical fiber propagation and explicitly ideal noiseless amplification
optical_rx = arof.optical_link.propagate(cfg,optical_tx);

%% 7. Photodetection: extract the optical beat near 60 GHz
% The optical/RF carriers are represented analytically, not undersampled.
% Optional received_optical_power_dbm sets total two-tone power at the PD.
detected = arof.optical_link.photodetect(cfg,optical_rx);

%% 8. Millimeter-wave wireless hop
wireless_link = arof.optical_link.wireless(cfg,detected);

%% 9. Envelope detection to IF and digital baseband conversion
receiver_baseband = arof.receiver.downconvert(cfg,wireless_link);

%% 10. Receive matched filter, timing/gain calibration and symbol sampling
% Calibration uses only the known prefix, not the held-out payload.
rx = arof.receiver.recover(cfg,receiver_baseband,tx);
rx_symbols = rx.sampled;

%% 11. Noiseless low-drive control: unequalized payload BER and EVM
metrics = arof.receiver.measure(cfg,optical_tx,rx);
fprintf("Noiseless low-drive control (not the 400 mVpp case):\n");
disp(metrics);

%% 12. Optional low-drive physical consistency report
if cfg.baseline_report_enabled
    result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
        "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
    baseline_dir = arof.baseline_report(result);
end

%% 13. BER versus received optical power, fixed 16-QAM / 400 mVpp
% Separate Fig. 13(c) two-endpoint calibration; predict intermediate powers.
% Its frozen fit does not overwrite the earlier EVM/BER voltage calibration.
if cfg.received_power_comparison_enabled
    power_settings = arof.received_power.settings(cfg.profile);
    power_run_dir = arof.received_power.run(cfg,power_settings);
end

%% 14. Physical-baseline review checkpoint
% After review: MATLAB dataset export -> FFE/Volterra -> supplied
% nn_training_model/ (Q8.24 MSE/online SGD) -> activation comparisons.
% No NN is invoked while establishing the physical baseline.

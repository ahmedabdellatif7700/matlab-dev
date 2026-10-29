function s = paper_settings(profile)
%PAPER_SETTINGS Reproduction attempt; fitted parameters are not measurements.
if nargin<1, profile="study"; end
s.profile = profile;
s.input_vpp_mv = [100 140 160 180 200 220 240 260 300 380 400 420 440 460 480 500];
s.modulations = [4 16 64];
s.seeds = [7301 7302 7303];
s.calibration_seed = 4201;
s.payload_symbols = 16384;
s.calibration_payload_symbols = s.payload_symbols;
s.driver_saturation_grid_v = [2.8 3.0 3.2 3.3 3.4 3.6 4 4.5 5 6];
s.receiver_models = ["linear_amplitude" "compressed_amplitude" "compressed_square_law"];
s.calibration_low_evm_percent = 10.5;
s.calibration_high_evm_percent = 8.1;
s.anchor_input_vpp_mv = 200;
s.anchor_output_vpp_v = 5;
if profile=="smoke"
    s.input_vpp_mv = [100 200 400 500]; s.seeds=7301;
    s.payload_symbols=1024; s.calibration_payload_symbols=1024;
    s.driver_saturation_grid_v=[2.8 3.2 3.6];
end
end

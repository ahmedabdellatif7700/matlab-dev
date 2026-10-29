function s = received_power_settings(profile)
%RECEIVED_POWER_SETTINGS Figure 13(c), unequalized 16-QAM at fixed 400 mVpp.
if nargin<1, profile="study"; end
s.profile=profile; s.modulation_order=16; s.input_vpp_mv=400;
s.calibration_seed=8601; s.calibration_payload_symbols=16384;
s.validation_seeds=8701:8705; s.validation_payload_symbols=65536;
s.gain_grid=12:2:26;
s.calibration_powers_dbm=[-22 -17.5];
if profile=="smoke"
    s.calibration_payload_symbols=2048;
    s.validation_seeds=8701:8702; s.validation_payload_symbols=4096;
end
end

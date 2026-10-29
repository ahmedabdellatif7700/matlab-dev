function s = single_case_settings(profile)
%SINGLE_CASE_SETTINGS One unequalized condition; fit targets are not validation data.
if nargin<1, profile="study"; end
s.profile=profile;
s.modulation_order=16;
s.input_vpp_mv=400;
reference=readtable(fullfile(fileparts(fileparts(mfilename('fullpath'))), ...
    'reference_data','single_case_reference.csv'),'TextType','string');
evm=reference(reference.metric=="evm_percent",:);
ber=reference(reference.metric=="ber",:);
assert(height(evm)==1 && height(ber)==1 && all(reference.modulation==16) && ...
    all(reference.input_vpp_mv==400),'Reference must describe one common operating point.');
s.target_evm_percent=evm.value;
s.evm_read_allowance_pp=evm.read_uncertainty;
s.target_ber=ber.value;
s.ber_read_allowance_log10=ber.read_uncertainty;
s.gain_grid=12.5:.5:25;
s.calibration_seed=8401;
s.calibration_payload_symbols=65536;
s.validation_seeds=8501:8505;
s.validation_payload_symbols=65536;
if profile=="smoke"
    s.calibration_payload_symbols=2048;
    s.validation_payload_symbols=4096;
    s.validation_seeds=8501:8502;
end
end

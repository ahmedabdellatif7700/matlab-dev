function opts = dataset_configuration(profile)
%DATASET_CONFIGURATION Independent sequences for the next equalizer study.
if nargin<1, profile="study"; end
assert(any(string(profile)==["study" "smoke"]),'Unknown dataset profile.');
opts.profile=string(profile);
opts.rop_dbm=[-22 -20 -17.5];
opts.split_names=["train" "validation" "test"];
opts.payload_symbols=[32768 16384 65536];
% Rows are powers, columns are splits. All symbol/noise streams are distinct.
opts.seeds=[9101 9201 9301; 9102 9202 9302; 9103 9203 9303];
opts.tap_offsets_symbols=0:-1:-4; % Current I/Q, then four previous symbols.
if opts.profile=="smoke"
    opts.payload_symbols=[2048 1024 4096];
end
end

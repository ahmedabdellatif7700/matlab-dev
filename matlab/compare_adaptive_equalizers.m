function folder = compare_adaptive_equalizers(profile,run_id)
%COMPARE_ADAPTIVE_EQUALIZERS Export independent data for one A-RoF topology.
% Python runs the Q8.24 LMS/ANN training stages after MATLAB export.
if nargin<1, profile="study"; end
cfg=link_configuration();
% Numerical refinement for nonlinear/bias-offset conditions; AWG remains 4 GSa/s.
cfg.internal_sample_rate_hz=40e9;
if nargin>1, cfg.run_id=string(run_id); end
opts=dataset_configuration(profile);
opts.tap_offsets_symbols=[0 -1];
if string(profile)=="study", opts.payload_symbols=[8192 4096 65536]; end
folder=fullfile(cfg.output_root,'adaptive_equalizers',cfg.run_id);
assert(~isfolder(folder),'Refusing to overwrite comparison data.'); mkdir(folder);
conditions=[struct('name',"reference",'bias_v',2.2,'driver',"linear",'saturation_v',Inf), ...
    struct('name',"bias_offset",'bias_v',2.42,'driver',"linear",'saturation_v',Inf), ...
    struct('name',"driver_compression",'bias_v',2.2,'driver',"rapp_envelope",'saturation_v',3.5)];
rows=struct([]);
for k=1:numel(conditions)
    condition=conditions(k); c=cfg;
    c.run_id=cfg.run_id+"_"+condition.name;
    c.mzm_bias_v=condition.bias_v; c.driver_model=condition.driver;
    c.driver_saturation_v=condition.saturation_v; c.driver_smoothness=2;
    opts.seeds=[15101 15201 15301;15102 15202 15302;15103 15203 15303]+1000*(k-1);
    data_folder=export_datasets(c,opts);
    rows=[rows; struct('condition',condition.name,'dataset_folder',string(data_folder), ...
        'mzm_bias_v',c.mzm_bias_v,'driver_model',c.driver_model, ...
        'driver_saturation_v',c.driver_saturation_v,'input_vpp_v',c.input_vpp_v, ...
        'fiber_length_km',c.fiber_length_km)]; %#ok<AGROW>
    writetable(struct2table(rows),fullfile(folder,'conditions.csv'));
end
fprintf('Export complete. Run Python: python/equalizer.py --workflow adaptive --run "%s"\n',folder);
end

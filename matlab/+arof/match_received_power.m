function folder = match_received_power(cfg,s)
%MATCH_RECEIVED_POWER Independent Figure 13(c) calibration and frozen sweep.
if nargin<2, s=arof.received_power_settings(cfg.profile); end
folder=fullfile(cfg.output_root,'received_power',cfg.run_id);
assert(~isfolder(folder),'Refusing to overwrite received-power run.'); mkdir(folder);
copyfile(fullfile(cfg.root,'matlab','reference_data','received_power_reference.csv'), ...
    fullfile(folder,'paper_reference.csv'));
arof.calibrate_received_power(cfg,s,folder);
folder=arof.sweep_received_power(folder,cfg);
end

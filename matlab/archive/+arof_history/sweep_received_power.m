function folder = sweep_received_power(point_folder,cfg)
%SWEEP_RECEIVED_POWER Full physical-noise runs after two-anchor calibration.
frozen=jsondecode(fileread(fullfile(point_folder,'frozen_calibration.json')));
s=frozen.settings; base=frozen.configuration;
base.driver_model=string(base.driver_model); base.envelope_detector_model=string(base.envelope_detector_model);
paper=readtable(fullfile(point_folder,'paper_reference.csv'),'TextType','string');
folder=fullfile(point_folder,'validation');
assert(~isfolder(folder),'Refusing to overwrite received-power validation.'); mkdir(folder);
copyfile(fullfile(point_folder,'frozen_calibration.json'),fullfile(folder,'frozen_calibration.json'));
copyfile(fullfile(point_folder,'paper_reference.csv'),fullfile(folder,'paper_reference.csv'));
copyfile(fullfile(point_folder,'calibration_search.csv'),fullfile(folder,'calibration_search.csv'));
rows=struct([]); snapshots=struct([]); seeds=s.validation_seeds(:).';
for seed=seeds
    c=base; c.seed=seed; c.noise_seed=seed+900000;
    c=arof_history.paper_operating_point(c,s.modulation_order,seed,s.input_vpp_mv);
    % Power varies only after the fiber/EDFA, at the PD input. Reuse identical
    % transmitted samples and optical propagation within a seed.
    tx=arof_history.transmit(c); optical_tx=arof_history.modulate(c,tx.if_voltage);
    optical_rx=arof_history.propagate(c,optical_tx);
    for k=1:height(paper)
        c.received_optical_power_dbm=paper.received_power_dbm(k);
        detected=arof_history.photodetect(c,optical_rx);
        assert(abs(detected.received_optical_power_dbm-c.received_optical_power_dbm)<1e-10);
        rx=arof_history.receive(c,detected,tx); m=arof_history.measure(c,optical_tx,rx);
        ids=rx.payload_indices; ref=rx.reference(ids); prediction=rx.sampled(ids);
        rows=[rows; struct('seed',seed,'received_power_dbm',c.received_optical_power_dbm, ...
            'actual_received_power_dbm',detected.received_optical_power_dbm, ...
            'evm_percent',m.evm_percent,'bit_errors',m.bit_errors,'bits',m.bits,'ber',m.ber, ...
            'error_energy',sum(abs(prediction-ref).^2),'reference_energy',sum(abs(ref).^2), ...
            'driver_gain',c.driver_voltage_gain,'if_noise_psd',c.if_noise_psd, ...
            'driver_input_vpp_v',max(tx.if_before_driver)-min(tx.if_before_driver), ...
            'mzm_vpp_v',m.drive_vpp_v)]; %#ok<AGROW>
        if seed==seeds(1) && ismember(k,[1 5 height(paper)])
            snapshots=[snapshots; struct('received_power_dbm',c.received_optical_power_dbm, ...
                'seed',seed,'prediction',prediction)]; %#ok<AGROW>
        end
    end
    writetable(struct2table(rows),fullfile(folder,'seed_metrics.csv'));
    fprintf('Received-power sweep seed %d complete.\n',seed);
end
m=struct2table(rows);
assert(isscalar(unique(m.driver_gain)) && isscalar(unique(m.if_noise_psd)));
summary=struct([]);
for k=1:height(paper)
    p=paper(k,:); r=m(m.received_power_dbm==p.received_power_dbm,:);
    errors=sum(r.bit_errors); bits=sum(r.bits); ber=errors/bits; z=1.96;
    center=(ber+z*z/(2*bits))/(1+z*z/bits);
    radius=z*sqrt(ber*(1-ber)/bits+z*z/(4*bits^2))/(1+z*z/bits);
    summary=[summary; struct('received_power_dbm',p.received_power_dbm, ...
        'paper_ber',p.ber,'simulation_ber',ber,'log10_difference',log10(ber/p.ber), ...
        'within_read_allowance',ber>0 && abs(log10(ber/p.ber))<=p.read_uncertainty_log10, ...
        'is_calibration_power',p.role=="calibration",'bit_errors',errors,'bits',bits, ...
        'ber_lower',max(0,center-radius),'ber_upper',center+radius, ...
        'evm_percent',100*sqrt(sum(r.error_energy)/sum(r.reference_energy)), ...
        'evm_seed_std',std(r.evm_percent))]; %#ok<AGROW>
end
writetable(struct2table(summary),fullfile(folder,'curve_summary.csv'));
save(fullfile(folder,'constellations.mat'),'snapshots','-v7');
arof_history.plot_received_power(folder,cfg);
fprintf('Received optical power report: %s\n',folder);
end

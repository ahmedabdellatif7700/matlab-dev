function folder = sweep_single_case(point_folder,cfg,voltages)
%SWEEP_SINGLE_CASE Predict a curve from one frozen fit; no per-point fitting.
if nargin<3, voltages=380:20:500; end
assert(isvector(voltages) && all(isfinite(voltages)) && all(voltages>0) && ...
    numel(unique(voltages))==numel(voltages),'Invalid voltage grid.');
frozen=jsondecode(fileread(fullfile(point_folder,"frozen_calibration.json")));
point_metrics=readtable(fullfile(point_folder,"validation_metrics.csv"));
point_summary=jsondecode(fileread(fullfile(point_folder,"summary.json")));
base=frozen.configuration; s=frozen.settings;
assert(s.modulation_order==16 && s.input_vpp_mv==400,'Expected the agreed 16-QAM/400 mV fit.');
base.driver_model=string(base.driver_model);
base.envelope_detector_model=string(base.envelope_detector_model);
base.payload_symbols=s.validation_payload_symbols; base.noise_enabled=true;
seeds=s.validation_seeds(:).';
assert(isequal(sort(point_metrics.seed(:)),sort(seeds(:))),'Cached validation seeds disagree.');
folder=fullfile(cfg.output_root,"single_case_curve",cfg.run_id);
assert(~isfolder(folder),'Refusing to overwrite curve output.'); mkdir(folder);
copyfile(fullfile(point_folder,"frozen_calibration.json"),fullfile(folder,"frozen_calibration.json"));
reference_file=fullfile(cfg.root,"matlab","reference_data","single_case_curve_reference.csv");
copyfile(reference_file,fullfile(folder,"paper_reference.csv"));
rows=struct([]); snapshots=struct([]);
for seed=seeds
    c=base; c.seed=seed; c.noise_seed=seed+900000;
    unit=arof.transmit(c);
    voltage_scale=c.drive_rms_v/(max(unit.if_before_driver)-min(unit.if_before_driver));
    indices=c.guard_symbols+c.training_symbols+(1:c.payload_symbols);
    reference_energy=sum(abs(unit.symbols(indices)).^2);
    clear unit;
    for voltage=voltages(:).'
        reused=voltage==s.input_vpp_mv;
        if reused
            m=point_metrics(point_metrics.seed==seed,:);
            evm=m.evm_percent; errors=m.bit_errors; bits=m.bits; mzm_vpp=m.mzm_vpp_v;
        else
            c.drive_rms_v=voltage_scale*voltage/1000;
            r=arof.simulate(c);
            evm=r.metrics.evm_percent; errors=r.metrics.bit_errors; bits=r.metrics.bits;
            mzm_vpp=r.metrics.drive_vpp_v;
            if seed==seeds(1) && ismember(voltage,[380 440 500])
                ids=r.rx.payload_indices;
                snapshots=[snapshots; struct("input_vpp_mv",voltage,"seed",seed, ...
                    "prediction",r.rx.sampled(ids))]; %#ok<AGROW>
            end
        end
        rows=[rows; struct("seed",seed,"input_vpp_mv",voltage,"evm_percent",evm, ...
            "bit_errors",errors,"bits",bits,"ber",errors/bits,"mzm_vpp_v",mzm_vpp, ...
            "error_energy",(evm/100)^2*reference_energy,"reference_energy",reference_energy, ...
            "driver_gain",base.driver_voltage_gain,"if_noise_psd",base.if_noise_psd, ...
            "reused_400mv_validation",reused)]; %#ok<AGROW>
    end
    writetable(struct2table(rows),fullfile(folder,"seed_metrics.csv"));
    fprintf('Frozen-parameter 16-QAM curve: seed %d complete.\n',seed);
end
metrics=struct2table(rows);
assert(isscalar(unique(metrics.driver_gain)) && isscalar(unique(metrics.if_noise_psd)), ...
    'Parameters changed during the curve sweep.');
summary_rows=struct([]);
for voltage=sort(voltages(:)).'
    m=metrics(metrics.input_vpp_mv==voltage,:);
    n=sum(m.bits); errors=sum(m.bit_errors); ber=errors/n; z=1.96;
    center=(ber+z*z/(2*n))/(1+z*z/n);
    radius=z*sqrt(ber*(1-ber)/n+z*z/(4*n^2))/(1+z*z/n);
    evm=100*sqrt(sum(m.error_energy)/sum(m.reference_energy));
    summary_rows=[summary_rows; struct("input_vpp_mv",voltage,"evm_percent",evm, ...
        "evm_seed_std",std(m.evm_percent),"ber",ber,"bit_errors",errors,"bits",n, ...
        "ber_lower",max(0,center-radius),"ber_upper",center+radius, ...
        "mzm_vpp_v",mean(m.mzm_vpp_v),"is_calibration_voltage",voltage==400)]; %#ok<AGROW>
end
summary=struct2table(summary_rows);
anchor=summary(summary.input_vpp_mv==400,:);
if ~isempty(anchor)
    assert(abs(anchor.evm_percent-point_summary.evm_percent)<1e-10 && ...
        anchor.ber==point_summary.ber,'Cached anchor changed.');
end
writetable(summary,fullfile(folder,"curve_summary.csv"));
reference=readtable(reference_file,'TextType','string');
comparisons=struct([]);
for k=1:height(summary)
    m=summary(k,:);
    e=reference(reference.input_vpp_mv==m.input_vpp_mv & reference.metric=="evm_percent",:);
    b=reference(reference.input_vpp_mv==m.input_vpp_mv & reference.metric=="ber",:);
    if isempty(e) || isempty(b), continue; end
    comparisons=[comparisons; struct("input_vpp_mv",m.input_vpp_mv, ...
        "paper_evm_percent",e.value,"simulation_evm_percent",m.evm_percent, ...
        "evm_difference_pp",m.evm_percent-e.value, ...
        "evm_within_read_allowance",abs(m.evm_percent-e.value)<=e.read_uncertainty, ...
        "paper_ber",b.value,"simulation_ber",m.ber,"ber_ratio",m.ber/b.value, ...
        "ber_within_read_allowance",m.ber>0 && abs(log10(m.ber/b.value))<=b.read_uncertainty, ...
        "is_calibration_voltage",m.input_vpp_mv==400)]; %#ok<AGROW>
end
writetable(struct2table(comparisons),fullfile(folder,"paper_comparison.csv"));
save(fullfile(folder,"constellations.mat"),"snapshots","-v7");
arof.plot_single_case_curve(folder,cfg);
fprintf('Unequalized curve report: %s\n',folder);
end

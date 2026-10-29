function folder = match_unequalized_case(cfg,s)
%MATCH_UNEQUALIZED_CASE One calibrated 16-QAM point, then fresh Monte Carlo.
% Effective driver gain and noise are inferred; this is NOT a unique device fit.
if nargin<2, s=arof_history.single_case_settings(); end
folder=fullfile(cfg.output_root,"single_unequalized_case",cfg.run_id);
assert(~isfolder(folder),"Refusing to overwrite single-case output."); mkdir(folder);
copyfile(fullfile(cfg.root,"matlab","reference_data","single_case_reference.csv"), ...
    fullfile(folder,"paper_targets.csv"));
base=cfg; base.modulation_order=s.modulation_order; base.driver_model="linear";
base.envelope_detector_model="ideal_amplitude"; base.noise_enabled=false;
base.payload_symbols=s.calibration_payload_symbols;
base=arof_history.paper_operating_point(base,s.modulation_order,s.calibration_seed,s.input_vpp_mv);
trials=struct([]); bracket=[]; previous=[];
for gain=s.gain_grid
    value=objective(gain);
    if ~isfinite(value), continue; end
    if ~isempty(previous) && sign(value)~=sign(previous(2))
        bracket=[previous(1) gain]; break;
    end
    previous=[gain value];
end
assert(~isempty(bracket),"No joint solution in the declared gain range; do not invent extra impairments.");
fitted_gain=fzero(@objective,bracket,optimset('TolX',1e-4,'Display','off'));
objective(fitted_gain);
fit=trials(end);
writetable(struct2table(trials),fullfile(folder,"calibration_search.csv"));
frozen=base; frozen.driver_voltage_gain=fitted_gain; frozen.if_noise_psd=fit.if_noise_psd;
frozen.noise_enabled=true;
write_json(fullfile(folder,"frozen_calibration.json"),struct("settings",s,"configuration",frozen, ...
    "fit",fit,"interpretation","Both target metrics are fitted; fresh seeds test Monte Carlo repeatability, not independent experimental replication."));
% Calibration is frozen above, before validation or held-out errors are read.
rows=struct([]); example=struct(); total_error_energy=0; total_reference_energy=0;
for seed=s.validation_seeds
    c=frozen; c.seed=seed; c.noise_seed=seed+900000;
    c.payload_symbols=s.validation_payload_symbols;
    c=arof_history.paper_operating_point(c,s.modulation_order,seed,s.input_vpp_mv);
    r=arof_history.simulate(c); ids=r.rx.payload_indices;
    prediction=r.rx.sampled(ids); reference=r.rx.reference(ids);
    total_error_energy=total_error_energy+sum(abs(prediction-reference).^2);
    total_reference_energy=total_reference_energy+sum(abs(reference).^2);
    rows=[rows; struct("seed",seed,"evm_percent",r.metrics.evm_percent, ...
        "ber",r.metrics.ber,"bit_errors",r.metrics.bit_errors,"bits",r.metrics.bits, ...
        "driver_input_vpp_v",max(r.tx.if_before_driver)-min(r.tx.if_before_driver), ...
        "mzm_vpp_v",r.metrics.drive_vpp_v)]; %#ok<AGROW>
    if seed==s.validation_seeds(1)
        example=struct("prediction",prediction,"reference",reference);
    end
    fprintf("Single unequalized case, seed %d: EVM %.5f%%, BER %.6g\n", ...
        seed,r.metrics.evm_percent,r.metrics.ber);
end
metrics=struct2table(rows);
writetable(metrics,fullfile(folder,"validation_metrics.csv"));
errors=sum(metrics.bit_errors); bits=sum(metrics.bits); ber=errors/bits; z=1.96;
center=(ber+z*z/(2*bits))/(1+z*z/bits);
radius=z*sqrt(ber*(1-ber)/bits+z*z/(4*bits^2))/(1+z*z/bits);
evm=100*sqrt(total_error_energy/total_reference_energy);
summary=struct("evm_percent",evm,"ber",ber,"bit_errors",errors,"bits",bits, ...
    "ber_lower",max(0,center-radius),"ber_upper",center+radius, ...
    "target_evm_percent",s.target_evm_percent,"target_ber",s.target_ber, ...
    "evm_within_read_allowance",abs(evm-s.target_evm_percent)<=s.evm_read_allowance_pp, ...
    "ber_point_within_read_allowance",ber>0 && abs(log10(ber/s.target_ber))<=s.ber_read_allowance_log10);
write_json(fullfile(folder,"summary.json"),summary);
save(fullfile(folder,"constellation.mat"),"example","-v7");

% Numerical check at the fitted condition, with noise off and a short record.
c=frozen; c.noise_enabled=false; c.payload_symbols=4096;
c=arof_history.paper_operating_point(c,s.modulation_order,s.calibration_seed,s.input_vpp_mv);
a=arof_history.simulate(c); c.internal_sample_rate_hz=2*c.internal_sample_rate_hz;
b=arof_history.simulate(c); ids=a.rx.payload_indices;
difference=sqrt(mean(abs(a.rx.sampled(ids)-b.rx.sampled(ids)).^2)/mean(abs(a.rx.reference(ids)).^2));
write_json(fullfile(folder,"sampling_check.json"),struct("normalized_rms_difference",difference, ...
    "tolerance",1e-3,"passed",difference<1e-3,"evm_base_percent",a.metrics.evm_percent, ...
    "evm_double_rate_percent",b.metrics.evm_percent));

arof_history.report_single_case(folder,cfg);

    function residual = objective(gain)
        c=base; c.driver_voltage_gain=gain;
        r=arof_history.simulate(c); pre=1:c.training_symbols; ids=r.rx.payload_indices;
        raw=r.rx.sampled/r.rx.calibration_gain; target=r.rx.reference;
        p=mean(abs(raw(pre)).^2); cross=mean(conj(raw(pre)).*target(pre));
        py=mean(abs(raw(ids)).^2); cy=mean(conj(raw(ids)).*target(ids)); es=mean(abs(target(ids)).^2);
        expected_mse=@(variance) (abs(cross/(p+variance))^2*(py+variance)- ...
            2*real(conj(cross/(p+variance))*cy)+es)/es;
        if expected_mse(0)>=(s.target_evm_percent/100)^2
            residual=NaN; return;
        end
        variance=fzero(@(v) expected_mse(v)-(s.target_evm_percent/100)^2,[0,p]);
        g=cross/(p+variance);
        expected_ber=arof_history.gaussian_bit_error(g*raw(ids),abs(g)*sqrt(variance/2),target(ids),s.modulation_order);
        trials=[trials; struct("effective_gain",gain,"mzm_vpp_v",r.metrics.drive_vpp_v, ...
            "noiseless_evm_percent",r.metrics.evm_percent,"expected_evm_percent",100*sqrt(expected_mse(variance)), ...
            "expected_ber",expected_ber,"if_noise_psd",variance/(4*c.internal_sample_rate_hz))];
        fprintf("Calibration gain %.5f: expected BER %.6g\n",gain,expected_ber);
        residual=expected_ber-s.target_ber;
    end
end

function write_json(path,value)
write_text(path,jsonencode(value,"PrettyPrint",true));
end
function write_text(path,value)
[fid,msg]=fopen(path,'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',value);
end

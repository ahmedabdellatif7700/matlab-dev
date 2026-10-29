function frozen = calibrate_received_power(cfg,s,folder)
%CALIBRATE_RECEIVED_POWER Two declared BER anchors identify gain and noise.
% No voltage-curve calibration is overwritten. No per-power fitting is used.
reference=readtable(fullfile(folder,'paper_reference.csv'),'TextType','string');
a=reference(reference.role=="calibration",:);
assert(height(a)==2 && isequal(a.received_power_dbm(:).',s.calibration_powers_dbm));
base=cfg; base.modulation_order=s.modulation_order; base.driver_model="linear";
base.envelope_detector_model="ideal_amplitude"; base.noise_enabled=false;
base.payload_symbols=s.calibration_payload_symbols;
base=arof_history.paper_operating_point(base,s.modulation_order,s.calibration_seed,s.input_vpp_mv);
base.received_optical_power_dbm=a.received_power_dbm(1);
trials=struct([]); bracket=[]; previous=[];
for gain=s.gain_grid
    value=objective(gain);
    if ~isfinite(value), continue; end
    if ~isempty(previous) && sign(value)~=sign(previous(2))
        bracket=[previous(1) gain]; break;
    end
    previous=[gain value];
end
writetable(struct2table(trials),fullfile(folder,'calibration_search.csv'));
assert(~isempty(bracket),'No two-anchor fit in the declared gain range; retain the model limitation.');
gain=fzero(@objective,bracket,optimset('TolX',1e-4,'Display','off'));
objective(gain); fit=trials(end);
writetable(struct2table(trials),fullfile(folder,'calibration_search.csv'));
base.driver_voltage_gain=gain; base.if_noise_psd=fit.if_noise_psd;
base.noise_enabled=true; base.payload_symbols=s.validation_payload_symbols;
frozen=struct('settings',s,'configuration',base,'fit',fit, ...
    'interpretation','Separate Figure 13(c) two-anchor calibration, not an independent apparatus reconstruction. ROP is assumed total two-tone power incident on the PD; effective responsivity 1 A/W and normalized RF gain are conventions.');
write_json(fullfile(folder,'frozen_calibration.json'),frozen);

    function residual=objective(gain)
        c=base; c.driver_voltage_gain=gain;
        r=arof_history.simulate(c); pre=1:c.training_symbols; ids=r.rx.payload_indices;
        raw=r.rx.sampled/r.rx.calibration_gain; target=r.rx.reference;
        p=mean(abs(raw(pre)).^2); u=raw/sqrt(p);
        cross=mean(conj(u(pre)).*target(pre));
        probability=@(x) arof_history.gaussian_bit_error(cross/(1+x)*u(ids), ...
            abs(cross/(1+x))*sqrt(x/2),target(ids),s.modulation_order);
        clean=probability(0);
        if clean>=a.ber(1), residual=NaN; return; end
        x=fzero(@(v) probability(v)-a.ber(1),[0 100],optimset('TolX',1e-10));
        % Beat current is proportional to optical watts. Constant receiver-noise
        % PSD therefore gives symbol noise/signal power proportional to P^-2.
        x_high=x*10^(2*(a.received_power_dbm(1)-a.received_power_dbm(2))/10);
        high=probability(x_high);
        trials=[trials; struct('effective_gain',gain,'mzm_vpp_v',r.metrics.drive_vpp_v, ...
            'noiseless_evm_percent',r.metrics.evm_percent,'noiseless_ber',clean, ...
            'expected_low_power_ber',probability(x),'expected_high_power_ber',high, ...
            'if_noise_psd',x*p/(4*c.internal_sample_rate_hz))];
        fprintf('ROP calibration gain %.5f: high-power BER %.6g (target %.6g)\n',gain,high,a.ber(2));
        residual=high-a.ber(2);
    end
end
function write_json(path,value)
[fid,msg]=fopen(path,'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end

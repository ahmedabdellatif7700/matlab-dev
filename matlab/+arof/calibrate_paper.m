function [models,fit_table] = calibrate_paper(cfg,s)
%CALIBRATE_PAPER One gain anchor and two approximate 4-QAM EVM points only.
base=cfg; base.payload_symbols=s.calibration_payload_symbols;
base.seed=s.calibration_seed; base.noise_enabled=false;
base.envelope_detector_model="ideal_amplitude";
linear=base; linear.driver_model="linear"; linear.driver_voltage_gain=25;
linear=identify_noise(linear,s);
rows=[]; candidates=cell(numel(s.driver_saturation_grid_v),1);
for k=1:numel(s.driver_saturation_grid_v)
    c=base; c.driver_model="rapp_envelope";
    c.driver_saturation_v=s.driver_saturation_grid_v(k);
    c.driver_voltage_gain=25/(1-(2.5/c.driver_saturation_v)^4)^(1/4);
    anchor_cfg=arof.paper_operating_point(c,4,s.calibration_seed,200);
    anchor_tx=arof.transmit(anchor_cfg);
    for iteration=1:5
        voltage=arof.driver(c,anchor_tx.if_before_driver);
        optical=arof.modulate(c,voltage);
        vpp=max(optical.drive_voltage)-min(optical.drive_voltage);
        c.driver_voltage_gain=c.driver_voltage_gain*5/vpp;
    end
    c=identify_noise(c,s);
    a=arof.simulate(arof.paper_operating_point(c,4,s.calibration_seed,200));
    high=arof.simulate(arof.paper_operating_point(c,4,s.calibration_seed,500));
    es=mean(abs(high.rx.reference).^2);
    expected_evm=100*sqrt((high.metrics.evm_percent/100)^2+ ...
        4*c.if_noise_psd*c.internal_sample_rate_hz*abs(high.rx.calibration_gain)^2/es);
    rows=[rows; c.driver_saturation_v c.driver_voltage_gain c.if_noise_psd ...
        a.metrics.drive_vpp_v expected_evm abs(expected_evm-s.calibration_high_evm_percent)]; %#ok<AGROW>
    candidates{k}=c;
end
fit_table=array2table(rows,'VariableNames', ...
    ["saturation_peak_v","small_signal_voltage_gain","if_noise_psd", ...
    "anchor_mzm_vpp_v","expected_4qam_500mv_evm_percent","absolute_fit_error_pp"]);
[~,best]=min(rows(:,6)); compressed=candidates{best};
% The alternative detector uses the SAME driver, with one low-drive noise fit
% because its detector-output units differ. No 16/64-QAM points are fitted.
square=compressed; square.envelope_detector_model="square_law";
square=identify_noise(square,s);
models={linear,compressed,square};
end

function cfg = identify_noise(cfg,s)
low=arof.simulate(arof.paper_operating_point(cfg,4,s.calibration_seed,100));
es=mean(abs(low.rx.reference).^2);
remaining=(s.calibration_low_evm_percent/100)^2-(low.metrics.evm_percent/100)^2;
assert(remaining>=0,"Deterministic low-drive EVM already exceeds calibration target.");
% Real IF noise -> complex downmix by 2 -> unit-energy matched RRC.
cfg.if_noise_psd=remaining*es/(4*abs(low.rx.calibration_gain)^2*cfg.internal_sample_rate_hz);
end

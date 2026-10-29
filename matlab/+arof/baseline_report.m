function run_dir = baseline_report(result)
%BASELINE_REPORT Physical consistency controls and unequalized diagnostics.
% These controls do not calibrate experimental noise, powers or device curves.
cfg = result.configuration;
run_dir = fullfile(cfg.output_root,cfg.run_id);
assert(~isfolder(run_dir),"Refusing to overwrite baseline results."); mkdir(run_dir);
check_names = strings(0,1); errors = []; tolerances = [];

% Balanced single-arm field must reproduce its intensity transfer exactly.
c = cfg; c.mzm_bandwidth_hz = Inf;
v = linspace(-c.mzm_vpi_v,c.mzm_vpi_v,1001).';
o = arof.optical_link.modulate(c,v);
expected = (1+cos(pi*(c.mzm_bias_v+v)/c.mzm_vpi_v))/2;
add_check("MZM field/intensity identity",max(abs(abs(o.modulated_field).^2-expected)),1e-12);
add_check("Passive normalized MZM",max(0,max(abs(o.modulated_field))-1),1e-12);

% Direct algebraic square-law detection must agree with the extracted beat.
u = result.optical_rx.modulated_field(1:1024);
lo = result.optical_rx.lo_field;
t = (0:numel(u)-1).'/cfg.internal_sample_rate_hz;
carrier = exp(1j*2*pi*cfg.rf_carrier_hz*t);
intensity = abs(u+lo*carrier).^2;
beat = real(2*lo.*conj(u).*carrier);
add_check("Square-law beat identity",max(abs(intensity-abs(u).^2-abs(lo)^2-beat)),1e-12);

% A bin-centered optical tone has a known complex dispersion response.
c = cfg; c.filter_guard_samples = 0; c.ideal_edfa_gain_db = 0;
n = 4096; f = 100*cfg.internal_sample_rate_hz/n;
tone = exp(1j*2*pi*(0:n-1).'*100/n);
input = struct("modulated_field",tone,"lo_field",1);
output = arof.optical_link.propagate(c,input);
beta2 = -(c.dispersion_ps_nm_km*1e-6)*c.modulated_wavelength_m^2/(2*pi*299792458);
gain = 10^(-c.attenuation_db_km*c.fiber_length_km/20);
expected = gain*exp(-1j*beta2*c.fiber_length_km*1000*(2*pi*f)^2/2);
add_check("Optical tone dispersion",max(abs(output.modulated_field-expected*tone)),1e-11);
expected_lo = gain*exp(-1j*beta2*c.fiber_length_km*1000*(2*pi*c.rf_carrier_hz)^2/2);
add_check("LO dispersion phase",abs(output.lo_field-expected_lo),1e-12);
c.fiber_length_km = 0;
output = arof.optical_link.propagate(c,input);
add_check("Zero-length optical identity",max(abs(output.modulated_field-tone)),1e-12);

% Ideal amplitude-modulated RF envelope isolates waveform/recovery numerics.
rf = max(abs(result.tx.if_voltage))+1+result.tx.if_voltage;
control_rx = arof.receiver.run(cfg,struct("rf_envelope",rf),result.tx);
loopback = arof.receiver.measure(cfg,result.optical_tx,control_rx);
add_check("Ideal IF loopback EVM fraction",loopback.evm_percent/100,1e-3);
add_check("Ideal IF loopback BER",loopback.ber,0);

% Double the internal optical-envelope grid, keeping the same AWG waveform.
c = cfg; c.internal_sample_rate_hz = 2*cfg.internal_sample_rate_hz;
fine = arof.simulate(c);
ids = result.rx.payload_indices;
scale = sqrt(mean(abs(result.rx.reference(ids)).^2));
difference = sqrt(mean(abs(fine.rx.sampled(ids)-result.rx.sampled(ids)).^2))/scale;
add_check("Internal sampling convergence",difference,1e-3);

% Force a larger FFT period, not merely a shifted guard on the same FFT grid.
c = cfg; c.filter_guard_samples = max(2*cfg.filter_guard_samples, ...
    2^nextpow2(numel(result.tx.if_voltage)));
guarded = arof.simulate(c);
difference_guard = sqrt(mean(abs(guarded.rx.sampled(ids)-result.rx.sampled(ids)).^2))/scale;
add_check("FFT period/guard convergence",difference_guard,1e-6);
passed = errors <= tolerances;
checks = table(check_names,errors,tolerances,passed, ...
    'VariableNames',["check","error","tolerance","passed"]);
writetable(checks,fullfile(run_dir,"physical_checks.csv"));
assert(all(passed),"Physical/numerical consistency check failed. Inspect physical_checks.csv.");

% This is an amplitude anchor only, not a calibrated reproduction of Fig. 9.
c = cfg; c.drive_rms_v = cfg.drive_rms_v*5/result.metrics.drive_vpp_v;
anchor = arof.simulate(c);
labels = ["low_drive_baseline";"ideal_IF_loopback";"double_internal_sampling";"reported_drive_amplitude_anchor"];
values = [result.metrics;loopback;fine.metrics;anchor.metrics];
evm = [values.evm_percent].'; ber = [values.ber].';
bit_errors = [values.bit_errors].'; bits = [values.bits].';
drive_rms = [values.drive_rms_v].'; drive_vpp = [values.drive_vpp_v].';
metrics = table(labels,evm,ber,bit_errors,bits,drive_rms,drive_vpp, ...
    'VariableNames',["diagnostic","evm_percent","ber","bit_errors","bits","drive_rms_v","drive_vpp_v"]);
writetable(metrics,fullfile(run_dir,"baseline_metrics.csv"));
write_text(fullfile(run_dir,"configuration.json"),jsonencode(cfg,"PrettyPrint",true));
copyfile(fullfile(cfg.root,"matlab","PHYSICAL_LINK.md"),fullfile(run_dir,"PHYSICAL_LINK.md"));
save(fullfile(run_dir,"baseline.mat"),"result","checks","metrics","-v7");

figure("Visible",cfg.visible,"Position",[100 100 1100 700]);
tiledlayout(2,2,"TileSpacing","compact");
nexttile; spectrum(result.tx.if_awg,cfg.awg_sample_rate_hz,0,[-1.5 1.5]);
title("Electrical IF at AWG output");
nexttile; spectrum(result.optical_tx.modulated_field,cfg.internal_sample_rate_hz,0,[-2 2]);
xlabel("Offset from modulated laser (GHz)");
title("Modulated optical band; second tone at +"+sprintf("%.3f",cfg.rf_carrier_hz/1e9)+" GHz");
nexttile; spectrum(result.detected.rf_envelope,cfg.internal_sample_rate_hz,cfg.rf_carrier_hz, ...
    cfg.rf_carrier_hz/1e9+[-2 2]); title("Detected RF band (analytic envelope)");
nexttile; scatter(real(result.rx.sampled(ids)),imag(result.rx.sampled(ids)),5,"."); hold on;
ideal = qammod((0:cfg.modulation_order-1).',cfg.modulation_order,"gray");
plot(real(ideal),imag(ideal),"ko","MarkerSize",8); axis equal; grid on;
xlabel("I"); ylabel("Q"); title(sprintf("No equalizer: EVM %.4f%%",result.metrics.evm_percent));
exportgraphics(gcf,fullfile(run_dir,"baseline_stages.png"),"Resolution",160); close(gcf);

figure("Visible",cfg.visible,"Position",[100 100 900 400]);
tiledlayout(1,2);
for k = 1:2
    nexttile;
    if k == 1, r = result; label = "Low drive"; else, r = anchor; label = "5 Vpp amplitude anchor"; end
    scatter(real(r.rx.sampled(ids)),imag(r.rx.sampled(ids)),5,"."); hold on;
    plot(real(ideal),imag(ideal),"ko","MarkerSize",8); axis equal; grid on;
    xlabel("I"); ylabel("Q"); title(sprintf("%s: EVM %.3f%%",label,r.metrics.evm_percent));
end
exportgraphics(gcf,fullfile(run_dir,"drive_diagnostic.png"),"Resolution",160); close(gcf);
text = sprintf(join(["# Physical baseline review\n\n" ...
    "Single 0.4 Gbaud, 16-QAM optical-beating link; no equalizer.\n\n" ...
    "Calculated optical separation: %.6f GHz. Low-drive EVM: %.6f%%; errors: %d/%d bits.\n" ...
    "Actual MZM drive: %.6f V RMS, %.6f Vpp. Prefix-only timing offset: %.6f internal samples.\n\n" ...
    "All %d physical/numerical checks pass. Doubling the internal sample rate changes the normalized " ...
    "recovered payload by %.6g RMS; increasing the FFT period/padding changes it by %.6g.\n\n" ...
    "The 5 Vpp run matches only the approximate drive amplitude mentioned in the paper: " ...
    "EVM %.6f%%, errors %d/%d. It is not a reproduction of measured noise or sensitivity.\n\n" ...
    "Optical fields have normalized power. RF envelope detection uses an ideal amplitude law, " ...
    "not a measured detector response. EDFA gain is ideal/noiseless; receiver noise, " ...
    "laser noise, driver compression and PD saturation are not calibrated or simulated. " ...
    "Bandwidth responses and SMF constants are explicit approximations. Zero observed " ...
    "errors in this noiseless diagnostic do not establish a practical zero BER.\n\n" ...
    "See [PHYSICAL_LINK.md](PHYSICAL_LINK.md) for assumptions. Review this baseline before dataset export " ...
    "or FFE/Volterra/NN comparisons.\n"],""),cfg.rf_carrier_hz/1e9,result.metrics.evm_percent, ...
    result.metrics.bit_errors,result.metrics.bits,result.metrics.drive_rms_v,result.metrics.drive_vpp_v, ...
    result.rx.timing_offset_samples,height(checks),difference,difference_guard, ...
    anchor.metrics.evm_percent,anchor.metrics.bit_errors,anchor.metrics.bits);
text = string(text)+newline+"![Signal stages](baseline_stages.png)"+newline+newline+ ...
    "![Drive diagnostic](drive_diagnostic.png)"+newline;
write_text(fullfile(run_dir,"README.md"),text);
disp(metrics); fprintf("Physical baseline results: %s\n",run_dir);

    function add_check(name,error_value,tolerance)
        check_names(end+1,1) = name;
        errors(end+1,1) = error_value;
        tolerances(end+1,1) = tolerance;
    end
end

function spectrum(signal,fs,center,limits)
n = 2^nextpow2(numel(signal));
f = ((-n/2:n/2-1).'*fs/n+center)/1e9;
power = abs(fftshift(fft(signal,n))).^2;
relative_db = 10*log10(max(power/max(power),1e-14));
keep = f>=limits(1) & f<=limits(2);
plot(f(keep),relative_db(keep)); ylim([-100 3]); xlim(limits); grid on;
xlabel("Frequency (GHz)"); ylabel("Relative spectrum (dB)");
end

function write_text(path,value)
[fid,msg] = fopen(path,"w"); assert(fid>=0,"%s",msg);
cleanup = onCleanup(@() fclose(fid)); fprintf(fid,"%s",value);
end

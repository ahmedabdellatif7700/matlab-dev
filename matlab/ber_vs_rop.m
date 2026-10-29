function folder = ber_vs_rop(cfg)
%BER_VS_ROP Simulate the physical stages at fixed drive, gain and noise PSD.
% No paper data or parameter fitting enters this runner.
if nargin<1, cfg=link_configuration(); end
folder=fullfile(cfg.output_root,'ber_vs_rop',cfg.run_id);
assert(~isfolder(folder),'Refusing to overwrite BER-versus-ROP results.'); mkdir(folder);
[fid,msg]=fopen(fullfile(folder,'configuration.json'),'w'); assert(fid>=0,'%s',msg);
fprintf(fid,'%s\n',jsonencode(cfg,'PrettyPrint',true)); fclose(fid);
rows=struct([]);
for seed=cfg.sweep_seeds(:).'
    c=cfg; c.seed=seed; c.noise_seed=seed+900000;
    c.payload_symbols=cfg.sweep_payload_symbols;
    c.drive_rms_v=0.1;
    data=arof.generate_symbols(c);
    baseband=arof.pulse_shape(c,data.symbols);
    tx=arof.upconvert(c,data,baseband);
    c.drive_rms_v=c.drive_rms_v*c.input_vpp_v/(max(tx.if_before_driver)-min(tx.if_before_driver));
    tx=arof.upconvert(c,data,baseband);
    tx.if_voltage=arof.driver(c,tx.if_before_driver);
    optical_tx=arof.modulate(c,tx.if_voltage);
    optical_rx=arof.propagate(c,optical_tx);
    for power=cfg.rop_dbm(:).'
        c.received_optical_power_dbm=power;
        detected=arof.photodetect(c,optical_rx);
        link=arof.wireless(c,detected);
        baseband_rx=arof.downconvert(c,link);
        rx=arof.recover(c,baseband_rx,tx);
        m=arof.measure(c,optical_tx,rx);
        ids=rx.payload_indices;
        rows=[rows; struct('seed',seed,'received_power_dbm',power, ...
            'bit_errors',m.bit_errors,'bits',m.bits,'ber',m.ber, ...
            'evm_percent',m.evm_percent, ...
            'error_energy',sum(abs(rx.sampled(ids)-rx.reference(ids)).^2), ...
            'reference_energy',sum(abs(rx.reference(ids)).^2))]; %#ok<AGROW>
    end
    writetable(struct2table(rows),fullfile(folder,'seed_metrics.csv'));
    fprintf('BER versus ROP: seed %d complete.\n',seed);
end
metrics=struct2table(rows); summary=struct([]);
for power=cfg.rop_dbm(:).'
    m=metrics(metrics.received_power_dbm==power,:);
    errors=sum(m.bit_errors); bits=sum(m.bits); ber=errors/bits; z=1.96;
    center=(ber+z*z/(2*bits))/(1+z*z/bits);
    radius=z*sqrt(ber*(1-ber)/bits+z*z/(4*bits^2))/(1+z*z/bits);
    summary=[summary; struct('received_power_dbm',power,'ber',ber, ...
        'bit_errors',errors,'bits',bits,'ber_lower',max(0,center-radius), ...
        'ber_upper',center+radius, ...
        'evm_percent',100*sqrt(sum(m.error_energy)/sum(m.reference_energy)))]; %#ok<AGROW>
end
summary=struct2table(summary);
writetable(summary,fullfile(folder,'ber_vs_rop.csv'));
figure('Visible',cfg.visible); hold on;
positive=summary.ber>0;
semilogy(summary.received_power_dbm(positive),summary.ber(positive),'o-','LineWidth',1.5);
semilogy(summary.received_power_dbm(~positive),summary.ber_upper(~positive),'v');
set(gca,'YScale','log'); grid on;
xlabel('Received optical power at PD (dBm)'); ylabel('BER');
title(sprintf('Unequalized %d-QAM, %.0f mVpp',cfg.modulation_order,1000*cfg.input_vpp_v));
exportgraphics(gcf,fullfile(folder,'ber_vs_rop.png'),'Resolution',180);
exportgraphics(gcf,fullfile(folder,'ber_vs_rop.pdf'),'ContentType','vector'); close(gcf);
fprintf('BER-versus-ROP results: %s\n',folder);
end

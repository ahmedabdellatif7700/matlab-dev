function plot_paper_attempt(cfg,s,metrics,reference,snapshots,folder)
%PLOT_PAPER_ATTEMPT Reference readings are approximate; no equalized curves.
colors=lines(numel(s.receiver_models));
figure("Visible",cfg.visible,"Position",[50 50 1350 420]); tiledlayout(1,3);
for modulation=s.modulations
    nexttile; hold on;
    for k=1:numel(s.receiver_models)
        m=metrics(metrics.model==s.receiver_models(k) & metrics.modulation==modulation,:);
        plot(m.input_vpp_mv,m.evm_percent,'-o','Color',colors(k,:), ...
            'MarkerSize',3,'DisplayName',strrep(s.receiver_models(k),"_"," "));
    end
    p=reference(reference.figure==9 & reference.modulation==modulation,:);
    errorbar(p.input_vpp_mv,p.value,p.read_uncertainty,'ks','LineStyle','none', ...
        'MarkerSize',5,'DisplayName','Paper Fig. 9: approximate read-off');
    xlabel('Input before driver (mVpp)'); ylabel('Continuous-output EVM (%)');
    title(sprintf('%d-QAM',modulation)); grid on; legend('Location','northwest','FontSize',7);
end
exportgraphics(gcf,fullfile(folder,"evm_comparison.png"),'Resolution',160); close(gcf);
figure("Visible",cfg.visible,"Position",[50 50 1100 440]); tiledlayout(1,2);
for modulation=[16 64]
    nexttile; hold on;
    p=reference(reference.metric=="ber" & reference.modulation==modulation,:);
    for k=1:numel(s.receiver_models)
        m=metrics(metrics.model==s.receiver_models(k) & metrics.modulation==modulation & ...
            metrics.input_vpp_mv>=min(p.input_vpp_mv) & metrics.input_vpp_mv<=max(p.input_vpp_mv),:);
        nonzero=m.bit_errors>0;
        plot(m.input_vpp_mv(nonzero),m.ber(nonzero),'-o','Color',colors(k,:), ...
            'MarkerSize',4,'DisplayName',strrep(s.receiver_models(k),"_"," "));
        plot(m.input_vpp_mv(~nonzero),m.ber_high(~nonzero),'v','Color',colors(k,:), ...
            'HandleVisibility','off');
    end
    plot(p.input_vpp_mv,p.value,'ks--','DisplayName','Paper: approximate read-off');
    set(gca,'YScale','log'); grid on; xlabel('Input before driver (mVpp)'); ylabel('Unequalized BER');
    title(sprintf('%d-QAM; zero errors shown at upper bound',modulation));
    legend('Location','best','FontSize',7);
end
exportgraphics(gcf,fullfile(folder,"ber_comparison.png"),'Resolution',160); close(gcf);
figure("Visible",cfg.visible,"Position",[50 50 750 450]); hold on;
for k=1:2
    m=metrics(metrics.model==s.receiver_models(k) & metrics.modulation==4,:);
    plot(m.input_vpp_mv,m.mzm_vpp_v,'-o','DisplayName',strrep(s.receiver_models(k),"_"," "));
end
plot(200,5,'ks','MarkerFaceColor','k','DisplayName','Paper: approximately 200 mV to 5 V');
xlabel('Input before driver (mVpp)'); ylabel('Sampled MZM electrode drive (Vpp)');
title('Linear extrapolation versus inferred driver compression'); grid on; legend('Location','northwest');
exportgraphics(gcf,fullfile(folder,"driver_mapping.png"),'Resolution',160); close(gcf);
if ~isempty(snapshots)
    figure("Visible",cfg.visible,"Position",[50 50 1100 650]); tiledlayout(2,3);
    for k=1:numel(snapshots)
        nexttile; r=snapshots(k);
        scatter(real(r.prediction),imag(r.prediction),2,'.'); hold on;
        ideal=qammod((0:r.modulation-1).',r.modulation,'gray');
        plot(real(ideal),imag(ideal),'ko','MarkerSize',4); axis equal; grid on;
        title(sprintf('%d-QAM, %d mVpp, no equalizer',r.modulation,r.input_vpp_mv));
        xlabel('I'); ylabel('Q');
    end
    exportgraphics(gcf,fullfile(folder,"unequalized_constellations.png"),'Resolution',160); close(gcf);
end
end

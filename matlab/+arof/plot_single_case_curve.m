function plot_single_case_curve(folder,cfg)
%PLOT_SINGLE_CASE_CURVE Approximate paper traces vs frozen-fit predictions.
m=readtable(fullfile(folder,"curve_summary.csv"));
p=readtable(fullfile(folder,"paper_reference.csv"),'TextType','string');
comparison=readtable(fullfile(folder,"paper_comparison.csv"));
frozen=jsondecode(fileread(fullfile(folder,"frozen_calibration.json")));
figure('Visible',cfg.visible,'Position',[100 100 1150 460]); tiledlayout(1,2);
nexttile; evm_panel(m,p); nexttile; ber_panel(m,p);
sgtitle('Unequalized 16-QAM: fixed 400 mVpp calibration; other points not fitted');
exportgraphics(gcf,fullfile(folder,"curve_comparison.png"),'Resolution',180);
exportgraphics(gcf,fullfile(folder,"curve_comparison.pdf"),'ContentType','vector'); close(gcf);
figure('Visible',cfg.visible,'Position',[100 100 700 480]); evm_panel(m,p);
exportgraphics(gcf,fullfile(folder,"evm_curve.png"),'Resolution',180);
exportgraphics(gcf,fullfile(folder,"evm_curve.pdf"),'ContentType','vector'); close(gcf);
figure('Visible',cfg.visible,'Position',[100 100 700 480]); ber_panel(m,p);
exportgraphics(gcf,fullfile(folder,"ber_curve.png"),'Resolution',180);
exportgraphics(gcf,fullfile(folder,"ber_curve.pdf"),'ContentType','vector'); close(gcf);
snapshots=struct([]);
if isfile(fullfile(folder,"constellations.mat"))
    load(fullfile(folder,"constellations.mat"),'snapshots');
end
if ~isempty(snapshots)
    figure('Visible',cfg.visible,'Position',[100 100 1150 390]); tiledlayout(1,numel(snapshots));
    ideal=qammod((0:15).',16,'gray');
    for k=1:numel(snapshots)
        nexttile; r=snapshots(k);
        scatter(real(r.prediction),imag(r.prediction),2,'.'); hold on;
        plot(real(ideal),imag(ideal),'ko','MarkerSize',5); axis equal; grid on;
        xlabel('I'); ylabel('Q'); title(sprintf('%d mVpp; seed %d',r.input_vpp_mv,r.seed));
    end
    exportgraphics(gcf,fullfile(folder,"curve_constellations.png"),'Resolution',180); close(gcf);
end
lines=["# Unequalized 16-QAM voltage curve"; ""; ...
    "This tests extrapolation from the calibrated 400 mVpp point. Driver gain and aggregate IF-noise PSD remain fixed; no other voltage is fitted. No equalizer or NN is used."; ""; ...
    sprintf('Frozen effective driver gain: %.8g; IF noise PSD: %.8g normalized units squared/Hz.', ...
        frozen.configuration.driver_voltage_gain,frozen.configuration.if_noise_psd); ...
    ""; "| Input mVpp | Paper EVM (%) | Simulation EVM (%) | Paper BER | Simulation BER | Fitted voltage? |"; ...
    "|---:|---:|---:|---:|---:|---|"];
for k=1:height(comparison)
    c=comparison(k,:);
    lines(end+1)=sprintf('| %d | %.3f | %.3f | %.4g | %.4g | %d |', ...
        c.input_vpp_mv,c.paper_evm_percent,c.simulation_evm_percent,c.paper_ber, ...
        c.simulation_ber,c.is_calibration_voltage); %#ok<AGROW>
end
predicted=~comparison.is_calibration_voltage;
n=nnz(predicted); e=nnz(comparison.evm_within_read_allowance & predicted);
b=nnz(comparison.ber_within_read_allowance & predicted);
lines=[lines; ""; sprintf('Outside the calibration voltage, %d/%d EVM points and %d/%d BER points fall within the declared paper read-off allowances. This descriptive count is not an experimental validation test.',e,n,b,n); ...
    ""; "![Curve comparison](curve_comparison.png)"; ""; ...
    "The paper points are approximate visual readings of Figures 9 and 11. Read-off allowances are not experimental confidence intervals. EVM simulation bars show seed standard deviation; BER bars show nominal Wilson intervals, assuming independent bits. Temporal dependence limits those intervals."; ...
    ""; "Each voltage uses the same preselected seeds and payload length as the single-point check. The existing 400 mVpp validation samples are reused exactly. Within a seed, waveform and noise draws are shared across voltages for paired comparisons; voltage points are not independent replicates."; ...
    ""; "Only a single operating-point gain and noise level were identified. The driver transfer curve and common received-power/noise conditions across paper figures are not established. Agreement at 400 mVpp does not establish agreement elsewhere. Deviations are retained rather than removed by per-point fitting."; ...
    ""; "PNG plots and vector PDF versions are provided. CSV files preserve per-seed counts, pooled metrics and differences from the paper."];
[fid,msg]=fopen(fullfile(folder,"README.md"),'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',join(lines,newline));
end

function evm_panel(m,p)
hold on; paper=p(p.metric=="evm_percent",:);
errorbar(paper.input_vpp_mv,paper.value,paper.read_uncertainty,'ks-', ...
    'DisplayName','Paper Fig. 9 (approximate readings)','MarkerSize',5);
errorbar(m.input_vpp_mv,m.evm_percent,m.evm_seed_std,'o-', ...
    'DisplayName','Simulation; fixed parameters','LineWidth',1.4,'MarkerSize',5);
a=m(m.input_vpp_mv==400,:);
plot(a.input_vpp_mv,a.evm_percent,'r*','MarkerSize',10,'DisplayName','400 mVpp calibration');
xlabel('Input voltage before driver (mVpp)'); ylabel('EVM (%)');
title('Unequalized EVM'); grid on; legend('Location','northwest','FontSize',8);
end

function ber_panel(m,p)
hold on; paper=p(p.metric=="ber",:);
errorbar(paper.input_vpp_mv,log10(paper.value),paper.read_uncertainty,'ks-', ...
    'DisplayName','Paper Fig. 11 (approximate readings)','MarkerSize',5);
positive=m.ber>0; y=NaN(height(m),1); y(positive)=log10(m.ber(positive));
plot(m.input_vpp_mv,y,'o-','DisplayName','Simulation; fixed parameters','LineWidth',1.4,'MarkerSize',5);
errorbar(m.input_vpp_mv(positive),y(positive), ...
    y(positive)-log10(m.ber_lower(positive)),log10(m.ber_upper(positive))-y(positive), ...
    'LineStyle','none','HandleVisibility','off');
plot(m.input_vpp_mv(~positive),log10(m.ber_upper(~positive)),'v','HandleVisibility','off');
a=m(m.input_vpp_mv==400 & m.ber>0,:);
plot(a.input_vpp_mv,log10(a.ber),'r*','MarkerSize',10,'DisplayName','400 mVpp calibration');
xlabel('Input voltage before driver (mVpp)'); ylabel('log_{10}(BER)');
title('Unequalized BER'); grid on; legend('Location','northwest','FontSize',8);
end

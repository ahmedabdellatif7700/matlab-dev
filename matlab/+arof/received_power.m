classdef received_power
%RECEIVED_POWER Figure 13(c) calibration, frozen power sweep and report.
    methods (Static)
        function s = settings(profile)
            %SETTINGS Figure 13(c), unequalized 16-QAM at fixed 400 mVpp.
            if nargin<1, profile="study"; end
            s.profile=profile; s.modulation_order=16; s.input_vpp_mv=400;
            s.calibration_seed=8601; s.calibration_payload_symbols=16384;
            s.validation_seeds=8701:8705; s.validation_payload_symbols=65536;
            s.gain_grid=12:2:26;
            s.calibration_powers_dbm=[-22 -17.5];
            if profile=="smoke"
                s.calibration_payload_symbols=2048;
                s.validation_seeds=8701:8702; s.validation_payload_symbols=4096;
            end
        end

        function frozen = calibrate(cfg,s,folder)
            %CALIBRATE Two declared BER anchors identify gain and noise.
            % No voltage-curve calibration is overwritten. No per-power fitting is used.
            reference=readtable(fullfile(folder,'paper_reference.csv'),'TextType','string');
            a=reference(reference.role=="calibration",:);
            assert(height(a)==2 && isequal(a.received_power_dbm(:).',s.calibration_powers_dbm));
            base=cfg; base.modulation_order=s.modulation_order; base.driver_model="linear";
            base.envelope_detector_model="ideal_amplitude"; base.noise_enabled=false;
            base.payload_symbols=s.calibration_payload_symbols;
            base=arof.transmitter.operating_point(base,s.modulation_order,s.calibration_seed,s.input_vpp_mv);
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
                    r=arof.simulate(c); pre=1:c.training_symbols; ids=r.rx.payload_indices;
                    raw=r.rx.sampled/r.rx.calibration_gain; target=r.rx.reference;
                    p=mean(abs(raw(pre)).^2); u=raw/sqrt(p);
                    cross=mean(conj(u(pre)).*target(pre));
                    probability=@(x) gaussian_bit_error(cross/(1+x)*u(ids), ...
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

        function folder = sweep(point_folder,cfg)
            %SWEEP Full physical-noise runs after two-anchor calibration.
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
                c=arof.transmitter.operating_point(c,s.modulation_order,seed,s.input_vpp_mv);
                % Power varies only after the fiber/EDFA, at the PD input. Reuse identical
                % transmitted samples and optical propagation within a seed.
                tx=arof.transmitter.run(c); optical_tx=arof.optical_link.modulate(c,tx.if_voltage);
                optical_rx=arof.optical_link.propagate(c,optical_tx);
                for k=1:height(paper)
                    c.received_optical_power_dbm=paper.received_power_dbm(k);
                    detected=arof.optical_link.photodetect(c,optical_rx);
                    assert(abs(detected.received_optical_power_dbm-c.received_optical_power_dbm)<1e-10);
                    rx=arof.receiver.run(c,detected,tx); m=arof.receiver.measure(c,optical_tx,rx);
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
            arof.received_power.plot(folder,cfg);
            fprintf('Received optical power report: %s\n',folder);
        end

        function plot(folder,cfg)
            %PLOT Paper Figure 13(c) versus frozen physical-link settings.
            m=readtable(fullfile(folder,'curve_summary.csv'));
            p=readtable(fullfile(folder,'paper_reference.csv'),'TextType','string');
            frozen=jsondecode(fileread(fullfile(folder,'frozen_calibration.json')));
            figure('Visible',cfg.visible,'Position',[100 100 800 520]); hold on;
            errorbar(p.received_power_dbm,p.log10_ber,p.read_uncertainty_log10,'ks-', ...
                'DisplayName','Paper Fig. 13(c), unequalized (approx.)','MarkerSize',5);
            y=NaN(height(m),1); positive=m.simulation_ber>0;
            y(positive)=log10(m.simulation_ber(positive));
            plot(m.received_power_dbm,y,'o-','LineWidth',1.5, ...
                'DisplayName','Physical model; fixed gain and noise');
            errorbar(m.received_power_dbm(positive),y(positive), ...
                y(positive)-log10(m.ber_lower(positive)),log10(m.ber_upper(positive))-y(positive), ...
                'LineStyle','none','HandleVisibility','off');
            plot(m.received_power_dbm(~positive),log10(m.ber_upper(~positive)),'v','HandleVisibility','off');
            a=logical(m.is_calibration_power);
            plot(m.received_power_dbm(a),y(a),'r*','MarkerSize',10,'DisplayName','Two calibration powers');
            xlabel('Received optical power at PD (dBm)'); ylabel('log_{10}(BER)'); grid on;
            title({'Unequalized 16-QAM, 400 mVpp';'Two endpoint anchors fitted; intermediate powers predicted'});
            legend('Location','northeast','FontSize',9);
            exportgraphics(gcf,fullfile(folder,'ber_received_power.png'),'Resolution',180);
            exportgraphics(gcf,fullfile(folder,'ber_received_power.pdf'),'ContentType','vector'); close(gcf);
            if isfile(fullfile(folder,'constellations.mat'))
                load(fullfile(folder,'constellations.mat'),'snapshots');
                figure('Visible',cfg.visible,'Position',[100 100 1150 390]); tiledlayout(1,numel(snapshots));
                ideal=qammod((0:15).',16,'gray');
                for k=1:numel(snapshots)
                    nexttile; r=snapshots(k); scatter(real(r.prediction),imag(r.prediction),2,'.'); hold on;
                    plot(real(ideal),imag(ideal),'ko','MarkerSize',5); axis equal; grid on;
                    xlabel('I'); ylabel('Q'); title(sprintf('%.1f dBm; seed %d',r.received_power_dbm,r.seed));
                end
                exportgraphics(gcf,fullfile(folder,'power_constellations.png'),'Resolution',180); close(gcf);
            end
            predicted=~m.is_calibration_power;
            [largest_difference,largest_index]=max(abs(m.log10_difference));
            lines=["# BER versus received optical power"; ""; ...
                "Paper Figure 13(c): unequalized 16-QAM at 400 mVpp, 0.4 Gbaud, 15 km fiber and 1 m millimeter-wave wireless link. No equalizer or NN is used."; ""; ...
                "Only the two endpoints (-22 and -17.5 dBm) identify effective driver gain and constant aggregate post-detector IF-noise PSD. All eight intermediate powers are excluded from the numerical fit. This is a separate calibration from Figures 9/11; the earlier voltage-curve files and parameters are preserved."; ""; ...
                sprintf('Effective gain %.8g; fixed noise PSD %.8g in effective detector-output units squared/Hz. Validation: %d seeds, %d payload symbols per seed, %d bits per power.', ...
                    frozen.configuration.driver_voltage_gain,frozen.configuration.if_noise_psd, ...
                    numel(frozen.settings.validation_seeds),frozen.settings.validation_payload_symbols,m.bits(1)); ...
                ""; "| ROP (dBm) | Approximate paper BER | Simulated BER | Errors / bits | Fitted power? |"; ...
                "|---:|---:|---:|---:|---|"];
            for k=1:height(m)
                r=m(k,:);
                lines(end+1)=sprintf('| %.1f | %.5g | %.5g | %d / %d | %d |',r.received_power_dbm, ...
                    r.paper_ber,r.simulation_ber,r.bit_errors,r.bits,r.is_calibration_power); %#ok<AGROW>
            end
            lines=[lines; ""; sprintf('%d/%d intermediate powers fall within the declared +/-0.08 log10(BER) read-off allowance. This is a descriptive comparison, not an experimental confidence test.', ...
                nnz(m.within_read_allowance & predicted),nnz(predicted)); ""; ...
                sprintf('Largest absolute log10(BER) difference: %.4f at %.1f dBm. The remaining discrepancy prevents a full quantitative curve-replication claim.', ...
                    largest_difference,m.received_power_dbm(largest_index)); ""; ...
                "![BER versus received power](ber_received_power.png)"; ""; ...
                "This fit and the earlier Figures 9/11 fit are alternative effective parameterizations of the same topology. Different fitted gains must not be presented as one jointly validated device model. The ROP fit is the current candidate for physical-baseline review."; ""; ...
                "ROP is assumed to mean total time-average power in both optical tones incident on the PD. Both fields scale by the square root of the requested power ratio while their relative powers and modulation remain fixed. Optical beating therefore makes RF current proportional to optical watts. Effective responsivity is set to 1 A/W; unresolved RF gains and noise are absorbed into a lumped receiver-noise parameter. This does not independently reconstruct detector responsivity, laser powers, ASE, shot noise or receiver noise figure."; ""; ...
                "The constant post-detector noise hypothesis predicts noise-to-signal power proportional to optical power to the minus second power. It is tested here, not established by the paper. No extra per-power gain, noise, distortion or curve correction is applied."; ""; ...
                "Paper values are approximate manual readings, with +/-0.08 log10(BER) allowance and approximately +/-0.08 dB horizontal reading uncertainty. They are not author-supplied data or experimental confidence intervals. The -21 dBm marker is partly obscured by the legend. Simulation BER bars are nominal Wilson intervals assuming independent bits; temporal dependence and finite seed count limit that interpretation."; ""; ...
                "Calibration uses seed 8601 and a conditional Gaussian decision-cell surrogate. Frozen settings are then checked using full waveform simulation with real IF noise and independent validation seeds 8701-8705 in the study profile. Within each seed, waveform and noise draws are paired across power values. Prefix-only receiver timing/gain recovery is repeated at each power. Held-out here means excluded from parameter fitting, not blind experimental validation."; ""; ...
                "Run from the repository root: `addpath('matlab'); c=arof.configuration(); arof.received_power.run(c);`. To rerender the saved BER plot without rerunning simulation, call `arof.received_power.plot('matlab/results/received_power',c)`. PNG and vector PDF plus raw per-seed CSV counts are supplied. Calibration provenance is in frozen_calibration.json and calibration_search.csv."];
            [fid,msg]=fopen(fullfile(folder,'README.md'),'w'); assert(fid>=0,'%s',msg);
            cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',join(lines,newline));
        end

        function folder = run(cfg,s)
            %RUN Independent Figure 13(c) calibration and frozen sweep.
            if nargin<2, s=arof.received_power.settings(cfg.profile); end
            folder=fullfile(cfg.output_root,'received_power',cfg.run_id);
            assert(~isfolder(folder),'Refusing to overwrite received-power run.'); mkdir(folder);
            copyfile(fullfile(cfg.root,'matlab','reference_data','received_power_reference.csv'), ...
                fullfile(folder,'paper_reference.csv'));
            arof.received_power.calibrate(cfg,s,folder);
            folder=arof.received_power.sweep(folder,cfg);
        end

    end
end

function ber = gaussian_bit_error(mu,standard_deviation,reference,modulation)
%GAUSSIAN_BIT_ERROR Conditional cell probabilities, not an EVM-to-BER conversion.
% Used only as a smooth calibration surrogate; validation runs physical noise.
ideal=qammod((0:modulation-1).',modulation,'gray');
labels=de2bi((0:modulation-1).',log2(modulation),'left-msb');
if standard_deviation<=0
    actual=qamdemod(mu,modulation,'gray','OutputType','bit');
    target=qamdemod(reference,modulation,'gray','OutputType','bit');
    ber=nnz(actual~=target)/numel(target); return;
end
levels=unique(real(ideal));
edges=[-Inf;(levels(1:end-1)+levels(2:end))/2;Inf];
truth=labels(qamdemod(reference,modulation,'gray')+1,:);
ber=0;
for k=1:modulation
    i=find(levels==real(ideal(k))); q=find(levels==imag(ideal(k)));
    prob_i=.5*erfc(-(edges(i+1)-real(mu))/(standard_deviation*sqrt(2)))- ...
        .5*erfc(-(edges(i)-real(mu))/(standard_deviation*sqrt(2)));
    prob_q=.5*erfc(-(edges(q+1)-imag(mu))/(standard_deviation*sqrt(2)))- ...
        .5*erfc(-(edges(q)-imag(mu))/(standard_deviation*sqrt(2)));
    ber=ber+mean(prob_i.*prob_q.*sum(truth~=labels(k,:),2)/log2(modulation));
end
end

function write_json(path,value)
[fid,msg]=fopen(path,'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',jsonencode(value,'PrettyPrint',true));
end

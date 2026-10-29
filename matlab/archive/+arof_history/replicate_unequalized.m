function run_dir = replicate_unequalized(cfg,settings)
%REPLICATE_UNEQUALIZED Calibrated attempt, not asserted experimental replication.
if nargin<2, settings=arof_history.paper_settings(); end
run_dir=fullfile(cfg.output_root,"paper_unequalized",cfg.run_id);
assert(~isfolder(run_dir),"Refusing to overwrite paper comparison."); mkdir(run_dir);
reference_path=fullfile(cfg.root,"matlab","reference_data","unequalized_points.csv");
reference=readtable(reference_path,'TextType','string');
copyfile(reference_path,fullfile(run_dir,"paper_points.csv"));
copyfile(fullfile(fileparts(reference_path),"README.md"),fullfile(run_dir,"paper_points_notes.md"));
[models,fit_table]=arof_history.calibrate_paper(cfg,settings);
writetable(fit_table,fullfile(run_dir,"driver_identification.csv"));
write_text(fullfile(run_dir,"settings.json"),jsonencode(settings,"PrettyPrint",true));
rows=struct([]); snapshots=struct([]);
for model=1:numel(models)
    base=models{model}; base.payload_symbols=settings.payload_symbols;
    for modulation=settings.modulations
        for seed=settings.seeds
            base.modulation_order=modulation; base.seed=seed;
            % Peak calibration is deterministic transmitter configuration.
            unit=arof_history.transmit(base);
            unit_scale=base.drive_rms_v/(max(unit.if_before_driver)-min(unit.if_before_driver));
            for input_mv=settings.input_vpp_mv
                c=base; c.drive_rms_v=unit_scale*input_mv/1000;
                c.noise_enabled=true; c.noise_seed=seed+900000;
                r=arof_history.simulate(c); m=r.metrics;
                item=struct("model",settings.receiver_models(model),"modulation",modulation, ...
                    "seed",seed,"input_vpp_mv",input_mv,"mzm_vpp_v",m.drive_vpp_v, ...
                    "evm_percent",m.evm_percent,"ber",m.ber,"bit_errors",m.bit_errors, ...
                    "bits",m.bits,"if_noise_psd",c.if_noise_psd, ...
                    "driver_gain",c.driver_voltage_gain,"driver_saturation_v",c.driver_saturation_v);
                rows=[rows; item]; %#ok<AGROW>
                if model==2 && seed==settings.seeds(1) && ...
                        ((modulation==16 && ismember(input_mv,[380 440 500])) || ...
                        (modulation==64 && ismember(input_mv,[140 200 260])))
                    ids=r.rx.payload_indices;
                    snapshots=[snapshots; struct("modulation",modulation,"input_vpp_mv",input_mv, ...
                        "prediction",r.rx.sampled(ids),"reference",r.rx.reference(ids))]; %#ok<AGROW>
                end
            end
            fprintf("Unequalized paper attempt: %s, %d-QAM, seed %d\n", ...
                settings.receiver_models(model),modulation,seed);
            writetable(struct2table(rows),fullfile(run_dir,"seed_metrics.csv"));
        end
    end
end
metrics=struct2table(rows);
summary=summarize(metrics);
writetable(summary,fullfile(run_dir,"summary.csv"));
comparisons=struct([]);
for k=1:height(reference)
    point=reference(k,:);
    for model=settings.receiver_models
        match=summary.model==model & summary.modulation==point.modulation & ...
            summary.input_vpp_mv==point.input_vpp_mv;
        if ~any(match), continue; end
        prediction=summary(match,:);
        if point.metric=="ber", value=prediction.ber; else, value=prediction.evm_percent; end
        comparisons=[comparisons; struct("model",model,"figure",point.figure, ...
            "modulation",point.modulation,"input_vpp_mv",point.input_vpp_mv, ...
            "metric",point.metric,"paper_value",point.value,"simulation_value",value, ...
            "absolute_difference",abs(value-point.value),"paper_role",point.role)]; %#ok<AGROW>
    end
end
comparison_table=struct2table(comparisons);
writetable(comparison_table,fullfile(run_dir,"paper_comparison.csv"));
quality=struct([]);
for model=settings.receiver_models
    for modulation=settings.modulations
        for metric=["evm_percent" "ber"]
            selected=comparison_table.model==model & comparison_table.modulation==modulation & ...
                comparison_table.metric==metric & comparison_table.paper_role~="calibration";
            q=comparison_table(selected,:);
            if isempty(q), continue; end
            if metric=="ber"
                positive=q.simulation_value>0;
                difference=log10(q.simulation_value(positive))-log10(q.paper_value(positive));
                units="log10(BER), positive predictions only";
            else
                positive=true(height(q),1);
                difference=q.simulation_value-q.paper_value; units="EVM percentage points";
            end
            quality=[quality; struct("model",model,"modulation",modulation,"metric",metric, ...
                "comparison_points",height(q),"scored_points",nnz(positive), ...
                "rmse",sqrt(mean(difference.^2)),"units",units)]; %#ok<AGROW>
        end
    end
end
writetable(struct2table(quality),fullfile(run_dir,"comparison_errors.csv"));
for k=1:numel(models)
    write_text(fullfile(run_dir,"model_"+k+".json"),jsonencode(models{k},"PrettyPrint",true));
end
save(fullfile(run_dir,"constellations.mat"),"snapshots","-v7");
arof_history.plot_paper_attempt(cfg,settings,summary,reference,snapshots,run_dir);
write_text(fullfile(run_dir,"README.md"),join([ ...
    "# Unequalized paper reproduction attempt"; ""; ...
    "This is a calibrated, conditional reproduction attempt, not a verified quantitative replication."; ""; ...
    "The link follows the paper's IF/optical-beating arrangement. Models compare a linear driver and ideal amplitude detector, a compressed bandpass-envelope driver and amplitude detector, and the same compressed driver with square-law RF detection."; ""; ...
    "Calibration uses only the approximate Figure 9 4-QAM EVM values at 100 mV (10.5%) and 500 mV (8.1%), plus the approximately 200 mV to 5 Vpp voltage anchor. Rapp smoothness is assumed to be 2. Driver saturation is selected on a declared grid; effective real IF noise PSD is then fixed across all drive and modulation settings for each detector hypothesis."; ""; ...
    "Other paper points are excluded from numerical fitting. They have been inspected during model development, so they are comparison points, not blind independent validation. Paper values are approximate visual readings, not raw experimental data; see paper_points_notes.md."; ""; ...
    "Fitted driver gain/saturation and aggregate noise are inferred surrogate parameters, not measured device specifications. The paper does not supply the full driver transfer curve, RF detector response, receiver-noise budget or enough received-power information to establish identical noise conditions across Figures 9, 11 and 12. The cross-figure BER comparison assumes a shared noise level."; ""; ...
    "EVM uses continuous I/Q and RMS reference-symbol energy with prefix-only timing/gain calibration. The paper's complete normalization procedure is unavailable. Noise is fixed post-detector real white noise in normalized units, not a calibrated optical ASE/shot/thermal model. It is scaled with sample rate. No equalizer or NN is used."; ""; ...
    "See REVIEW.md for numeric discrepancies. summary.csv retains raw bit counts, pooled BER, nominal Wilson intervals and EVM seed spread. Zero-error BER points are displayed as upper-bound markers. Within each seed, drive points share transmitted symbols and noise draws; they are paired points, not independent experiments."; ""; ...
    "![EVM comparison](evm_comparison.png)"; ""; ...
    "![Unequalized BER comparison](ber_comparison.png)"; ""; ...
    "![Driver model assumptions](driver_mapping.png)"; ""; ...
    "Inspect paper_comparison.csv and comparison_errors.csv for discrepancies. A trend match or a fitted calibration point does not validate the physical model, establish NN necessity, or justify proceeding to equalizer claims."],newline));
arof_history.review_paper_attempt(run_dir);
fprintf("Paper reproduction attempt: %s\n",run_dir);
end

function out = summarize(metrics)
rows=struct([]);
for model=unique(metrics.model,'stable').'
    for modulation=unique(metrics.modulation).'
        for voltage=unique(metrics.input_vpp_mv).'
            selected=metrics.model==model & metrics.modulation==modulation & metrics.input_vpp_mv==voltage;
            m=metrics(selected,:); n=sum(m.bits); errors=sum(m.bit_errors); p=errors/n; z=1.96;
            center=(p+z*z/(2*n))/(1+z*z/n);
            radius=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n);
            rows=[rows; struct("model",model,"modulation",modulation,"input_vpp_mv",voltage, ...
                "evm_percent",mean(m.evm_percent),"evm_seed_std",std(m.evm_percent), ...
                "ber",p,"bit_errors",errors,"bits",n,"ber_low",max(0,center-radius), ...
                "ber_high",center+radius,"mzm_vpp_v",mean(m.mzm_vpp_v))]; %#ok<AGROW>
        end
    end
end
out=struct2table(rows);
end

function write_text(path,value)
[fid,msg]=fopen(path,"w"); assert(fid>=0,"%s",msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,"%s\n",value);
end

function review_paper_attempt(folder)
%REVIEW_PAPER_ATTEMPT Descriptive discrepancies, not a validation certificate.
metrics=readtable(fullfile(folder,"summary.csv"),'TextType','string');
comparison=readtable(fullfile(folder,"paper_comparison.csv"),'TextType','string');
quality=readtable(fullfile(folder,"comparison_errors.csv"),'TextType','string');
lines=["# Unequalized comparison review"; ""; ...
    "Calibration does not establish replication. The following errors exclude the two fitted 4-QAM EVM points. Numerical parameter fitting did not use the other points, but the published curves were inspected during model development."; ""; ...
    "| Model | QAM | Metric | RMSE | Units |"; "|---|---:|---|---:|---|"];
for k=1:height(quality)
    q=quality(k,:);
    lines(end+1)=sprintf('| %s | %d | %s | %.4g | %s |', ...
        q.model,q.modulation,q.metric,q.rmse,q.units); %#ok<AGROW>
end
lines=[lines; ""; "BER log-RMSE omits zero predictions; their raw counts and upper bounds remain in summary.csv. Wilson ranges are nominal and assume independent bits; filtering and nonlinear memory can violate that assumption. Three seeds provide limited variability information. The paper read-off allowances are not experimental confidence intervals."; ...
    ""; "## Compressed-driver amplitude-detector comparison"; ""; ...
    "| QAM | Input mVpp | Metric | Approximate paper value | Simulation |"; ...
    "|---:|---:|---|---:|---:|"];
selected=comparison.model=="compressed_amplitude" & comparison.paper_role~="calibration";
for k=find(selected).'
    q=comparison(k,:);
    lines(end+1)=sprintf('| %d | %d | %s | %.5g | %.5g |', ...
        q.modulation,q.input_vpp_mv,q.metric,q.paper_value,q.simulation_value); %#ok<AGROW>
end
lines=[lines; ""; "## Identification limits"; ""];
for k=1:3
    c=jsondecode(fileread(fullfile(folder,"model_"+k+".json")));
    lines(end+1)=sprintf('Model %d: driver %s, gain %.6g, detector %s, effective IF noise PSD %.6g normalized units squared/Hz.', ...
        k,c.driver_model,c.driver_voltage_gain,c.envelope_detector_model,c.if_noise_psd); %#ok<AGROW>
    if c.driver_model~="linear"
        lines(end+1)=sprintf('Inferred saturation is %.6g V peak; smoothness is %.6g. These are fitted assumptions, not component measurements.', ...
            c.driver_saturation_v,c.driver_smoothness); %#ok<AGROW>
    end
end
lines=[lines; ""; ...
    "The model_N.json files describe calibration templates. Actual sweep runs enable noise, choose the row's modulation/seed/voltage and use noise_seed = seed + 900000. Settings and seed_metrics.csv define the runs."; ...
    ""; "The full driver transfer, RF detector law, physical noise budget, EVM normalization and common received-power conditions across figures are not established. Consequently these plots are conditional model-identification attempts, not independently verified reproductions. Do not proceed to NN claims on the basis of the fitted points."; ...
    ""; sprintf('The report contains %d pooled simulation points.',height(metrics))];
[fid,msg]=fopen(fullfile(folder,"REVIEW.md"),'w'); assert(fid>=0,'%s',msg);
cleanup=onCleanup(@() fclose(fid)); fprintf(fid,'%s\n',join(lines,newline));
end

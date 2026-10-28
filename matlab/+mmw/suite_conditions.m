function conditions = suite_conditions(cfg)
%SUITE_CONDITIONS Predeclare the operating points before fitting receivers.
template = struct("Condition", "", "Group", "", "Axis", 0, "XLabel", "", ...
    "ModOrder", 16, "Memory", false, "Drive", 1, "SNR", 30, ...
    "Enabled", true, "ActivationStudy", false);
rows = repmat(template, 0, 1);
if cfg.scenario == 2
    for order = [16 64]
        for memory = [false true]
            row = template; row.ModOrder = order; row.Memory = memory;
            row.Group = sprintf("%d-QAM, memory=%d", order, memory);
            row.XLabel = "MZM envelope drive index (rad RMS reference)";
            for drive = cfg.mzm_drive
                row.Axis = drive; row.Drive = drive; row.SNR = cfg.mzm_snr_db;
                row.Condition = sprintf("qam%d_mem%d_drive%03d", order, memory, round(100*drive));
                row.ActivationStudy = drive == 1 && ~memory;
                rows(end+1) = row; %#ok<AGROW>
            end
        end
    end
    row = template; row.Enabled = false; row.SNR = Inf;
    row.Condition = "linear_noiseless"; row.Group = "Control";
    row.XLabel = "Control"; rows(end+1) = row;
    for snr = [12 18]
        row = template; row.ModOrder = 2; row.Memory = true;
        row.Group = "BPSK adaptive activation"; row.XLabel = "Reference SNR (dB)";
        row.Axis = snr; row.SNR = snr; row.Drive = 1;
        row.Condition = sprintf("bpsk_snr%d", snr); row.ActivationStudy = true;
        rows(end+1) = row; %#ok<AGROW>
    end
else
    error("Unimplemented scenario.");
end
conditions = struct2table(rows);
end

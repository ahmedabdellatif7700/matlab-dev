function conditions = suite_conditions(cfg)
%SUITE_CONDITIONS Predeclare the operating points before fitting receivers.
template = struct("Condition", "", "Group", "", "Axis", 0, "XLabel", "", ...
    "ModOrder", 16, "Memory", false, "Drive", 1, "SNR", 30, ...
    "Enabled", true, "ActivationStudy", false, "Sideband", "DSB", ...
    "Fiber_km", 0, "Backoff", 3, "Guard", 4);
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
elseif cfg.scenario == 3
    for sideband = ["DSB" "SSB"]
        for pa = [false true]
            row = template; row.Sideband = sideband; row.Enabled = pa;
            row.Group = sprintf("%s, RF PA=%d", sideband, pa);
            row.XLabel = "Fiber length (km)"; row.SNR = 30;
            row.Drive = 10^(-row.Backoff/20);
            for length_km = cfg.fiber_lengths_km
                row.Fiber_km = length_km; row.Axis = length_km;
                row.Condition = sprintf("%s_pa%d_L%03d", sideband, pa, round(100*length_km));
                rows(end+1) = row; %#ok<AGROW>
            end
        end
    end
elseif cfg.scenario == 4
    for guard = cfg.multi_guards
        row = template; row.Guard = guard; row.SNR = 28;
        row.Group = sprintf("SC-FDMA, guard=%d bins", guard);
        row.XLabel = "Common MZM envelope drive index";
        for drive = cfg.multi_drive
            row.Drive = drive; row.Axis = drive;
            row.Condition = sprintf("guard%d_drive%03d", guard, round(100*drive));
            rows(end+1) = row; %#ok<AGROW>
        end
    end
    row = template; row.Enabled = false; row.SNR = Inf; row.Group = "Control";
    row.XLabel = "Linear noiseless control"; row.Condition = "linear_noiseless";
    rows(end+1) = row;
else
    error("Unimplemented scenario.");
end
conditions = struct2table(rows);
end

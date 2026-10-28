function cases = design_candidates()
%DESIGN_CANDIDATES Predeclare physically distinct families before tuning.
row = struct("ID", "", "Family", "", "ModOrder", 16, "PA", false, ...
    "Backoff_dB", 0, "AMPM_deg", 20, "MZM", false, "MZM_drive", 1, ...
    "Optical", false, "Sideband", "DSB", "Fiber_km", 0, ...
    "Echo_real", 0, "Echo_imag", 0, "Reference_SNR_dB", 30);
rows = repmat(row, 0, 1);
for drive = [1.35 1.5 1.65]
    c = row; c.ID = sprintf("mzm16_mu%03d",round(100*drive));
    c.Family = "MZM16"; c.MZM = true; c.MZM_drive = drive;
    rows(end+1) = c; %#ok<AGROW>
end
for drive = [0.9 1.05 1.2]
    c = row; c.ID = sprintf("mzm64_mu%03d",round(100*drive));
    c.Family = "MZM64_memory"; c.MZM = true; c.MZM_drive = drive;
    c.ModOrder = 64; c.Echo_real = 0.2*cos(pi/5); c.Echo_imag = 0.2*sin(pi/5);
    c.Reference_SNR_dB = 34;
    rows(end+1) = c; %#ok<AGROW>
end
for backoff = [-3 0 3]
    c = row; c.ID = sprintf("pa16_ibo%+03d",backoff);
    c.Family = "RFPA16_memory"; c.PA = true; c.Backoff_dB = backoff; c.AMPM_deg = 40;
    c.Echo_real = 0.25*cos(pi/4); c.Echo_imag = 0.25*sin(pi/4); c.Reference_SNR_dB = 32;
    rows(end+1) = c; %#ok<AGROW>
end
for length_km = [0.3 0.5 0.7]
    c = row; c.ID = sprintf("pa64_DSB_L%03d",round(100*length_km));
    c.Family = "RFPA64_DSB"; c.ModOrder = 64; c.PA = true;
    c.Optical = true; c.Fiber_km = length_km; c.Reference_SNR_dB = 36;
    rows(end+1) = c; %#ok<AGROW>
end
% High-SNR controls distinguish approximation error from a noise floor.
for drive = [1.05 1.2]
    c = row; c.ID = sprintf("mzm64_mu%03d_snr40",round(100*drive));
    c.Family = "MZM64_memory"; c.MZM = true; c.MZM_drive = drive;
    c.ModOrder = 64; c.Echo_real = 0.2*cos(pi/5); c.Echo_imag = 0.2*sin(pi/5);
    c.Reference_SNR_dB = 40; rows(end+1) = c; %#ok<AGROW>
end
for length_km = [0.3 0.5]
    c = row; c.ID = sprintf("pa64_DSB_L%03d_snr44",round(100*length_km));
    c.Family = "RFPA64_DSB"; c.ModOrder = 64; c.PA = true; c.AMPM_deg = 40;
    c.Optical = true; c.Fiber_km = length_km; c.Reference_SNR_dB = 44;
    rows(end+1) = c; %#ok<AGROW>
end
cases = struct2table(rows);
end

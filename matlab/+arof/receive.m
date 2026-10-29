function rx = receive(cfg,detected,tx)
%RECEIVE Friis hop, envelope detection, IF downconversion and matched filter.
assert(~cfg.noise_enabled,"Noise awaits justified component parameters; no fabricated SNR is added.");
lambda = 299792458/cfg.rf_carrier_hz;
path_loss_db = 20*log10(4*pi*cfg.wireless_distance_m/lambda);
path_gain = 10^((cfg.tx_antenna_gain_dbi+cfg.rx_antenna_gain_dbi-path_loss_db)/20);
rf = detected.rf_envelope*path_gain;
envelope = abs(rf);
envelope_ac = envelope-mean(envelope); % Receiver DC rejection, no target use.
t = (0:numel(envelope)-1).'/cfg.internal_sample_rate_hz;
baseband = 2*envelope_ac.*exp(-1j*2*pi*cfg.if_hz*t);
h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,tx.internal_sps,"sqrt").';
matched = conv(baseband,h);
% Sum of Tx and Rx RRC delays, expressed on the internal grid.
origin = cfg.rrc_span_symbols*tx.internal_sps+1;
keep = cfg.guard_symbols+(1:cfg.training_symbols+cfg.payload_symbols).';
centers = origin+(keep-1)*tx.internal_sps;
reference = tx.symbols(keep);
calibration = 1:cfg.training_symbols;
best_error = Inf; best_offset = 0; best_gain = 1;
for offset = linspace(-0.05*tx.internal_sps,0.05*tx.internal_sps,401)
    samples = interp1(matched,centers(calibration)+offset,"spline");
    gain = (samples'*reference(calibration))/(samples'*samples);
    error = mean(abs(gain*samples-reference(calibration)).^2);
    if error < best_error
        best_error = error; best_offset = offset; best_gain = gain;
    end
end
sampled = interp1(matched,centers+best_offset,"spline")*best_gain;
payload = cfg.training_symbols+(1:cfg.payload_symbols);
rx = struct("rf_envelope",rf,"envelope",envelope,"baseband",baseband, ...
    "sampled",sampled,"reference",reference,"payload_indices",payload, ...
    "timing_offset_samples",best_offset,"calibration_gain",best_gain, ...
    "wireless_path_loss_db",path_loss_db);
end

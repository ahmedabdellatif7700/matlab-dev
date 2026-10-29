function detector = downconvert(cfg,link)
%DOWNCONVERT RF envelope detection, receiver DC rejection and IF mixing.
assert(~cfg.noise_enabled,"Noise parameters have not been calibrated.");
assert(cfg.envelope_detector_model == "ideal_amplitude", ...
    "Only the declared ideal amplitude-envelope detector is implemented.");
rf = link.rf_envelope;
envelope = abs(rf);
envelope_ac = envelope-mean(envelope); % Receiver DC rejection, no target use.
t = (0:numel(envelope)-1).'/cfg.internal_sample_rate_hz;
baseband = 2*envelope_ac.*exp(-1j*2*pi*cfg.if_hz*t);
detector = struct("rf_envelope",rf,"envelope",envelope,"baseband",baseband, ...
    "wireless_path_loss_db",link.path_loss_db);
end

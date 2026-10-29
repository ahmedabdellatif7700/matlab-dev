function detector = downconvert(cfg,link)
%DOWNCONVERT RF envelope detection, receiver DC rejection and IF mixing.
rf = link.rf_envelope;
switch cfg.envelope_detector_model
    case "ideal_amplitude"
        envelope = abs(rf);
    case "square_law"
        envelope = abs(rf).^2;
    otherwise
        error("Unknown RF detector law.");
end
% A lumped post-detector real white-noise model. PSD scales with sample rate;
% this is NOT an independently calibrated thermal/shot/ASE noise budget.
if cfg.noise_enabled
    assert(isfinite(cfg.if_noise_psd) && cfg.if_noise_psd >= 0,"Invalid IF noise PSD.");
    noise_stream = RandStream("mt19937ar","Seed",cfg.noise_seed);
    envelope = envelope+sqrt(cfg.if_noise_psd*cfg.internal_sample_rate_hz)* ...
        randn(noise_stream,size(envelope));
end
envelope_ac = envelope-mean(envelope); % Receiver DC rejection, no target use.
t = (0:numel(envelope)-1).'/cfg.internal_sample_rate_hz;
baseband = 2*envelope_ac.*exp(-1j*2*pi*cfg.if_hz*t);
detector = struct("rf_envelope",rf,"envelope",envelope,"baseband",baseband, ...
    "wireless_path_loss_db",link.path_loss_db);
end

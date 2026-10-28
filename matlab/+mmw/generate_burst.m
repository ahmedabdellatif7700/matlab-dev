function [rx, tx, bits] = generate_burst(cfg, count, seed, pa_enabled, backoff_db, snr_db)
%GENERATE_BURST Independent random QAM burst through a complex-envelope PA.
% No optical link, dispersion, RF multipath, or synchronization errors.
% Rapp AM/AM and bounded AM/PM are illustrative, not device fits.
validateattributes(count, {'numeric'}, {'scalar', 'integer', 'positive'});
stream = RandStream("mt19937ar", "Seed", seed);
k = log2(cfg.mod_order);
assert(k == fix(k), "Modulation order must be a power of two.");
edge = cfg.rrc_span_symbols + cfg.equalizer_taps;
bits_all = randi(stream, [0 1], (count + 2*edge)*k, 1);
symbols = qammod(bits_all, cfg.mod_order, "gray", "InputType", "bit");
pulse = rcosdesign(cfg.rrc_rolloff, cfg.rrc_span_symbols, ...
    cfg.samples_per_symbol, "sqrt").';
up = zeros(numel(symbols)*cfg.samples_per_symbol, 1);
up(1:cfg.samples_per_symbol:end) = symbols;
wave = conv(up, pulse, "full");
ideal = qammod((0:cfg.mod_order-1).', cfg.mod_order, "gray");
symbol_energy = mean(abs(ideal).^2);
sat_amplitude = sqrt(symbol_energy / cfg.samples_per_symbol);
drive_gain = 10^(-backoff_db/20);
wave = drive_gain * wave;
if pa_enabled
    u = abs(wave)/sat_amplitude;
    gain = (1 + u.^(2*cfg.pa_smoothness)).^(-1/(2*cfg.pa_smoothness));
    phase = deg2rad(cfg.pa_phase_limit_deg)*u.^2./(1 + u.^2);
    wave = wave .* gain .* exp(1j*phase);
end
% Unit-energy matched filtering preserves the white-noise variance at symbol
% instants. Never normalize noise to distorted received power.
if isfinite(snr_db)
    noise_variance = symbol_energy / 10^(snr_db/10);
    wave = wave + sqrt(noise_variance/2) * ...
        (randn(stream, size(wave)) + 1j*randn(stream, size(wave)));
end
matched = conv(wave, pulse, "full");
indices = cfg.rrc_span_symbols*cfg.samples_per_symbol + 1 + ...
    (0:numel(symbols)-1).'*cfg.samples_per_symbol;
% Undo only the known transmit drive scaling, common to all receivers.
% This exposes the noise penalty of reduced drive, with no test-data fit.
received = matched(indices)/drive_gain;
keep = edge + (1:count);
rx = received(keep);
tx = symbols(keep);
bit_matrix = reshape(bits_all, k, []).';
bits = reshape(bit_matrix(keep, :).', [], 1);
end

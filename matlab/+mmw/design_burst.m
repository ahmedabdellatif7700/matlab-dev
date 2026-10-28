function [rx, tx] = design_burst(cfg, condition, count, seed)
%DESIGN_BURST Explicit physical stages shared by discovery and confirmation.
assert(~(condition.PA && condition.MZM), "These cases isolate modulator and PA nonlinearities.");
assert(~(condition.MZM && condition.Optical), "Large-signal MZM is not cascaded with small-signal CD.");
stream = RandStream("mt19937ar", "Seed", seed);
edge = cfg.rrc_span_symbols+max(abs(cfg.tap_offsets))+8;
indices = randi(stream, [0 condition.ModOrder-1], count+2*edge, 1);
tx_all = qammod(indices, condition.ModOrder, "gray");

% Stage 1: unit-energy RRC pulse shaping.
h = rcosdesign(cfg.rrc_rolloff, cfg.rrc_span_symbols, cfg.samples_per_symbol, "sqrt").';
up = zeros(numel(tx_all)*cfg.samples_per_symbol, 1); up(1:cfg.samples_per_symbol:end) = tx_all;
wave = conv(up, h);
energy = mean(abs(qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray")).^2);
reference_power = energy/cfg.samples_per_symbol;

% Stage 2: nonlinear RF driver or quadrature MZM fundamental.
known_gain = 1;
if condition.PA
    known_gain = 10^(-condition.Backoff_dB/20);
    wave = mmw.pa_envelope(known_gain*wave, reference_power, cfg.pa_smoothness, condition.AMPM_deg);
elseif condition.MZM
    known_gain = condition.MZM_drive;
    wave = mmw.mzm_envelope(wave, known_gain, reference_power);
end

% Stage 3: small-signal optical transport and an optional RF echo.
if condition.Optical
    wave = mmw.optical_fading(wave, cfg, condition.Sideband, condition.Fiber_km, cfg.fft_guard_samples);
end
echo = complex(condition.Echo_real, condition.Echo_imag);
if echo ~= 0
    wave = filter([1; zeros(cfg.samples_per_symbol-1, 1); echo], 1, wave);
end

% Stage 4: fixed receiver noise, never normalized to distorted power.
if isfinite(condition.Reference_SNR_dB)
    variance = energy/10^(condition.Reference_SNR_dB/10);
    wave = wave+sqrt(variance/2)*(randn(stream, size(wave))+1j*randn(stream, size(wave)));
end

% Stage 5: matched filtering, known delay/gain removal and edge trimming.
matched = conv(wave, h);
samples = cfg.rrc_span_symbols*cfg.samples_per_symbol+1+(0:numel(tx_all)-1).'*cfg.samples_per_symbol;
keep = edge+(1:count);
rx = matched(samples(keep))/known_gain;
tx = tx_all(keep);
end

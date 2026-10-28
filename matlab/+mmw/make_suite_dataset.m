function data = make_suite_dataset(cfg, seed, condition)
%MAKE_SUITE_DATASET Independent train/validation/test waveform realizations.
if cfg.scenario == 4
    data = mmw.make_multiuser_dataset(cfg, seed, condition);
    return;
elseif cfg.scenario == 5
    data = mmw.make_ofdm_dataset(cfg, seed, condition);
    return;
end
data = struct(); names = ["train" "validation" "test"];
counts = [cfg.train_symbols cfg.validation_symbols cfg.test_symbols];
for part = 1:3
    [rx, tx, diagnostic] = single_carrier(cfg, condition, counts(part), seed+100000*part);
    if isfield(cfg, "tap_offsets"), offsets = cfg.tap_offsets;
    else, offsets = 0:-1:1-cfg.equalizer_taps; end
    indices = (1-min(offsets):counts(part)-max(offsets)).';
    x = zeros(numel(indices), 2*cfg.equalizer_taps, "single");
    for tap = 1:cfg.equalizer_taps
        samples = rx(indices+offsets(tap));
        x(:, 2*tap-1) = single(real(samples)); x(:, 2*tap) = single(imag(samples));
    end
    data.("X_"+names(part)) = x;
    data.("Y_"+names(part)) = single([real(tx(indices)) imag(tx(indices))]);
    if part == 3
        data.baseline_test = rx(indices);
        data.diagnostic_json = jsonencode(diagnostic);
    end
end
data.seed = seed;
data.ideal_symbols = qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray");
data.ideal_bits = de2bi((0:condition.ModOrder-1).', log2(condition.ModOrder), "left-msb");
end

function [rx, tx, diagnostic] = single_carrier(cfg, condition, count, seed)
stream = RandStream("mt19937ar", "Seed", seed);
edge = cfg.rrc_span_symbols + 8;
indices = randi(stream, [0 condition.ModOrder-1], count+2*edge, 1);
symbols = qammod(indices, condition.ModOrder, "gray");
h = rcosdesign(cfg.rrc_rolloff, cfg.rrc_span_symbols, cfg.samples_per_symbol, "sqrt").';
up = zeros(numel(symbols)*cfg.samples_per_symbol, 1);
up(1:cfg.samples_per_symbol:end) = symbols;
wave = conv(up, h, "full");
ideal = qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray");
energy = mean(abs(ideal).^2); reference_power = energy/cfg.samples_per_symbol;
if cfg.scenario == 3
    wave = condition.Drive*wave;
    if condition.Enabled
        wave = mmw.pa_envelope(wave, reference_power, cfg.pa_smoothness, cfg.pa_phase_limit_deg);
    end
    wave = mmw.optical_fading(wave, cfg, condition.Sideband, condition.Fiber_km, cfg.fft_guard_samples);
elseif condition.Enabled
    wave = mmw.mzm_envelope(wave, condition.Drive, reference_power);
else
    wave = condition.Drive*wave;
end
if condition.Memory
    coefficient = complex(cfg.memory_coefficient(1), cfg.memory_coefficient(2));
    memory_filter = [1; zeros(cfg.samples_per_symbol-1, 1); coefficient];
    wave = filter(memory_filter, 1, wave);
end
if isfinite(condition.SNR)
    wave = wave + sqrt(energy/10^(condition.SNR/10)/2) * ...
        (randn(stream, size(wave)) + 1j*randn(stream, size(wave)));
end
matched = conv(wave, h, "full");
sample = cfg.rrc_span_symbols*cfg.samples_per_symbol + 1 + ...
    (0:numel(symbols)-1).'*cfg.samples_per_symbol;
keep = edge+(1:count);
rx = matched(sample(keep))/condition.Drive;
tx = symbols(keep);
diagnostic = struct("reference_snr_db", condition.SNR, "drive", condition.Drive, ...
    "effective_linear_snr_db", condition.SNR+20*log10(condition.Drive), ...
    "memory", condition.Memory);
end

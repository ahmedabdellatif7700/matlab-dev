function data = make_ofdm_dataset(cfg, seed, condition)
%MAKE_OFDM_DATASET Wideband OFDM with payload-independent known pilot blocks.
names = ["train" "validation" "test"];
counts = [cfg.train_symbols cfg.validation_symbols cfg.test_symbols];
nfft = cfg.ofdm_fft*cfg.ofdm_oversampling;
cp = cfg.ofdm_cp*cfg.ofdm_oversampling;
signed = [-cfg.ofdm_active/2:-1 1:cfg.ofdm_active/2].';
bins = mod(signed, nfft)+1;
energy = mean(abs(qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray")).^2);
reference_power = energy*cfg.ofdm_active/nfft;
if condition.Memory
    delay = 2*cfg.ofdm_oversampling;
    channel = [1; zeros(delay-1, 1); -0.35*exp(0.4j); zeros(delay-1, 1); 0.15*exp(-0.6j)];
else
    channel = 1;
end
assert(numel(channel)-1 <= cp, "Cyclic prefix shorter than configured linear channel.");
frequency = signed*cfg.ofdm_spacing_hz;
kappa = pi*cfg.dispersion_ps_nm_km*1e-6*cfg.optical_wavelength_m^2 * ...
    condition.Fiber_km*1000/299792458;
linear = fft(channel(:), nfft); linear = linear(bins).*exp(1j*kappa*frequency.^2);
data = struct();
for part = 1:3
    base_seed = seed+100000*part;
    payload_stream = RandStream("mt19937ar", "Seed", base_seed+cfg.ofdm_payload_seed_offset);
    pilot_stream = RandStream("mt19937ar", "Seed", base_seed+500000);
    noise_stream = RandStream("mt19937ar", "Seed", base_seed+1000000);
    frames = floor(counts(part)/cfg.ofdm_active);
    symbols = qammod(randi(payload_stream, [0 condition.ModOrder-1], cfg.ofdm_active, frames), condition.ModOrder, "gray");
    pilots = qammod(randi(pilot_stream, [0 condition.ModOrder-1], cfg.ofdm_active, cfg.ofdm_pilots), condition.ModOrder, "gray");
    guard = qammod(randi(pilot_stream, [0 condition.ModOrder-1], cfg.ofdm_active, 4), condition.ModOrder, "gray");
    transmitted = [guard(:, 1:2) pilots symbols guard(:, 3:4)];
    grid = zeros(nfft, size(transmitted, 2)); grid(bins, :) = transmitted;
    payload_wave = ifft(grid, [], 1)*sqrt(nfft);
    pilot_indices = 2+(1:cfg.ofdm_pilots);
    payload_indices = 2+cfg.ofdm_pilots+(1:frames);
    waveform = [payload_wave(end-cp+1:end, :); payload_wave];
    wave = condition.Drive*waveform(:);
    if condition.Enabled
        wave = mmw.pa_envelope(wave, reference_power, cfg.pa_smoothness, cfg.pa_phase_limit_deg);
    end
    wave = filter(channel, 1, wave);
    link = cfg; link.symbol_rate_hz = cfg.ofdm_sample_rate_hz; link.samples_per_symbol = 1;
    wave = mmw.optical_fading(wave, link, "SSB", condition.Fiber_km, cfg.fft_guard_samples);
    if isfinite(condition.SNR)
        wave = wave + sqrt(energy/10^(condition.SNR/10)/2) * ...
            (randn(noise_stream, size(wave))+1j*randn(noise_stream, size(wave)));
    end
    wave = reshape(wave/condition.Drive, nfft+cp, []);
    received = fft(wave(cp+1:end, :), [], 1)/sqrt(nfft);
    pilot_received = received(bins, pilot_indices);
    estimate = sum(conj(pilots).*pilot_received, 2)./sum(abs(pilots).^2, 2);
    gamma = (linear'*estimate)/(linear'*linear);
    if condition.EstimateMode == "known"
        % Diagnostic upper-information front end: known linear channel SHAPE,
        % with the common gain estimated using the same noisy pilots as LS.
        used = gamma*linear;
    else
        used = estimate;
    end
    used(abs(used) < 1e-6) = 1e-6;
    equalized = received(bins, payload_indices)./used;
    offsets = cfg.tap_offsets;
    indices = (1-min(offsets):cfg.ofdm_active-max(offsets)).';
    x = zeros(numel(indices)*frames, 2*numel(offsets), "single");
    for tap = 1:numel(offsets)
        value = reshape(equalized(indices+offsets(tap), :), [], 1);
        x(:, 2*tap-1) = single(real(value)); x(:, 2*tap) = single(imag(value));
    end
    target = reshape(symbols(indices, :), [], 1);
    data.("X_"+names(part)) = x;
    data.("Y_"+names(part)) = single([real(target) imag(target)]);
    data.("frontend_channel_"+names(part)) = used;
    if part == 3
        data.baseline_test = reshape(equalized(indices, :), [], 1);
        data.papr_ofdm_db = 10*log10(max(abs(payload_wave(:, payload_indices)).^2, [], 1)./ ...
            mean(abs(payload_wave(:, payload_indices)).^2, 1));
        data.papr_single_carrier_db = reference_papr(cfg, condition.ModOrder, frames, base_seed+2000000);
        shape_nmse = sum(abs(estimate-gamma*linear).^2)/sum(abs(gamma*linear).^2);
        raw_nmse = sum(abs(estimate-linear).^2)/sum(abs(linear).^2);
        data.diagnostic_json = jsonencode(struct("pilot_shape_nmse", shape_nmse, ...
            "pilot_raw_nmse", raw_nmse, "pilot_blocks", cfg.ofdm_pilots, ...
            "payload_blocks", frames, "effective_linear_esn0_db", condition.SNR+20*log10(condition.Drive), ...
            "estimator", condition.EstimateMode));
        data.linear_channel = linear;
    end
end
data.seed = seed;
data.ideal_symbols = qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray");
data.ideal_bits = de2bi((0:condition.ModOrder-1).', log2(condition.ModOrder), "left-msb");
end

function papr = reference_papr(cfg, order, frames, seed)
% Compare peak/mean over equal-duration (~52 ns) payload intervals. This is
% a transmitter PAPR reference, not a mismatched-channel BER comparison.
stream = RandStream("mt19937ar", "Seed", seed);
samples = round(cfg.symbol_rate_hz/cfg.ofdm_spacing_hz*cfg.samples_per_symbol);
edge = cfg.rrc_span_symbols*cfg.samples_per_symbol;
count = ceil((frames*samples+2*edge)/cfg.samples_per_symbol);
symbols = qammod(randi(stream, [0 order-1], count, 1), order, "gray");
up = zeros(count*cfg.samples_per_symbol, 1); up(1:cfg.samples_per_symbol:end) = symbols;
h = rcosdesign(cfg.rrc_rolloff, cfg.rrc_span_symbols, cfg.samples_per_symbol, "sqrt").';
wave = conv(up, h); wave = reshape(wave(edge+(1:frames*samples)), samples, frames);
papr = 10*log10(max(abs(wave).^2, [], 1)./mean(abs(wave).^2, 1));
end

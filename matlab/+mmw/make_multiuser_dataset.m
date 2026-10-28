function data = make_multiuser_dataset(cfg, seed, condition)
%MAKE_MULTIUSER_DATASET Localized SC-FDMA through individual PA/common MZM.
names = ["train" "validation" "test"];
counts = [cfg.train_symbols cfg.validation_symbols cfg.test_symbols];
[allocation, ~] = mmw.multiuser_bins(cfg, condition.Guard);
data = struct(); channel = [];
for part = 1:3
    stream = RandStream("mt19937ar", "Seed", seed+100000*part);
    frames = floor(counts(part)/cfg.multi_dft);
    symbols = qammod(randi(stream, [0 condition.ModOrder-1], ...
        cfg.multi_dft, frames, cfg.multi_users), condition.ModOrder, "gray");
    spread = fft(symbols, [], 1)/sqrt(cfg.multi_dft);
    grid = zeros(cfg.multi_fft, frames, cfg.multi_users);
    for u = 1:cfg.multi_users, grid(allocation(:, u), :, u) = spread(:, :, u); end
    wave = ifft(grid, [], 1)*sqrt(cfg.multi_fft);
    wave = [wave(end-cfg.multi_cp+1:end, :, :); wave];
    energy = mean(abs(qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray")).^2);
    for u = 1:cfg.multi_users
        if condition.Enabled
            gain = 10^(-cfg.multi_user_backoff(u)/20);
            branch = wave(:, :, u)*gain;
            branch = mmw.pa_envelope(branch, energy*cfg.multi_dft/cfg.multi_fft, ...
                cfg.pa_smoothness, cfg.pa_phase_limit_deg)/gain;
            % Linear user-specific RF memory, shorter than the cyclic prefix.
            branch = reshape(filter([1; 0; 0.12*(u-1)/3*exp(1j*pi*u/4)], 1, branch(:)), size(branch));
            wave(:, :, u) = branch;
        end
    end
    aggregate = sum(wave, 3);
    before = aggregate;
    if condition.Enabled
        aggregate = mmw.mzm_envelope(aggregate, condition.Drive, ...
            cfg.multi_users*energy*cfg.multi_dft/cfg.multi_fft);
    else
        aggregate = condition.Drive*aggregate;
    end
    after = aggregate/condition.Drive;
    if isfinite(condition.SNR)
        aggregate = aggregate + sqrt(energy/10^(condition.SNR/10)/2) * ...
            (randn(stream, size(aggregate))+1j*randn(stream, size(aggregate)));
    end
    received = fft(aggregate(cfg.multi_cp+1:end, :)/condition.Drive, [], 1)/sqrt(cfg.multi_fft);
    recovered = zeros(size(symbols));
    if part == 1, channel = zeros(cfg.multi_dft, cfg.multi_users); end
    for u = 1:cfg.multi_users
        observed = received(allocation(:, u), :);
        if part == 1
            % Training-payload labels only. Freeze this supervised linear
            % front-end for validation and test; no test-reference fitting.
            channel(:, u) = sum(conj(spread(:, :, u)).*observed, 2)./sum(abs(spread(:, :, u)).^2, 2);
        end
        h = channel(:, u); h(abs(h) < 1e-6) = 1e-6;
        recovered(:, :, u) = ifft(observed./h, [], 1)*sqrt(cfg.multi_dft);
    end
    indices = 2:cfg.multi_dft; rows = numel(indices)*frames;
    x = zeros(rows, 4*cfg.multi_users, "single"); y = zeros(rows, 2*cfg.multi_users, "single");
    baseline = zeros(rows, cfg.multi_users);
    for u = 1:cfg.multi_users
        current = reshape(recovered(indices, :, u), [], 1);
        previous = reshape(recovered(indices-1, :, u), [], 1);
        target = reshape(symbols(indices, :, u), [], 1);
        x(:, (u-1)*4+(1:4)) = single([real(current) imag(current) real(previous) imag(previous)]);
        y(:, (u-1)*2+(1:2)) = single([real(target) imag(target)]);
        baseline(:, u) = current;
    end
    data.("X_"+names(part)) = x; data.("Y_"+names(part)) = y;
    if part == 3
        data.baseline_test = baseline;
        data.spectrum_before = mean(abs(fftshift(fft(before(cfg.multi_cp+1:end, :), [], 1), 1)).^2, 2);
        data.spectrum_after = mean(abs(fftshift(fft(after(cfg.multi_cp+1:end, :), [], 1), 1)).^2, 2);
        data.spectrum_frequency_hz = (-cfg.multi_fft/2:cfg.multi_fft/2-1).'*cfg.multi_sample_rate_hz/cfg.multi_fft;
    end
end
data.feature_user = repelem(1:cfg.multi_users, 2);
data.frontend_channel_training = channel;
data.seed = seed;
data.ideal_symbols = qammod((0:condition.ModOrder-1).', condition.ModOrder, "gray");
data.ideal_bits = de2bi((0:condition.ModOrder-1).', log2(condition.ModOrder), "left-msb");
end

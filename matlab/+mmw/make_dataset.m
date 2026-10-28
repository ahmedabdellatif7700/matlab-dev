function data = make_dataset(cfg, seed, pa_enabled, backoff_db, snr_db)
%MAKE_DATASET Split at burst generation, before filtering or windowing.
counts = [cfg.train_symbols cfg.validation_symbols cfg.test_symbols];
names = {"train", "validation", "test"};
data = struct();
for partition = 1:3
    % Same symbols/noise draws for each seed/partition across conditions.
    % Different partitions use distinct random streams and separate filters.
    burst_seed = seed + partition*100000;
    [rx, tx] = mmw.generate_burst(cfg, counts(partition), burst_seed, ...
        pa_enabled, backoff_db, snr_db);
    indices = (cfg.equalizer_taps:counts(partition)).';
    x = zeros(numel(indices), 2*cfg.equalizer_taps, "single");
    for tap = 1:cfg.equalizer_taps
        samples = rx(indices-tap+1);
        x(:, 2*tap-1) = single(real(samples));
        x(:, 2*tap) = single(imag(samples));
    end
    y = single([real(tx(indices)) imag(tx(indices))]);
    data.("X_" + names{partition}) = x;
    data.("Y_" + names{partition}) = y;
end
data.mod_order = cfg.mod_order;
data.ideal_symbols = qammod((0:cfg.mod_order-1).', cfg.mod_order, "gray");
data.ideal_bits = de2bi((0:cfg.mod_order-1).', log2(cfg.mod_order), "left-msb");
data.seed = seed;
data.pa_enabled = pa_enabled;
data.backoff_db = backoff_db;
data.reference_snr_db = snr_db;
end

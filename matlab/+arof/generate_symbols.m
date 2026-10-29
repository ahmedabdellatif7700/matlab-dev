function data = generate_symbols(cfg)
%GENERATE_SYMBOLS Known calibration prefix, independent payload and guards.
stream = RandStream("mt19937ar","Seed",cfg.seed);
count = 2*cfg.guard_symbols+cfg.training_symbols+cfg.payload_symbols;
indices = randi(stream,[0 cfg.modulation_order-1],count,1);
symbols = qammod(indices,cfg.modulation_order,"gray");
data = struct("symbols",symbols,"indices",indices);
end

function baseband = pulse_shape(cfg,symbols)
%PULSE_SHAPE Unit-energy RRC on the AWG grid.
sps = cfg.awg_sample_rate_hz/cfg.symbol_rate_hz;
count = numel(symbols);
h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,sps,"sqrt").';
up = zeros(count*sps,1); up(1:sps:end) = symbols;
baseband = fftfilt(h,[up; zeros(numel(h)-1,1)]);
end

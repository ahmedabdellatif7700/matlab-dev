function tx = transmit(cfg)
%TRANSMIT Batch convenience; main.m calls these stages explicitly.
data = arof.generate_symbols(cfg);
baseband = arof.pulse_shape(cfg,data.symbols);
tx = arof.upconvert(cfg,data,baseband);
end

function tx = transmit(cfg)
%TRANSMIT Batch convenience; main.m calls these stages explicitly.
data = arof_history.generate_symbols(cfg);
baseband = arof_history.pulse_shape(cfg,data.symbols);
tx = arof_history.upconvert(cfg,data,baseband);
end

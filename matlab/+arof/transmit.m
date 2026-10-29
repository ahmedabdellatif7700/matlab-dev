function tx = transmit(cfg)
%TRANSMIT Batch convenience; main.m calls these stages explicitly.
data = arof.generate_symbols(cfg);
baseband = arof.pulse_shape(cfg,data.symbols);
tx = arof.upconvert(cfg,data,baseband);
tx.if_voltage = arof.driver(cfg,tx.if_before_driver);
end

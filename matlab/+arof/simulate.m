function result = simulate(cfg)
%SIMULATE Explicit physical stages; no equalizer or neural training.
tx = arof.transmit(cfg);
optical_tx = arof.modulate(cfg,tx.if_voltage);
optical_rx = arof.propagate(cfg,optical_tx);
detected = arof.photodetect(cfg,optical_rx);
rx = arof.receive(cfg,detected,tx);
metrics = arof.measure(cfg,optical_tx,rx);
result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
    "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
end

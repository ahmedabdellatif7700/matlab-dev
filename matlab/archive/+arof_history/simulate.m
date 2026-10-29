function result = simulate(cfg)
%SIMULATE Explicit physical stages; no equalizer or neural training.
tx = arof_history.transmit(cfg);
optical_tx = arof_history.modulate(cfg,tx.if_voltage);
optical_rx = arof_history.propagate(cfg,optical_tx);
detected = arof_history.photodetect(cfg,optical_rx);
rx = arof_history.receive(cfg,detected,tx);
metrics = arof_history.measure(cfg,optical_tx,rx);
result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
    "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
end

function result = simulate(cfg)
%SIMULATE Explicit physical stages; no equalizer or neural training.
tx = arof.transmitter.run(cfg);
optical_tx = arof.optical_link.modulate(cfg,tx.if_voltage);
optical_rx = arof.optical_link.propagate(cfg,optical_tx);
detected = arof.optical_link.photodetect(cfg,optical_rx);
rx = arof.receiver.run(cfg,detected,tx);
metrics = arof.receiver.measure(cfg,optical_tx,rx);
result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
    "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
end

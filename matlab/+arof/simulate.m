function result = simulate(cfg)
%SIMULATE Explicit physical stages; no equalizer or neural training.
tx = arof.transmit(cfg);
optical_tx = arof.modulate(cfg,tx.if_voltage);
optical_rx = arof.propagate(cfg,optical_tx);
detected = arof.photodetect(cfg,optical_rx);
rx = arof.receive(cfg,detected,tx);
payload = rx.payload_indices;
reference = rx.reference(payload); prediction = rx.sampled(payload);
actual_bits = qamdemod(reference,cfg.modulation_order,"gray","OutputType","bit");
predicted_bits = qamdemod(prediction,cfg.modulation_order,"gray","OutputType","bit");
metrics = struct("evm_percent",100*sqrt(mean(abs(prediction-reference).^2)/mean(abs(reference).^2)), ...
    "ber",nnz(actual_bits ~= predicted_bits)/numel(actual_bits), ...
    "bit_errors",nnz(actual_bits ~= predicted_bits),"bits",numel(actual_bits), ...
    "drive_rms_v",rms(optical_tx.drive_voltage), ...
    "drive_vpp_v",max(optical_tx.drive_voltage)-min(optical_tx.drive_voltage), ...
    "timing_offset_samples",rx.timing_offset_samples,"calibration_gain",rx.calibration_gain);
result = struct("configuration",cfg,"tx",tx,"optical_tx",optical_tx, ...
    "optical_rx",optical_rx,"detected",detected,"rx",rx,"metrics",metrics);
end

function cfg = paper_operating_point(base,modulation,seed,input_mv)
%PAPER_OPERATING_POINT Known transmitted waveform scaled BEFORE the driver.
% Vpp depends on its PAPR; receiver data are not used to set transmit gain.
cfg=base; cfg.modulation_order=modulation; cfg.seed=seed;
cfg.drive_rms_v=.1;
tx=arof.transmit(cfg);
input_vpp=max(tx.if_before_driver)-min(tx.if_before_driver);
cfg.drive_rms_v=.1*(input_mv/1000)/input_vpp;
end

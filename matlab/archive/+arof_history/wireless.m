function link = wireless(cfg,detected)
%WIRELESS Ideal Friis hop; no calibrated RF amplifier/noise model.
lambda = 299792458/cfg.rf_carrier_hz;
path_loss_db = 20*log10(4*pi*cfg.wireless_distance_m/lambda);
path_gain = 10^((cfg.tx_antenna_gain_dbi+cfg.rx_antenna_gain_dbi-path_loss_db)/20);
rf = detected.rf_envelope*path_gain;
link = struct("rf_envelope",rf,"path_loss_db",path_loss_db);
end

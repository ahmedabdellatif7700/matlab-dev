function tx = upconvert(cfg,data,baseband)
%UPCONVERT Electrical IF, fixed RMS calibration and analog reconstruction.
sps = cfg.awg_sample_rate_hz/cfg.symbol_rate_hz;
ratio = cfg.internal_sample_rate_hz/cfg.awg_sample_rate_hz;
assert(sps == round(sps) && ratio == round(ratio),"Integer sample ratios required.");
assert(cfg.if_hz+(1+cfg.rrc_rolloff)*cfg.symbol_rate_hz/2 < cfg.awg_sample_rate_hz/2, ...
    "The IF signal exceeds the AWG Nyquist band.");
t = (0:numel(baseband)-1).'/cfg.awg_sample_rate_hz;
energy = mean(abs(qammod((0:cfg.modulation_order-1).',cfg.modulation_order,"gray")).^2);
nominal_rms = sqrt(energy/(2*sps));
voltage_scale = cfg.drive_rms_v/nominal_rms;
if_awg = voltage_scale*real(baseband.*exp(1j*2*pi*cfg.if_hz*t));
if_before_driver = resample(if_awg,ratio,1);
tx = struct("symbols",data.symbols,"indices",data.indices,"baseband_awg",baseband, ...
    "if_awg",if_awg,"if_before_driver",if_before_driver,"voltage_scale",voltage_scale, ...
    "awg_sps",sps,"internal_sps",sps*ratio);
end

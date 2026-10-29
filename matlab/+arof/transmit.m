function tx = transmit(cfg)
%TRANSMIT Gray QAM/RRC at the AWG grid, IF, then analog reconstruction.
sps = cfg.awg_sample_rate_hz/cfg.symbol_rate_hz;
ratio = cfg.internal_sample_rate_hz/cfg.awg_sample_rate_hz;
assert(sps == round(sps) && ratio == round(ratio),"Integer sample ratios required.");
assert(cfg.if_hz+(1+cfg.rrc_rolloff)*cfg.symbol_rate_hz/2 < cfg.awg_sample_rate_hz/2, ...
    "The IF signal exceeds the AWG Nyquist band.");
stream = RandStream("mt19937ar","Seed",cfg.seed);
count = 2*cfg.guard_symbols+cfg.training_symbols+cfg.payload_symbols;
indices = randi(stream,[0 cfg.modulation_order-1],count,1);
symbols = qammod(indices,cfg.modulation_order,"gray");
h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,sps,"sqrt").';
up = zeros(count*sps,1); up(1:sps:end) = symbols;
baseband = conv(up,h);
t = (0:numel(baseband)-1).'/cfg.awg_sample_rate_hz;
energy = mean(abs(qammod((0:cfg.modulation_order-1).',cfg.modulation_order,"gray")).^2);
nominal_rms = sqrt(energy/(2*sps));
voltage_scale = cfg.drive_rms_v/nominal_rms;
if_awg = voltage_scale*real(baseband.*exp(1j*2*pi*cfg.if_hz*t));
if_voltage = resample(if_awg,ratio,1)*cfg.driver_voltage_gain;
tx = struct("symbols",symbols,"indices",indices,"baseband_awg",baseband, ...
    "if_awg",if_awg,"if_voltage",if_voltage,"voltage_scale",voltage_scale, ...
    "awg_sps",sps,"internal_sps",sps*ratio);
end

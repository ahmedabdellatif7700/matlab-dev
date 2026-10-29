classdef transmitter
%TRANSMITTER QAM symbols, RRC shaping and electrical IF drive.
    methods (Static)
        function data = generate_symbols(cfg)
            %GENERATE_SYMBOLS Known calibration prefix, independent payload and guards.
            stream = RandStream("mt19937ar","Seed",cfg.seed);
            count = 2*cfg.guard_symbols+cfg.training_symbols+cfg.payload_symbols;
            indices = randi(stream,[0 cfg.modulation_order-1],count,1);
            symbols = qammod(indices,cfg.modulation_order,"gray");
            data = struct("symbols",symbols,"indices",indices);
        end

        function baseband = pulse_shape(cfg,symbols)
            %PULSE_SHAPE Unit-energy RRC on the AWG grid.
            sps = cfg.awg_sample_rate_hz/cfg.symbol_rate_hz;
            count = numel(symbols);
            h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,sps,"sqrt").';
            up = zeros(count*sps,1); up(1:sps:end) = symbols;
            baseband = fftfilt(h,[up; zeros(numel(h)-1,1)]);
        end

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
            if_voltage = driver(cfg,if_before_driver);
            tx = struct("symbols",data.symbols,"indices",data.indices,"baseband_awg",baseband, ...
                "if_awg",if_awg,"if_voltage",if_voltage,"if_before_driver",if_before_driver,"voltage_scale",voltage_scale, ...
                "awg_sps",sps,"internal_sps",sps*ratio);
        end

        function tx = run(cfg)
            %RUN Batch convenience; main.m calls these stages explicitly.
            data = arof.transmitter.generate_symbols(cfg);
            baseband = arof.transmitter.pulse_shape(cfg,data.symbols);
            tx = arof.transmitter.upconvert(cfg,data,baseband);
        end

        function cfg = operating_point(base,modulation,seed,input_mv)
            %OPERATING_POINT Known transmitted waveform scaled BEFORE the driver.
            % Vpp depends on its PAPR; receiver data are not used to set transmit gain.
            cfg=base; cfg.modulation_order=modulation; cfg.seed=seed;
            cfg.drive_rms_v=.1;
            tx=arof.transmitter.run(cfg);
            input_vpp=max(tx.if_before_driver)-min(tx.if_before_driver);
            cfg.drive_rms_v=.1*(input_mv/1000)/input_vpp;
        end

    end
end

function voltage = driver(cfg,input)
%DRIVER Real IF driver; compressed option is a declared identification model.
voltage = cfg.driver_voltage_gain*input;
if cfg.driver_model == "linear"
    return;
end
assert(any(cfg.driver_model == ["rapp" "rapp_envelope"]),"Unknown driver model.");
assert(isfinite(cfg.driver_saturation_v) && cfg.driver_saturation_v>0,"Invalid saturation.");
p = cfg.driver_smoothness;
assert(p>0,"Driver smoothness must be positive.");
if cfg.driver_model == "rapp_envelope"
    analytic = hilbert(voltage);
    voltage = real(analytic./(1+(abs(analytic)/cfg.driver_saturation_v).^(2*p)).^(1/(2*p)));
else
    voltage = voltage./(1+(abs(voltage)/cfg.driver_saturation_v).^(2*p)).^(1/(2*p));
end
end

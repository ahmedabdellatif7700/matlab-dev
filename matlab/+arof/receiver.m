classdef receiver
%RECEIVER Envelope detection, baseband recovery and BER/EVM.
    methods (Static)
        function detector = downconvert(cfg,link)
            %DOWNCONVERT RF envelope detection, receiver DC rejection and IF mixing.
            rf = link.rf_envelope;
            switch cfg.envelope_detector_model
                case "ideal_amplitude"
                    envelope = abs(rf);
                case "square_law"
                    envelope = abs(rf).^2;
                otherwise
                    error("Unknown RF detector law.");
            end
            % A lumped post-detector real white-noise model. PSD scales with sample rate;
            % this is NOT an independently calibrated thermal/shot/ASE noise budget.
            if cfg.noise_enabled
                assert(isfinite(cfg.if_noise_psd) && cfg.if_noise_psd >= 0,"Invalid IF noise PSD.");
                noise_stream = RandStream("mt19937ar","Seed",cfg.noise_seed);
                envelope = envelope+sqrt(cfg.if_noise_psd*cfg.internal_sample_rate_hz)* ...
                    randn(noise_stream,size(envelope));
            end
            envelope_ac = envelope-mean(envelope); % Receiver DC rejection, no target use.
            t = (0:numel(envelope)-1).'/cfg.internal_sample_rate_hz;
            baseband = 2*envelope_ac.*exp(-1j*2*pi*cfg.if_hz*t);
            detector = struct("rf_envelope",rf,"envelope",envelope,"baseband",baseband, ...
                "wireless_path_loss_db",link.path_loss_db);
        end

        function rx = recover(cfg,detector,tx)
            %RECOVER Matched RRC, prefix-only timing calibration and symbol sampling.
            baseband = detector.baseband;
            h = rcosdesign(cfg.rrc_rolloff,cfg.rrc_span_symbols,tx.internal_sps,"sqrt").';
            matched = fftfilt(h,[baseband; zeros(numel(h)-1,1)]);
            % Sum of Tx and Rx RRC delays, expressed on the internal grid.
            origin = cfg.rrc_span_symbols*tx.internal_sps+1;
            keep = cfg.guard_symbols+(1:cfg.training_symbols+cfg.payload_symbols).';
            centers = origin+(keep-1)*tx.internal_sps;
            reference = tx.symbols(keep);
            calibration = 1:cfg.training_symbols;
            interpolator = spline((1:numel(matched)).',matched);
            best_error = Inf; best_offset = 0; best_gain = 1;
            for offset = linspace(-0.05*tx.internal_sps,0.05*tx.internal_sps,401)
                samples = reshape(ppval(interpolator,centers(calibration)+offset),[],1);
                gain = (samples'*reference(calibration))/(samples'*samples);
                error = mean(abs(gain*samples-reference(calibration)).^2);
                if error < best_error
                    best_error = error; best_offset = offset; best_gain = gain;
                end
            end
            sampled = reshape(ppval(interpolator,centers+best_offset),[],1)*best_gain;
            payload = cfg.training_symbols+(1:cfg.payload_symbols);
            rx = detector;
            rx.sampled = sampled; rx.reference = reference; rx.payload_indices = payload;
            rx.timing_offset_samples = best_offset; rx.calibration_gain = best_gain;
        end

        function metrics = measure(cfg,optical_tx,rx)
            %MEASURE Held-out payload BER and continuous-output EVM, no equalizer.
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
        end

        function rx = run(cfg,detected,tx)
            %RUN Batch convenience; main.m exposes each receiver stage.
            link = arof.optical_link.wireless(cfg,detected);
            detector = arof.receiver.downconvert(cfg,link);
            rx = arof.receiver.recover(cfg,detector,tx);
        end

    end
end

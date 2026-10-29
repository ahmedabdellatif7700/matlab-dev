classdef optical_link
%OPTICAL_LINK MZM, optical fiber, photodetection and wireless hop.
    methods (Static)
        function optical = modulate(cfg,voltage)
            %MODULATE Balanced single-arm field transfer, including common optical phase.
            if isinf(cfg.mzm_bandwidth_hz)
                drive = voltage;
            else
                drive = real(filter_envelope(voltage,cfg.internal_sample_rate_hz, ...
                    cfg.filter_guard_samples,@(f) 1./(1+1j*f/cfg.mzm_bandwidth_hz)));
            end
            phase = pi*(cfg.mzm_bias_v+drive)/cfg.mzm_vpi_v;
            field = (1+exp(1j*phase))/2;
            optical = struct("modulated_field",field,"lo_field",1, ...
                "drive_voltage",drive,"power_units","normalized; not calibrated optical watts");
        end

        function optical = propagate(cfg,optical)
            %PROPAGATE Each optical band sees its own dispersion phase.
            % Common beta0/beta1 removed. LO offset is relative to the modulated laser.
            length_m = cfg.fiber_length_km*1000;
            beta2 = -(cfg.dispersion_ps_nm_km*1e-6)*cfg.modulated_wavelength_m^2/(2*pi*299792458);
            field_gain = 10^((-cfg.attenuation_db_km*cfg.fiber_length_km+cfg.ideal_edfa_gain_db)/20);
            phase = @(f) exp(-1j*beta2*length_m*(2*pi*f).^2/2);
            optical.modulated_field = field_gain*filter_envelope(optical.modulated_field, ...
                cfg.internal_sample_rate_hz,cfg.filter_guard_samples,phase);
            optical.lo_field = field_gain*optical.lo_field*phase(cfg.rf_carrier_hz);
            optical.beta2_s2_m = beta2;
        end

        function detected = photodetect(cfg,optical)
            %PHOTODETECT Analytic square-law beat term around the positive RF carrier.
            % DC and optical-frequency components are excluded; no 60 GHz sampled tone.
            power_dbm=NaN; responsivity=1;
            % JSON round trips decode the disabled NaN setting as an empty value.
            if isfield(cfg,'received_optical_power_dbm') && ~isempty(cfg.received_optical_power_dbm) && ...
                    ~isnan(cfg.received_optical_power_dbm)
                % ROP is defined as total time-average two-tone optical power at the PD.
                % The optical-carrier cross term averages out on an optical power meter.
                assert(isscalar(cfg.received_optical_power_dbm) && isfinite(cfg.received_optical_power_dbm));
                total=mean(abs(optical.modulated_field).^2)+mean(abs(optical.lo_field).^2);
                watts=1e-3*10^(cfg.received_optical_power_dbm/10);
                scale=sqrt(watts/total);
                optical.modulated_field=scale*optical.modulated_field;
                optical.lo_field=scale*optical.lo_field;
                power_dbm=10*log10((mean(abs(optical.modulated_field).^2)+ ...
                    mean(abs(optical.lo_field).^2))/1e-3);
            end
            if isfield(cfg,'pd_responsivity_a_w'), responsivity=cfg.pd_responsivity_a_w; end
            assert(isscalar(responsivity) && isfinite(responsivity) && responsivity>0);
            beat = 2*responsivity*optical.lo_field.*conj(optical.modulated_field);
            if ~isinf(cfg.pd_bandwidth_hz)
                beat = filter_envelope(beat,cfg.internal_sample_rate_hz, ...
                    cfg.filter_guard_samples,@(f) 1./(1+1j*(cfg.rf_carrier_hz+f)/cfg.pd_bandwidth_hz));
            end
            detected = struct("rf_envelope",beat,"rf_carrier_hz",cfg.rf_carrier_hz, ...
                "received_optical_power_dbm",power_dbm);
        end

        function link = wireless(cfg,detected)
            %WIRELESS Ideal Friis hop; no calibrated RF amplifier/noise model.
            lambda = 299792458/cfg.rf_carrier_hz;
            path_loss_db = 20*log10(4*pi*cfg.wireless_distance_m/lambda);
            path_gain = 10^((cfg.tx_antenna_gain_dbi+cfg.rx_antenna_gain_dbi-path_loss_db)/20);
            rf = detected.rf_envelope*path_gain;
            link = struct("rf_envelope",rf,"path_loss_db",path_loss_db);
        end

    end
end

function y = filter_envelope(x,fs,guard,response)
%FILTER_ENVELOPE Zero-padded FFT filtering on the declared frequency grid.
x = x(:); nfft = 2^nextpow2(numel(x)+2*guard);
f = (-nfft/2:nfft/2-1).'*fs/nfft;
padded = zeros(nfft,1,"like",complex(x));
padded(guard+(1:numel(x))) = x;
filtered = ifft(fft(padded).*ifftshift(response(f)));
y = filtered(guard+(1:numel(x)));
end

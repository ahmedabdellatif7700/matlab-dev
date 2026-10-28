function [y, frequency, response] = optical_fading(x, cfg, sideband, length_km, guard)
%OPTICAL_FADING Small-signal chirp-free RF-over-fiber dispersion response.
% DSB is carrier/sideband beating, not optical-field all-pass dispersion.
% SSB removes the known carrier phase and bulk RF group delay analytically.
x = x(:);
nfft = 2^nextpow2(numel(x)+2*guard);
frequency = (-nfft/2:nfft/2-1).'*cfg.symbol_rate_hz*cfg.samples_per_symbol/nfft;
kappa = pi*(cfg.dispersion_ps_nm_km*1e-6)*cfg.optical_wavelength_m^2 * ...
    (length_km*1000)/299792458;
if sideband == "DSB"
    response = cos(kappa*(cfg.rf_carrier_hz+frequency).^2);
elseif sideband == "SSB"
    response = exp(1j*kappa*frequency.^2);
else
    error("Choose DSB or SSB.");
end
padded = zeros(nfft, 1); padded(guard+(1:numel(x))) = x;
filtered = ifft(fft(padded).*ifftshift(response));
y = filtered(guard+(1:numel(x)));
end

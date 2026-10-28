function y = pa_envelope(x, reference_power, smoothness, phase_limit_deg)
%PA_ENVELOPE Fixed saturation referenced to nominal waveform power.
u = abs(x)/sqrt(reference_power);
gain = (1+u.^(2*smoothness)).^(-1/(2*smoothness));
phase = deg2rad(phase_limit_deg)*u.^2./(1+u.^2);
y = x.*gain.*exp(1j*phase);
end

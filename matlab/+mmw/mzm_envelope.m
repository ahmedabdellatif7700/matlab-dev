function y = mzm_envelope(x, drive, reference_power)
%MZM_ENVELOPE Fundamental RF band of quadrature-biased ideal IM/DD MZM.
% P/P0 = (1+cos(pi/2 + pi*v_RF/Vpi))/2.
% RF band projection: -J1(m)exp(j*arg(x)), m=drive*|x|/sqrt(P_ref).
% Remove the fixed detector sign/gain: y=drive*x*2*J1(m)/m.
% All RF harmonic bands are rejected; narrowband envelope approximation.
assert(drive > 0 && reference_power > 0, "Positive MZM drive and reference power required.");
m = drive*abs(x)/sqrt(reference_power);
ratio = ones(size(m));
nonzero = m > 1e-10;
ratio(nonzero) = 2*besselj(1, m(nonzero))./m(nonzero);
y = drive*x.*ratio;
end

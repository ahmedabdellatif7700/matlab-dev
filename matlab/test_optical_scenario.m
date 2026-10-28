function tests = test_optical_scenario
tests = functiontests(localfunctions);
end

function testZeroLengthAndSsbMagnitude(testCase)
cfg = mmw.scenario_config(); cfg.dispersion_ps_nm_km = 17;
cfg.optical_wavelength_m = 1550e-9;
stream = RandStream("mt19937ar", "Seed", 77);
x = randn(stream, 4096, 1)+1j*randn(stream, 4096, 1);
for kind = ["DSB" "SSB"]
    y = mmw.optical_fading(x, cfg, kind, 0, 1024);
    verifyEqual(testCase, y, x, "AbsTol", 1e-12);
end
[~, ~, h] = mmw.optical_fading(x, cfg, "SSB", 10, 1024);
verifyEqual(testCase, abs(h), ones(size(h)), "AbsTol", 1e-14);
end

function testDsbCarrierNull(testCase)
cfg = mmw.scenario_config(); cfg.dispersion_ps_nm_km = 17;
cfg.optical_wavelength_m = 1550e-9;
length_km = 299792458/(2*17e-6*cfg.optical_wavelength_m^2*cfg.rf_carrier_hz^2)/1000;
[~, f, h] = mmw.optical_fading(ones(1024, 1), cfg, "DSB", length_km, 1024);
verifyLessThan(testCase, abs(h(f == 0)), 1e-12);
verifyGreaterThan(testCase, length_km, 0.9);
verifyLessThan(testCase, length_km, 1.1);
end

function testGuardConvergence(testCase)
cfg = mmw.scenario_config(); cfg.dispersion_ps_nm_km = 17;
cfg.optical_wavelength_m = 1550e-9;
stream = RandStream("mt19937ar", "Seed", 78);
up = zeros(16000, 1); up(1:8:end) = randn(stream, 2000, 1)+1j*randn(stream, 2000, 1);
h = rcosdesign(.25, 24, 8, "sqrt").'; x = conv(up, h);
a = mmw.optical_fading(x, cfg, "DSB", 1.02, 4096);
b = mmw.optical_fading(x, cfg, "DSB", 1.02, 8192);
verifyLessThan(testCase, norm(a-b)/norm(b), 0.005);
end

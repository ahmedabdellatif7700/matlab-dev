function tests = test_pa_scenario
tests = functiontests(localfunctions);
end

function testLinearCalibrationAndBackoff(testCase)
cfg = mmw.scenario_config();
[rx, tx, bits] = mmw.generate_burst(cfg, 12000, 101, false, 0, Inf);
verifyLessThan(testCase, norm(rx-tx)/norm(tx), 0.005);
verifyEqual(testCase, qamdemod(tx, 16, "gray", "OutputType", "bit"), bits);
[rx0, tx0] = mmw.generate_burst(cfg, 20000, 102, false, 0, 30);
[rx6, tx6] = mmw.generate_burst(cfg, 20000, 102, false, 6, 30);
snr0 = 10*log10(sum(abs(tx0).^2)/sum(abs(rx0-tx0).^2));
snr6 = 10*log10(sum(abs(tx6).^2)/sum(abs(rx6-tx6).^2));
verifyEqual(testCase, tx0, tx6);
verifyLessThan(testCase, abs(snr0-30), 0.5);
verifyLessThan(testCase, abs(snr6-24), 0.5);
end

function testIndependentPartitionsAndCausalWindows(testCase)
cfg = mmw.scenario_config();
cfg.train_symbols = 200; cfg.validation_symbols = 100; cfg.test_symbols = 300;
a = mmw.make_dataset(cfg, 42, true, 3, 30);
b = mmw.make_dataset(cfg, 42, true, 3, 30);
verifyEqual(testCase, a.X_train, b.X_train);
verifyEqual(testCase, size(a.X_train), [199 4]);
verifyEqual(testCase, size(a.X_validation), [99 4]);
verifyEqual(testCase, size(a.X_test), [299 4]);
verifyEqual(testCase, a.X_train(2:end, 3:4), a.X_train(1:end-1, 1:2));
verifyFalse(testCase, isequal(a.Y_train(1:99, :), a.Y_validation));
c = mmw.make_dataset(cfg, 42, true, 0, 24);
verifyEqual(testCase, a.Y_train, c.Y_train);
verifyEqual(testCase, a.Y_test, c.Y_test);
end

function testCompressionAndPhase(testCase)
cfg = mmw.scenario_config();
[linear, tx] = mmw.generate_burst(cfg, 2000, 103, false, 0, Inf);
[distorted, tx2] = mmw.generate_burst(cfg, 2000, 103, true, 0, Inf);
verifyEqual(testCase, tx, tx2);
verifyLessThan(testCase, norm(distorted), norm(linear));
verifyGreaterThan(testCase, norm(distorted-tx)/norm(tx), 0.1);
verifyGreaterThan(testCase, angle(sum(distorted.*conj(tx))), 0);
end

function testOversamplingConvergence(testCase)
cfg = mmw.scenario_config();
[rx8, tx8] = mmw.generate_burst(cfg, 4000, 104, true, -3, Inf);
cfg.samples_per_symbol = 16;
[rx16, tx16] = mmw.generate_burst(cfg, 4000, 104, true, -3, Inf);
verifyEqual(testCase, tx8, tx16);
verifyLessThan(testCase, norm(rx8-rx16)/norm(rx16), 0.005);
end

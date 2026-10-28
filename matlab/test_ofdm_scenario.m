function tests = test_ofdm_scenario
tests = functiontests(localfunctions);
end

function testNoiselessPilotAndCpRecovery(testCase)
cfg = mmw.suite_config(5); cfg.train_symbols = 1040;
cfg.validation_symbols = 520; cfg.test_symbols = 1040;
conditions = mmw.suite_conditions(cfg);
for order = [16 64]
    row = table2struct(conditions(find(conditions.ModOrder == order & conditions.Group == "Control", 1), :));
    data = mmw.make_ofdm_dataset(cfg, 42, row);
    target = complex(double(data.Y_test(:, 1)), double(data.Y_test(:, 2)));
    verifyEqual(testCase, data.baseline_test, target, "AbsTol", 1e-11);
    verifyEqual(testCase, data.frontend_channel_test, data.linear_channel, "AbsTol", 1e-12);
    verifyEqual(testCase, size(data.X_test), [1000 6]);
end
end

function testPilotEstimateDoesNotUsePayload(testCase)
cfg = mmw.suite_config(5); cfg.train_symbols = 520;
cfg.validation_symbols = 520; cfg.test_symbols = 1040;
conditions = mmw.suite_conditions(cfg); row = table2struct(conditions(conditions.Condition == "qam16_linear_noiseless", :));
a = mmw.make_ofdm_dataset(cfg, 42, row);
cfg.ofdm_payload_seed_offset = 123;
b = mmw.make_ofdm_dataset(cfg, 42, row);
verifyFalse(testCase, isequal(a.Y_test, b.Y_test));
verifyEqual(testCase, a.frontend_channel_test, b.frontend_channel_test, "AbsTol", 1e-12);
end

function testNonlinearOversamplingConvergence(testCase)
cfg = mmw.suite_config(5); cfg.train_symbols = 520;
cfg.validation_symbols = 520; cfg.test_symbols = 1040;
conditions = mmw.suite_conditions(cfg); row = table2struct(conditions(1, :));
row.SNR = Inf; row.Fiber_km = 0;
a = mmw.make_ofdm_dataset(cfg, 42, row);
cfg.ofdm_oversampling = 8;
cfg.ofdm_sample_rate_hz = cfg.ofdm_fft*cfg.ofdm_oversampling*cfg.ofdm_spacing_hz;
b = mmw.make_ofdm_dataset(cfg, 42, row);
verifyEqual(testCase, a.Y_test, b.Y_test);
verifyLessThan(testCase, norm(a.baseline_test-b.baseline_test)/norm(b.baseline_test), 0.005);
end

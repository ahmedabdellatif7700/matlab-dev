function tests = test_multiuser_scenario
tests = functiontests(localfunctions);
end

function testAllocationsAndRoundtrip(testCase)
cfg = mmw.suite_config(4); cfg.train_symbols = 256;
cfg.validation_symbols = 128; cfg.test_symbols = 512;
conditions = mmw.suite_conditions(cfg);
for guard = cfg.multi_guards
    [bins, signed] = mmw.multiuser_bins(cfg, guard);
    verifyEqual(testCase, numel(unique(bins)), cfg.multi_users*cfg.multi_dft);
    verifyEqual(testCase, signed(1, 2)-signed(end, 1)-1, guard);
end
control = table2struct(conditions(end, :));
data = mmw.make_multiuser_dataset(cfg, 42, control);
target = complex(double(data.Y_test(:, 1:2:end)), double(data.Y_test(:, 2:2:end)));
verifyEqual(testCase, data.baseline_test, target, "AbsTol", 1e-12);
verifyEqual(testCase, size(data.X_test), [480 16]);
verifyEqual(testCase, size(data.Y_test), [480 8]);
verifyEqual(testCase, data.feature_user, [1 1 2 2 3 3 4 4]);
verifyEqual(testCase, data.frontend_channel_training, ones(16, 4), "AbsTol", 1e-12);
end

function testNonlinearReproducibilityAndWindows(testCase)
cfg = mmw.suite_config(4); cfg.train_symbols = 256;
cfg.validation_symbols = 128; cfg.test_symbols = 512;
conditions = mmw.suite_conditions(cfg); row = table2struct(conditions(2, :));
a = mmw.make_multiuser_dataset(cfg, 42, row);
b = mmw.make_multiuser_dataset(cfg, 42, row);
verifyEqual(testCase, a.X_train, b.X_train);
verifyTrue(testCase, all(isfinite(a.X_test), "all"));
current = reshape(a.X_train(:, 1), 15, []);
previous = reshape(a.X_train(:, 3), 15, []);
verifyEqual(testCase, current(1:14, :), previous(2:15, :));
verifyFalse(testCase, isequal(a.Y_train(1:size(a.Y_validation, 1), :), a.Y_validation));
end

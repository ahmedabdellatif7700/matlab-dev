function tests = test_mzm_scenario
tests = functiontests(localfunctions);
end

function testStrongDriveOversampling(testCase)
cfg = mmw.suite_config(2); cfg.train_symbols = 400;
cfg.validation_symbols = 200; cfg.test_symbols = 3000;
conditions = mmw.suite_conditions(cfg);
row = table2struct(conditions(conditions.Condition == "qam64_mem1_drive180", :));
row.SNR = Inf;
a = mmw.make_suite_dataset(cfg, 42, row);
cfg.samples_per_symbol = 16;
b = mmw.make_suite_dataset(cfg, 42, row);
verifyEqual(testCase, a.Y_test, b.Y_test);
verifyLessThan(testCase, norm(a.baseline_test-b.baseline_test)/norm(b.baseline_test), 0.005);
end

function testFundamentalAgainstDirectIntensity(testCase)
theta = (0:32767).'*2*pi/32768;
for amplitude = [0 0.1 0.7 1.5 2.5]
    for phase = [0 0.7 -1.3]
        intensity = 0.5*(1+cos(pi/2+amplitude*cos(theta+phase)));
        fundamental = -4*mean(intensity.*exp(-1j*theta));
        expected = mmw.mzm_envelope(amplitude*exp(1j*phase), 1, 1);
        verifyEqual(testCase, fundamental, expected, "AbsTol", 1e-12);
    end
end
end

function testSmallSignalAndFolding(testCase)
x = [0; 0.1+0.2j; -0.3+0.4j];
verifyEqual(testCase, mmw.mzm_envelope(x, 1e-5, 1)/1e-5, x, "AbsTol", 1e-10);
verifyLessThan(testCase, real(mmw.mzm_envelope(4, 1, 1)), 0);
end

function testDatasetAndLabels(testCase)
cfg = mmw.suite_config(2); cfg.train_symbols = 200;
cfg.validation_symbols = 100; cfg.test_symbols = 300;
conditions = mmw.suite_conditions(cfg);
for row = [1 height(conditions)]
    condition = table2struct(conditions(row, :));
    a = mmw.make_suite_dataset(cfg, 42, condition);
    b = mmw.make_suite_dataset(cfg, 42, condition);
    verifyEqual(testCase, a.X_train, b.X_train);
    verifyEqual(testCase, size(a.X_test), [299 4]);
    target = complex(a.Y_test(:, 1), a.Y_test(:, 2));
    labels = qamdemod(target, condition.ModOrder, "gray");
    verifyEqual(testCase, real(a.ideal_symbols(labels+1)), double(real(target)));
    verifyEqual(testCase, imag(a.ideal_symbols(labels+1)), double(imag(target)));
    verifyFalse(testCase, isequal(a.Y_train(1:99, :), a.Y_validation));
end
end

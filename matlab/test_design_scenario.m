function tests = test_design_scenario
tests = functiontests(localfunctions);
end
function testLinearLimit(testCase)
cfg = mmw.design_config(); cases = mmw.design_candidates();
c = table2struct(cases(1,:)); c.MZM = false; c.Reference_SNR_dB = Inf;
[rx, tx] = mmw.design_burst(cfg,c,2000,911);
evm = sqrt(mean(abs(rx-tx).^2)/mean(abs(tx).^2));
verifyLessThan(testCase,evm,0.002);
end
function testIndependentNoiseAndDeterminism(testCase)
cfg = mmw.design_config(); cases = mmw.design_candidates(); c = table2struct(cases(1,:));
[a,t] = mmw.design_burst(cfg,c,256,911);
[b,u] = mmw.design_burst(cfg,c,256,911);
[d,v] = mmw.design_burst(cfg,c,256,912);
verifyEqual(testCase,a,b); verifyEqual(testCase,t,u);
verifyGreaterThan(testCase,norm(a-d),0); verifyGreaterThan(testCase,norm(t-v),0);
end
function testUnmodelledCascadeRejected(testCase)
cfg = mmw.design_config(); cases = mmw.design_candidates(); c = table2struct(cases(1,:));
c.Optical = true;
rejected = false;
try
    mmw.design_burst(cfg,c,256,911);
catch
    rejected = true;
end
verifyTrue(testCase,rejected);
end

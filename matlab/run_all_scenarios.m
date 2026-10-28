function run_dirs = run_all_scenarios(profile, visible)
%RUN_ALL_SCENARIOS Run all five studies without changing branches or commits.
if nargin < 1, profile = "study"; end
if nargin < 2, visible = "off"; end
addpath(fileparts(mfilename("fullpath")));
run_dirs = strings(5, 1);
for scenario = 1:5
    fprintf("\n=== Starting scenario %d/5 ===\n", scenario);
    if scenario == 1, cfg = mmw.scenario_config();
    else, cfg = mmw.suite_config(scenario); end
    cfg.profile = string(profile); cfg.visible = string(visible);
    if scenario == 1, run_dirs(scenario) = mmw.run_pa_benchmark(cfg);
    else, run_dirs(scenario) = mmw.run_suite(cfg); end
end
disp(table((1:5).', run_dirs, 'VariableNames', {'Scenario', 'RunDirectory'}));
end

%% Millimeter-wave impairment research scenarios
% Run from any working directory. See matlab/README.md for assumptions.
% The original MATLAB/Python/FPGA workflow is available in legacy_main.m.
clear;
clc;
close all;

matlab_dir = fileparts(mfilename("fullpath"));
addpath(matlab_dir);
scenario = 3; % 1: RF driver; 2: MZM; 3: combined RF/optical DSB and SSB.
if scenario == 1
    cfg = mmw.scenario_config();
else
    cfg = mmw.suite_config(scenario);
end

% A smoke run checks the full data/export/inference/plotting path.
cfg.profile = "study"; % "smoke" or "study"
cfg.visible = "on";    % "off" for batch execution
% cfg.python_exe = "/path/to/environment/bin/python";

if scenario == 1
    run_dir = mmw.run_pa_benchmark(cfg);
else
    run_dir = mmw.run_suite(cfg);
end
fprintf("\nScenario results: %s\n", run_dir);

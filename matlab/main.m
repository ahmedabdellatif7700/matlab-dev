%% Scenario 1: 60 GHz single-carrier RF driver benchmark
% Run from any working directory. See matlab/README.md for assumptions.
% The original MATLAB/Python/FPGA workflow is available in legacy_main.m.
clear;
clc;
close all;

matlab_dir = fileparts(mfilename("fullpath"));
addpath(matlab_dir);
cfg = mmw.scenario_config();

% A smoke run checks the full data/export/inference/plotting path.
cfg.profile = "study"; % "smoke" or "study"
cfg.visible = "on";    % "off" for batch execution
% cfg.python_exe = "/path/to/environment/bin/python";

run_dir = mmw.run_pa_benchmark(cfg);
fprintf("\nScenario results: %s\n", run_dir);

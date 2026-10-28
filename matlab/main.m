%% 60 GHz neural equalizer design: all experiment stages
% Legacy hardware workflow: legacy_main.m. Existing five sweeps: mmw.run_suite.
clear; clc; close all;
matlab_dir = fileparts(mfilename("fullpath")); addpath(matlab_dir);

%% 1. Physical link and experiment configuration
% Single carrier Gray QAM -> RRC -> RF PA or MZM -> optical/echo channel
% -> independent AWGN -> matched RRC -> known delay/gain removal -> tap window.
cfg = mmw.design_config();
% cfg.python_exe = "/path/to/environment/bin/python";
candidates = mmw.design_candidates();
disp(candidates);

%% 2. Independent training and validation waveforms; no test data generated
% design_burst exposes modulation, pulse shaping, nonlinearities, channel,
% noise, receive filtering, synchronization and sampling in order.
run_dir = mmw.prepare_design_data(cfg, candidates, "discovery");

%% 3. Python architecture/activation search and strong classical controls
% BCE bit heads; arbitrary hidden widths. FFE, full Volterra and RBF controls.
% All hyperparameters and four conditions selected exclusively on validation.
script = fullfile(cfg.root,"python","tune_equalizers.py");
command = sprintf('"%s" -B "%s" --run "%s" --phase discovery', ...
    cfg.python_exe, script, run_dir);
[status, output] = system(command, "-echo");
assert(status == 0, "Discovery failed: %s", output);

%% 3b. Focused 64-QAM bottleneck and small-activation ablation
extension = fullfile(cfg.root,"python","extend_design_search.py");
command = sprintf('"%s" -B "%s" --run "%s"',cfg.python_exe,extension,run_dir);
[status, output] = system(command,"-echo");
assert(status == 0,"Extended validation search failed: %s",output);
command = sprintf('"%s" -B "%s" --run "%s" --phase discovery', ...
    cfg.python_exe,script,run_dir);
[status, output] = system(command,"-echo");
assert(status == 0,"Final design locking failed: %s",output);

%% 4. Lock conditions, network architectures and BER targets before testing
selection = jsondecode(fileread(fullfile(run_dir,"selection.json")));
assert(selection.complete, ...
    "Not all four families qualified. Review discovery_models.csv; no success is assumed.");
ids = string(arrayfun(@(s) string(s.condition.ID), selection.selected));
selected = candidates(ismember(candidates.ID, ids), :);

%% 5. Fresh independent training/validation/test realizations for confirmation
% Every method receives identical inputs. Never choose models on test BER.
mmw.prepare_design_data(cfg, selected, "confirmation", run_dir);
command = sprintf('"%s" -B "%s" --run "%s" --phase confirmation', ...
    cfg.python_exe, script, run_dir);
[status, output] = system(command, "-echo");
assert(status == 0, "Confirmation failed: %s", output);

%% 6. BER, decision EVM and resource comparison, with paired uncertainty
mmw.verify_design_decisions(run_dir);
command = sprintf('"%s" -B "%s" --run "%s" --phase report', ...
    cfg.python_exe, script, run_dir);
[status, output] = system(command, "-echo");
assert(status == 0, "Report generation failed: %s", output);
metrics = readtable(fullfile(run_dir,"confirmation_metrics.csv"));
disp(metrics(:,["case","seed","method","ber","multiplications"]));
fprintf("Full metrics and frozen discovery design: %s\n", run_dir);
% Multiplications are algorithmic counts, not measured FPGA power or latency.
% A kernel or higher-order Volterra may match BER at greater arithmetic cost.

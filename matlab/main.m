%% Single physical A-RoF link: establish the baseline before equalization
clear; clc; close all;
matlab_dir = fileparts(mfilename("fullpath")); addpath(matlab_dir);
cfg = arof.configuration();
% The reported experiment uses IF modulation and optical beating near 60 GHz.
% Review PHYSICAL_LINK.md before interpreting these diagnostic results.
result = arof.simulate(cfg);
disp(result.metrics);
% Step 3 adds baseline reporting and physical/numerical controls.
% Dataset export and FFE/Volterra/NN comparison follow baseline review.
% NN development will use the supplied nn_training_model/ stages unchanged.

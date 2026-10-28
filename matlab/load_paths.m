function paths = load_paths()

paths = struct();

matlab_dir = fileparts(mfilename("fullpath"));
paths.root = fileparts(matlab_dir);

paths.cfg_file = fullfile(paths.root, "config", "paths.json");
paths.cfg = jsondecode(fileread(paths.cfg_file));

paths.matlab_dir = fullfile( ...
    paths.root, paths.cfg.directories.matlab);

paths.python_dir = fullfile( ...
    paths.root, paths.cfg.directories.python);

paths.cpp_dir = fullfile( ...
    paths.root, paths.cfg.directories.cpp);

paths.datasets_dir = fullfile( ...
    paths.root, paths.cfg.directories.datasets);

paths.models_dir = fullfile( ...
    paths.root, paths.cfg.directories.models);

paths.quantized_dir = fullfile( ...
    paths.root, paths.cfg.directories.quantized);

paths.rtl_dir = fullfile( ...
    paths.root, paths.cfg.directories.rtl_data);

paths.python_interpreter = ...
    paths.cfg.interpreters.python;

paths.cpp_program = fullfile( ...
    paths.root, paths.cfg.executables.main);

paths.train_script = fullfile( ...
    paths.root, paths.cfg.scripts.train);

paths.inference_script = fullfile( ...
    paths.root, paths.cfg.scripts.inference);

paths.quantize_script = fullfile( ...
    paths.root, paths.cfg.scripts.quantize);

paths.export_script = fullfile( ...
    paths.root, paths.cfg.scripts.export);

paths.model_file = fullfile( ...
    paths.root, paths.cfg.models.network);

paths.quantized_file = fullfile( ...
    paths.root, paths.cfg.quantized.network);

addpath(paths.matlab_dir);

ensure_dir(paths.datasets_dir);
ensure_dir(paths.models_dir);
ensure_dir(paths.quantized_dir);
ensure_dir(paths.rtl_dir);

end


function ensure_dir(folder)

if ~exist(folder, "dir")
    mkdir(folder);
end

end
function summary = run_numerical_tests()
%RUN_NUMERICAL_TESTS Explicit file discovery avoids MATLAB name ambiguity.
folder = fileparts(mfilename("fullpath")); addpath(folder);
files = dir(fullfile(folder, "test_*scenario.m"));
assert(numel(files) == 6, "Expected numerical tests for all six scenario test files.");
passed = 0;
for k = 1:numel(files)
    result = runtests(fullfile(files(k).folder, files(k).name));
    assertSuccess(result); passed = passed+numel(result);
end
sources = [dir(fullfile(folder, "+mmw", "*.m")); dir(fullfile(folder, "main.m")); ...
    dir(fullfile(folder, "run_all_scenarios.m")); dir(fullfile(folder, "run_numerical_tests.m")); files];
diagnostics = 0;
for k = 1:numel(sources)
    messages = checkcode(fullfile(sources(k).folder, sources(k).name), "-id");
    diagnostics = diagnostics+numel(messages);
    for j = 1:numel(messages)
        fprintf("%s:%d %s\n", sources(k).name, messages(j).line, messages(j).message);
    end
end
summary = struct("tests_passed", passed, "test_files", numel(files), ...
    "code_analyzer_diagnostics", diagnostics);
fprintf("Passed %d numerical tests in %d files; Code Analyzer diagnostics: %d\n", ...
    passed, numel(files), diagnostics);
end

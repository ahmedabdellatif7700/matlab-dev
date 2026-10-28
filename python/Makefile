	CONDA_ENV := machine-learning

# Always write the console log inside this python/ directory.
PROJECT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
SHELL := /bin/bash
.SHELLFLAGS := -o pipefail -c

PYTHON := conda run --no-capture-output -n $(CONDA_ENV) python
PIP := conda run --no-capture-output -n $(CONDA_ENV) python -m pip

MAIN_SCRIPT := equalizer.py
SOFTWARE_SCRIPT := software/software_inference.py
SOFTWARE_TRAINING_SCRIPT := software/software_training.py
DUMP_INIT_SCRIPT := dump_init.py
Q824_TRAIN_SCRIPT := train.py
GOLDEN_SCRIPT := gen_golden.py
OUTPUT := $(PROJECT_DIR)/output.txt

# Same dataset as config.DATASET_PATH; override with: make train-q824 DATASET=/path/training_data.mat
DATASET ?= $(abspath $(PROJECT_DIR)/../datasets/training_data.mat)
INIT ?= $(PROJECT_DIR)/init_params.txt
EPOCHS ?= 50
LR_SHIFT ?= 10

.PHONY: clean build run software-inference software-training dump-init train-q824 golden

clean:
	rm -rf "$(PROJECT_DIR)/artifacts/models"
	# Preserve MATLAB-generated test_inputs.mem; only clean Python exports.
	rm -f "$(PROJECT_DIR)/artifacts/inference/parameters.mem"
	rm -f "$(PROJECT_DIR)/artifacts/inference/software_outputs.csv"
	rm -rf "$(PROJECT_DIR)/plots"
	rm -f "$(OUTPUT)"

build:
	cd "$(PROJECT_DIR)" && $(PIP) install -r requirements.txt

run:
	@cd "$(PROJECT_DIR)" && $(PYTHON) -u "$(MAIN_SCRIPT)" 2>&1 | tee "$(OUTPUT)"

software-inference:
	@cd "$(PROJECT_DIR)" && $(PYTHON) $(SOFTWARE_SCRIPT)

software-training:
	@cd "$(PROJECT_DIR)" && $(PYTHON) $(SOFTWARE_TRAINING_SCRIPT)

# PyTorch initial weights (seed 42, before training) -> init_params.txt
dump-init:
	@cd "$(PROJECT_DIR)" && $(PYTHON) $(DUMP_INIT_SCRIPT)

# Unified Q8.24 on-chip training model vs float64 (starts from init_params.txt if present)
train-q824:
	@cd "$(PROJECT_DIR)" && $(PYTHON) -u $(Q824_TRAIN_SCRIPT) --dataset "$(DATASET)" --epochs $(EPOCHS) --lr-shift $(LR_SHIFT) $(if $(wildcard $(INIT)),--init "$(INIT)")

# Golden RAM images for the atomic testbenches -> artifacts/training/golden
golden:
	@cd "$(PROJECT_DIR)" && $(PYTHON) $(GOLDEN_SCRIPT) --dataset "$(DATASET)"

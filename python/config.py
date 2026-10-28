"""Project configuration: network, activation, training, fixed-point and paths.

The baseline verified FPGA inference configuration is MODE='real' and
ACTIVATION='sigmoid' (4-12-2). Other activations are software experiments.
"""
from pathlib import Path

# Root of the overall MATLAB/Python/FPGA project.
PYTHON_DIR = Path(__file__).resolve().parent
ROOT = PYTHON_DIR.parent
DATASET_PATH = ROOT / "datasets" / "training_data.mat"
# Two separate Python inference files; equalized_output_fpga.mat is written by C++.
ONNX_FP32_OUTPUT_PATH = ROOT / "datasets" / "equalized_output_onnx_fp32.mat"
SOFTWARE_Q8_8_OUTPUT_PATH = ROOT / "datasets" / "equalized_output_software_q8_8.mat"
ARTIFACT_DIR = PYTHON_DIR / "artifacts"
INFERENCE_DIR = ARTIFACT_DIR / "inference"
SOFTWARE_OUTPUT_PATH = INFERENCE_DIR / "software_outputs.csv"
# Exported by MATLAB; consumed directly by Python Q8.8 and KV260 C++.
TEST_INPUTS_MEM_PATH = INFERENCE_DIR / "test_inputs.mem"
PARAMETERS_MEM_PATH = INFERENCE_DIR / "parameters.mem"

# Unified real-valued equalizer: 2 temporal taps x (I, Q) -> 4-12-2.
# Sigmoid is the only activation currently verified in Q8.8 FPGA RTL.
MODE = "real"
ACTIVATION = "sigmoid"  # sigmoid, variable_alpha, stable_alpha
SUPPORTED_ACTIVATIONS = ("sigmoid", "variable_alpha", "stable_alpha")

INPUT_SIZE = 4
HIDDEN_SIZE = 12
OUTPUT_SIZE = 2

# Activation coefficients (variable-alpha is trainable; stable-alpha is fixed).
ALPHA_INITIAL = 4.0
STABLE_ALPHA = 4.0

# Unified online-style software SGD: one chronologically ordered sample/update.
# SGD hyperparameters are starting values; convergence requires evaluation.
OPTIMIZER = "sgd"  # Supported: sgd, adam (optional historical comparison)
LEARNING_RATE = 1e-3
BATCH_SIZE = 1
SHUFFLE = False
EPOCHS = 50
TRAIN_RATIO = 0.8
SEED = 42

# The existing inference hardware uses signed 16-bit Q8.8.
# Master-weight precision for FPGA SGD remains in hardware/q8_24_training_arithmetic.py.
TOTAL_BITS = 16
FRAC_BITS = 8

# The current KV260 RTL activation is standard sigmoid only.
# All three Python configurations export the same artifact filenames, but
# custom-alpha outputs are software models until supported in FPGA RTL.
FPGA_INFERENCE_SUPPORTED = (
    MODE == "real"
    and ACTIVATION == "sigmoid"
    and (INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE) == (4, 12, 2)
)

# Fixed artifact filenames: every experiment overwrites the previous run.
MODEL_PATH = ARTIFACT_DIR / "models" / "equalizer.onnx"
QUANT_PATH = ARTIFACT_DIR / "models" / "equalizer_q8_8.npz"

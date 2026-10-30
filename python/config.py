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

# Default legacy model; floating-point software also supports other widths/depths.
# Sigmoid is the only activation currently verified in Q8.8 FPGA RTL.
MODE = "real"
ACTIVATION = "sigmoid"  # See SUPPORTED_ACTIVATIONS.
SUPPORTED_ACTIVATIONS = (
    "sigmoid", "variable_alpha", "stable_alpha", "tanh", "silu", "softsign",
    "multilevel", "learned_multilevel",
)

INPUT_SIZE = 4
HIDDEN_SIZE = 12
HIDDEN_SIZES = (HIDDEN_SIZE,)  # e.g. (16, 8) for a deeper software model.
EXPORT_FIXED_POINT = True  # Disable for arbitrary software topology/activation.
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
    and (INPUT_SIZE, tuple(HIDDEN_SIZES), OUTPUT_SIZE) == (4, (12,), 2)
)

# Fixed artifact filenames: every experiment overwrites the previous run.
MODEL_PATH = ARTIFACT_DIR / "models" / "equalizer.onnx"
QUANT_PATH = ARTIFACT_DIR / "models" / "equalizer_q8_8.npz"

# One entry point: equalizer.py. No arguments preserves the original workflow.
WORKFLOW = "legacy"  # MATLAB research stages pass --run and --phase.
RESEARCH_RUN_DIR = None
RESEARCH_PHASE = "discovery"
# Research decisions live here; MATLAB owns physical link/waveform settings.
RESEARCH = dict(
    nn_epochs=600, nn_patience=70, nn_polish_steps=12, nn_restarts=3,
    ber_targets=[1e-4,5e-4,1e-3,3.8e-3,1e-2], required_cost_ratio=1.5,
    max_nn_validation_ber=3.8e-3, min_validation_frontier_gap=1e-4, polynomial_orders=[1,2,3,5,7],
    bit_polynomial_orders=[1,2,3,5], kernel_centers=[16,64,256],
)
NETWORKS = [
    ([4],"tanh"), ([6],"tanh"), ([8],"tanh"), ([16],"tanh"),
    ([24],"silu"), ([16,16],"tanh"), ([16],"softsign"),
    ([16],"multilevel"), ([16],"learned_multilevel"),
]
EXTENDED_NETWORKS = [
    ([7],"tanh"), ([8],"silu"), ([8],"softsign"), ([2,8],"tanh"),
    ([3,6],"tanh"), ([8],"learned_multilevel"),
]
EXTENDED_TRAINING = dict(epochs=900,patience=90,polish=20)


NN_LEARNING_RATE = .003
NN_WEIGHT_DECAY = 1e-4
NN_BATCH_SIZE = 512
NN_OPTIMIZER = "adamw"



def network_catalog():
    return [dict(widths=list(widths),activation=activation,lr=NN_LEARNING_RATE,weight_decay=NN_WEIGHT_DECAY,batch_size=NN_BATCH_SIZE,optimizer=NN_OPTIMIZER)
            for widths,activation in NETWORKS]

# Current A-RoF comparison: authoritative supplied Q8.24 training atomics.
# Selected only on validation MSE/EVM. Test labels never update weights.
ADAPTIVE = dict(
    epochs=8,
    lms_taps=[2,5,9],
    lms_learning_rates=[0.00025,0.001,0.004],
    ann_candidates=[
        dict(hidden=12,activation="sigmoid",learning_rate=0.004),
        dict(hidden=12,activation="sigmoid",learning_rate=0.016),
        dict(hidden=12,activation="tanh",learning_rate=0.001),
        dict(hidden=12,activation="tanh",learning_rate=0.004),
        dict(hidden=16,activation="tanh",learning_rate=0.001),
        dict(hidden=16,activation="tanh",learning_rate=0.004),
    ],
    initialization_seed=42,
)

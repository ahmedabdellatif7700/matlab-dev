import csv
import copy
import math
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from activations.variable_alpha_sigmoid import (
    VariableAlphaSigmoid,
    variable_alpha_sigmoid,
)
from activations.stable_alpha_sigmoid import (
    StableAlphaSigmoid,
    stable_alpha_sigmoid,
)

from activations.sigmoid import (
    Sigmoid,
    sigmoid,
)

from activations.activation import (
    ComplexMultilevelSigmoid,
    MultilevelSigmoid,
    complex_multilevel_sigmoid,
    multilevel_sigmoid,
)

from hardware.q8_8_inference_arithmetic import (
    ACC_BITS,
    dequantize_fixed,
    mac_q8_8,
    quantize_fixed,
    sigmoid_q8_8,
)


# Shared configuration remains available as equalizer.<SETTING> for the
# existing SGD vector exporter and any research scripts that import it.
from config import (
    HIDDEN_SIZES, EXPORT_FIXED_POINT, ROOT, DATASET_PATH, ONNX_FP32_OUTPUT_PATH, SOFTWARE_Q8_8_OUTPUT_PATH, ARTIFACT_DIR, MODEL_PATH,
    QUANT_PATH, INFERENCE_DIR, SOFTWARE_OUTPUT_PATH, TEST_INPUTS_MEM_PATH, PARAMETERS_MEM_PATH,
    MODE, ACTIVATION, SUPPORTED_ACTIVATIONS, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE,
    OPTIMIZER, LEARNING_RATE, BATCH_SIZE, SHUFFLE, EPOCHS, TRAIN_RATIO, SEED,
    ALPHA_INITIAL, STABLE_ALPHA, TOTAL_BITS, FRAC_BITS, FPGA_INFERENCE_SUPPORTED,
)

# Legacy complex/multilevel code is retained for future experiments, but is
# intentionally not selectable in the current unified real-valued workflow.
C1 = 6.0
LEVELS = 4

# ---------------------------------------------------------------------
# Neural equalizer
# ---------------------------------------------------------------------

class RealEqualizer(nn.Module):
    """Configurable real MLP, retaining legacy fc1/fc2 parameter names."""
    def __init__(self, input_size=None, hidden_sizes=None, output_size=None, activation=None):
        super().__init__()
        from activations import make_activation
        from config import HIDDEN_SIZES
        inputs = INPUT_SIZE if input_size is None else input_size
        outputs = OUTPUT_SIZE if output_size is None else output_size
        widths = tuple(HIDDEN_SIZES if hidden_sizes is None else hidden_sizes)
        name = ACTIVATION if activation is None else activation
        if not widths or any(w < 1 for w in widths):
            raise ValueError("hidden_sizes must contain positive widths")
        alpha = STABLE_ALPHA if name == 'stable_alpha' else ALPHA_INITIAL
        self.fc1 = nn.Linear(inputs, widths[0])
        self.activation = make_activation(name, alpha=alpha)
        self.hidden_layers = nn.ModuleList([
            nn.Sequential(nn.Linear(a,b),make_activation(name,alpha=alpha))
            for a,b in zip(widths[:-1],widths[1:])])
        self.fc2 = nn.Linear(widths[-1],outputs)

    def forward(self,x):
        x = self.activation(self.fc1(x))
        for layer in self.hidden_layers:
            x = layer(x)
        return self.fc2(x)


class ComplexLinear(nn.Module):

    def __init__(
        self,
        in_features,
        out_features,
    ):
        super().__init__()

        self.weight_real = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )

        self.weight_imag = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
            )
        )

        self.bias_real = nn.Parameter(
            torch.zeros(out_features)
        )

        self.bias_imag = nn.Parameter(
            torch.zeros(out_features)
        )

        nn.init.xavier_uniform_(
            self.weight_real
        )

        nn.init.xavier_uniform_(
            self.weight_imag
        )

    def forward(
        self,
        real,
        imag,
    ):

        output_real = (
            real @ self.weight_real.T
            - imag @ self.weight_imag.T
            + self.bias_real
        )

        output_imag = (
            real @ self.weight_imag.T
            + imag @ self.weight_real.T
            + self.bias_imag
        )

        return (
            output_real,
            output_imag,
        )


class ComplexEqualizer(nn.Module):

    def __init__(self):
        super().__init__()

        self.fc1 = ComplexLinear(
            2,
            HIDDEN_SIZE,
        )

        self.activation = ComplexMultilevelSigmoid(
            c1=C1,
            levels=LEVELS,
        )

        self.fc2 = ComplexLinear(
            HIDDEN_SIZE,
            1,
        )

        self.output_activation = ComplexMultilevelSigmoid(
            c1=C1,
            levels=LEVELS,
        )

    def forward(self, x):

        real = x[:, [0, 2]]
        imag = x[:, [1, 3]]

        real, imag = self.fc1(
            real,
            imag,
        )

        real, imag = self.activation(
            real,
            imag,
        )

        real, imag = self.fc2(
            real,
            imag,
        )

        real, imag = self.output_activation(
            real,
            imag,
        )

        return torch.cat(
            (real, imag),
            dim=1,
        )


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

def load_dataset():

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"dataset not found: {DATASET_PATH}"
        )

    data = scipy.io.loadmat(
        DATASET_PATH
    )

    x = data["X"].astype(
        np.float32
    )

    y = data["Y"].astype(
        np.float32
    )

    if x.shape[1] != INPUT_SIZE:
        raise ValueError(
            f"expected {INPUT_SIZE} inputs, "
            f"got {x.shape[1]}"
        )

    if y.shape[1] != OUTPUT_SIZE:
        raise ValueError(
            f"expected {OUTPUT_SIZE} outputs, "
            f"got {y.shape[1]}"
        )

    return x, y


# ---------------------------------------------------------------------
# Training (shared by MATLAB workflow and RTL artifact exporter)
# ---------------------------------------------------------------------

def seed_training():
    """Use the original random initialization and DataLoader seed."""
    np.random.seed(SEED)
    torch.manual_seed(SEED)


def split_dataset(x, y):
    """Preserve the original sequential 80/20 train/test split."""
    split = int(TRAIN_RATIO * len(x))
    return x[:split], y[:split], x[split:], y[split:], split


def validate_configuration():
    """Validate software choices and the optional fixed-point export contract."""
    if MODE != "real":
        raise ValueError("Use MODE='real' for the joint I/Q software equalizer")
    if ACTIVATION not in SUPPORTED_ACTIVATIONS:
        raise ValueError(f"Unsupported activation {ACTIVATION!r}; choose from {SUPPORTED_ACTIVATIONS}")
    if EXPORT_FIXED_POINT and (INPUT_SIZE, tuple(HIDDEN_SIZES), OUTPUT_SIZE) != (4, (12,), 2):
        raise ValueError("Fixed-point export requires 4-12-2; set EXPORT_FIXED_POINT=False for arbitrary software networks")
    if EXPORT_FIXED_POINT and ACTIVATION not in ("sigmoid","variable_alpha","stable_alpha"):
        raise ValueError("This activation requires EXPORT_FIXED_POINT=False")
    if OUTPUT_SIZE != 2:
        raise ValueError("The legacy MAT interface has I/Q outputs; use research stages for Gray-bit heads")
    if OPTIMIZER not in ("sgd", "adam"):
        raise ValueError("OPTIMIZER must be 'sgd' or 'adam'")
    if LEARNING_RATE <= 0 or BATCH_SIZE < 1 or EPOCHS < 1:
        raise ValueError("LEARNING_RATE, BATCH_SIZE and EPOCHS must be positive")
    if not (0 < TRAIN_RATIO < 1):
        raise ValueError("TRAIN_RATIO must be between 0 and 1")
    if not isinstance(SHUFFLE, bool):
        raise ValueError("SHUFFLE must be a boolean")
    if (TOTAL_BITS, FRAC_BITS) != (16, 8):
        raise ValueError("This FPGA inference implementation requires Q8.8 (16 total, 8 fractional bits)")


def create_model(device):
    """Build the configured software model using the shared activation registry."""
    return RealEqualizer().to(device)


def create_optimizer(model):
    """Use one optimizer and learning rate for every activation configuration."""
    if OPTIMIZER == "sgd":
        return torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
    if OPTIMIZER == "adam":
        return torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}")


def train(model, x_train, y_train, device, on_step=None):
    """Train using the original PyTorch operations.

    Optional on_step(stage=..., epoch=..., batch=..., model=..., optimizer=...,
                     inputs=..., targets=..., predictions=..., loss=...)
    is called immediately before and after optimizer.step(). Epoch and batch
    indices are 1-based. Returning True from the 'after_update' callback
    stops training after that complete optimizer update. Normal MATLAB runs
    pass no callback and retain their original behavior.
    """
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    optimizer = create_optimizer(model)
    loss_fn = nn.MSELoss()
    for epoch in range(EPOCHS):

        model.train()
        loss_sum = 0.0

        for batch_number, (x_batch, y_batch) in enumerate(loader, start=1):
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            prediction = model(x_batch)
            loss = loss_fn(prediction, y_batch)
            loss.backward()

            if on_step is not None:
                on_step(stage="before_update", epoch=epoch + 1,
                        batch=batch_number, model=model, optimizer=optimizer,
                        inputs=x_batch, targets=y_batch,
                        predictions=prediction, loss=loss)

            optimizer.step()

            if on_step is not None:
                stop = on_step(stage="after_update", epoch=epoch + 1,
                               batch=batch_number, model=model, optimizer=optimizer,
                               inputs=x_batch, targets=y_batch,
                               predictions=prediction, loss=loss)
                if stop:
                    return

            loss_sum += loss.item() * x_batch.size(0)

        average_loss = loss_sum / len(dataset)
        epoch_message = f"epoch {epoch + 1:3d}/{EPOCHS} loss={average_loss:.8f}"
        if ACTIVATION == "variable_alpha":
            epoch_message += f" alpha={model.activation.alpha.detach().item():.8f}"
        print(epoch_message)


# ---------------------------------------------------------------------
# ONNX export
# ---------------------------------------------------------------------

def export_onnx(
    model,
    device,
):

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model.eval()

    sample = torch.zeros(
        1,
        INPUT_SIZE,
        dtype=torch.float32,
        device=device,
    )

    # ONNX is required for an ordinary full inference/export run, but not
    # when importing equalizer.py for the independent SGD vector exporter.
    torch.onnx.export(
        model,
        sample,
        MODEL_PATH,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch"},
            "output": {0: "batch"},
        },
        opset_version=18,
        dynamo=False,
    )

    print(
        f"model: {MODEL_PATH}"
    )


# ---------------------------------------------------------------------
# Floating-point inference
# ---------------------------------------------------------------------

def floating_inference(x):
    """Evaluate the original FP32 ONNX model without Q8.8 quantization."""
    import onnxruntime as ort

    session = ort.InferenceSession(
        str(MODEL_PATH),
        providers=["CPUExecutionProvider"],
    )
    model_input = session.get_inputs()[0]
    model_output = session.get_outputs()[0]
    if model_input.type != "tensor(float)" or model_output.type != "tensor(float)":
        raise TypeError(
            "ONNX reference must have float32 input/output; "
            f"got {model_input.type} -> {model_output.type}"
        )

    # The model was exported from the trained FP32 PyTorch network before
    # quantize_model() is called. Do not use parameters.mem in this path.
    inputs = np.ascontiguousarray(x, dtype=np.float32)
    prediction = np.asarray(
        session.run(None, {model_input.name: inputs})[0]
    )
    if prediction.dtype != np.float32 or prediction.shape != (len(inputs), OUTPUT_SIZE):
        raise ValueError(
            f"Invalid ONNX result: dtype={prediction.dtype}, shape={prediction.shape}"
        )
    if not np.all(np.isfinite(prediction)):
        raise ValueError("ONNX FP32 inference returned non-finite values")

    print("ONNX inference: FP32 model, FP32 inputs, FP32 output (no Q8.8 arithmetic)")
    return prediction


# ---------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------

def quantize(values):

    scale = 1 << FRAC_BITS

    minimum = -(
        1 << (TOTAL_BITS - 1)
    )

    maximum = (
        1 << (TOTAL_BITS - 1)
    ) - 1

    fixed = np.rint(
        values * scale
    ).astype(np.int64)

    return np.clip(
        fixed,
        minimum,
        maximum,
    )


def dequantize(values):

    return (
        values.astype(np.float32)
        / float(1 << FRAC_BITS)
    )


def get_parameter(
    model,
    name,
):

    from onnx import numpy_helper

    for parameter in model.graph.initializer:

        if parameter.name == name:

            return numpy_helper.to_array(
                parameter
            )

    raise KeyError(
        f"parameter not found: {name}"
    )


def quantize_model(model):

    if MODE == "real":

        parameters = {
            "fc1_weights": quantize(
                model.fc1.weight.detach().cpu().numpy()
            ),
            "fc1_biases": quantize(
                model.fc1.bias.detach().cpu().numpy()
            ),
            "fc2_weights": quantize(
                model.fc2.weight.detach().cpu().numpy()
            ),
            "fc2_biases": quantize(
                model.fc2.bias.detach().cpu().numpy()
            ),
        }

        if ACTIVATION == "variable_alpha":
            parameters["alpha"] = quantize(
                model.activation.alpha.detach().cpu().numpy()
            )
        elif ACTIVATION == "stable_alpha":
            # Store the fixed coefficient so the software verifier can
            # reproduce this run without relying on a hidden default.
            parameters["alpha"] = quantize(
                np.asarray(model.activation.alpha, dtype=np.float32)
            )

    else:

        parameters = {
            "fc1_weights_real": quantize(
                model.fc1.weight_real.detach().cpu().numpy()
            ),
            "fc1_weights_imag": quantize(
                model.fc1.weight_imag.detach().cpu().numpy()
            ),
            "fc1_biases_real": quantize(
                model.fc1.bias_real.detach().cpu().numpy()
            ),
            "fc1_biases_imag": quantize(
                model.fc1.bias_imag.detach().cpu().numpy()
            ),
            "fc2_weights_real": quantize(
                model.fc2.weight_real.detach().cpu().numpy()
            ),
            "fc2_weights_imag": quantize(
                model.fc2.weight_imag.detach().cpu().numpy()
            ),
            "fc2_biases_real": quantize(
                model.fc2.bias_real.detach().cpu().numpy()
            ),
            "fc2_biases_imag": quantize(
                model.fc2.bias_imag.detach().cpu().numpy()
            ),
        }

    QUANT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez(
        QUANT_PATH,
        **parameters,
        total_bits=TOTAL_BITS,
        frac_bits=FRAC_BITS,
    )

    print(
        f"quantized: {QUANT_PATH}"
    )

    return parameters


# ---------------------------------------------------------------------
# Quantized reference inference
# ---------------------------------------------------------------------

def quantized_inference(
    x,
    parameters,
):

    if MODE == "real":

        w1 = dequantize(
            parameters["fc1_weights"]
        )

        b1 = dequantize(
            parameters["fc1_biases"]
        )

        w2 = dequantize(
            parameters["fc2_weights"]
        )

        b2 = dequantize(
            parameters["fc2_biases"]
        )

        hidden = (
            x @ w1.T
            + b1
        )

        if ACTIVATION == "variable_alpha":
            alpha = dequantize(
                parameters["alpha"]
            )

            hidden = variable_alpha_sigmoid(
                hidden,
                alpha=float(alpha),
            )
        elif ACTIVATION == "sigmoid":
            hidden = sigmoid(hidden)
        else:
            hidden = stable_alpha_sigmoid(
                hidden,
                alpha=float(dequantize(parameters["alpha"])),
            )

        output = (
            hidden @ w2.T
            + b2
        )

        return output.astype(
            np.float32
        )

    w1 = (
        dequantize(
            parameters["fc1_weights_real"]
        )
        + 1j
        * dequantize(
            parameters["fc1_weights_imag"]
        )
    )

    b1 = (
        dequantize(
            parameters["fc1_biases_real"]
        )
        + 1j
        * dequantize(
            parameters["fc1_biases_imag"]
        )
    )

    w2 = (
        dequantize(
            parameters["fc2_weights_real"]
        )
        + 1j
        * dequantize(
            parameters["fc2_weights_imag"]
        )
    )

    b2 = (
        dequantize(
            parameters["fc2_biases_real"]
        )
        + 1j
        * dequantize(
            parameters["fc2_biases_imag"]
        )
    )

    complex_input = (
        x[:, [0, 2]]
        + 1j * x[:, [1, 3]]
    )

    hidden = (
        complex_input @ w1.T
        + b1
    )

    hidden = complex_multilevel_sigmoid(
        hidden,
        c1=C1,
        levels=LEVELS,
    )

    output = (
        hidden @ w2.T
        + b2
    )

    output = complex_multilevel_sigmoid(
        output,
        c1=C1,
        levels=LEVELS,
    )

    return np.column_stack(
        (
            output.real[:, 0],
            output.imag[:, 0],
        )
    ).astype(np.float32)


# ---------------------------------------------------------------------
# Q8.8 software inference (RTL bit-exact only for standard sigmoid)
# ---------------------------------------------------------------------

def fixed_point_inference(inputs_q8_8, parameters):
    """Infer from *already quantized* signed Q8.8 words loaded from .mem.

    Standard sigmoid reproduces verified FPGA arithmetic. Alpha-based
    activations remain software-only with their alpha in the NPZ model.
    """
    if MODE != "real":
        raise ValueError("Q8.8 software inference requires real mode")

    x_fixed = np.asarray(inputs_q8_8, dtype=np.int64)
    if x_fixed.ndim != 2 or x_fixed.shape[1] != INPUT_SIZE:
        raise ValueError(f"Expected Q8.8 inputs (N, {INPUT_SIZE})")
    if np.any((x_fixed < -32768) | (x_fixed > 32767)):
        raise ValueError("Input words are outside signed 16-bit Q8.8")
    hidden = mac_q8_8(
        x_fixed, parameters["fc1_weights"], parameters["fc1_biases"]
    )

    if ACTIVATION == "sigmoid":
        hidden = sigmoid_q8_8(hidden)
    elif ACTIVATION in ("variable_alpha", "stable_alpha"):
        alpha = float(np.asarray(dequantize(parameters["alpha"])).item())
        hidden_float = dequantize_fixed(hidden)
        if ACTIVATION == "variable_alpha":
            hidden_float = variable_alpha_sigmoid(hidden_float, alpha=alpha)
        else:
            hidden_float = stable_alpha_sigmoid(hidden_float, alpha=alpha)
        hidden = quantize_fixed(hidden_float)
    else:
        raise ValueError(f"Unsupported Q8.8 software activation: {ACTIVATION}")

    output = mac_q8_8(
        hidden, parameters["fc2_weights"], parameters["fc2_biases"]
    )
    return dequantize_fixed(output).astype(np.float32)


# ---------------------------------------------------------------------
# RTL export
# ---------------------------------------------------------------------

def write_mem(
    path,
    values,
):

    mask = (
        1 << TOTAL_BITS
    ) - 1

    width = (
        TOTAL_BITS + 3
    ) // 4

    with path.open("w") as file:

        for value in values.reshape(-1):

            file.write(
                f"{int(value) & mask:0{width}X}\n"
            )


def export_parameter_memory(parameters):
    """Export 86 network weights/biases in the fixed 128-word Q8.8 layout.

    The alpha coefficient of custom activations is NOT represented here;
    it is retained in equalizer_q8_8.npz for software inference.
    """
    layout = (
        ("fc1_weights", 48),
        ("fc1_biases", 12),
        ("fc2_weights", 24),
        ("fc2_biases", 2),
    )
    words = []
    for name, count in layout:
        values = np.asarray(parameters[name]).reshape(-1)
        if values.size != count:
            raise ValueError(f"{name}: expected {count} values, got {values.size}")
        words.extend(values.tolist())

    if len(words) != 86:
        raise ValueError(f"expected 86 parameters, got {len(words)}")

    # Reserved addresses are defined as zero for repeatable initialization.
    words.extend([0] * (128 - len(words)))
    INFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    path = INFERENCE_DIR / "parameters.mem"
    write_mem(path, np.asarray(words, dtype=np.int64))
    print(f"rtl: {path} values=128 (86 parameters + 42 reserved)")


def read_q8_8_memory(path, expected_words):
    """Read four-digit, two's-complement signed Q8.8 hex memory words."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing .mem input: {path} (generate it in MATLAB first)")

    words = []
    for index, raw in enumerate(path.read_text().splitlines(), start=1):
        token = raw.strip()
        if len(token) != 4 or any(ch not in "0123456789abcdefABCDEF" for ch in token):
            raise ValueError(f"Invalid 16-bit hex word in {path}, line {index}: {token!r}")
        value = int(token, 16)
        words.append(value - 65536 if value >= 32768 else value)

    if len(words) != expected_words:
        raise ValueError(f"{path}: expected {expected_words} words, found {len(words)}")
    return np.asarray(words, dtype=np.int64)


def load_test_inputs(expected_samples):
    """Consume MATLAB's row-major Q8.8 test vector memory unchanged."""
    words = read_q8_8_memory(TEST_INPUTS_MEM_PATH, expected_samples * INPUT_SIZE)
    inputs = words.reshape(expected_samples, INPUT_SIZE)
    print(f"Q8.8 inputs loaded from MATLAB: {TEST_INPUTS_MEM_PATH} ({expected_samples} samples)")
    return inputs


def load_parameter_memory(quantized_parameters):
    """Consume the exact 86 Q8.8 parameters that the KV260 C++ loads."""
    words = read_q8_8_memory(PARAMETERS_MEM_PATH, 128)
    if np.any(words[86:] != 0):
        raise ValueError("Nonzero reserved words in parameters.mem")

    layout = (("fc1_weights", (HIDDEN_SIZE, INPUT_SIZE)),
              ("fc1_biases", (HIDDEN_SIZE,)),
              ("fc2_weights", (OUTPUT_SIZE, HIDDEN_SIZE)),
              ("fc2_biases", (OUTPUT_SIZE,)))
    parameters = {}
    index = 0
    for name, shape in layout:
        count = int(np.prod(shape))
        parameters[name] = words[index:index + count].reshape(shape)
        if not np.array_equal(parameters[name], np.asarray(quantized_parameters[name])):
            raise ValueError(f"parameters.mem does not match the quantized model: {name}")
        index += count
    # Optional custom activation coefficient is not present in RTL memory.
    if "alpha" in quantized_parameters:
        parameters["alpha"] = quantized_parameters["alpha"]
    print(f"Q8.8 parameters loaded from: {PARAMETERS_MEM_PATH} (86 active words)")
    return parameters


def export_python_outputs(fixed_point):
    """Save Q8.8 software predictions in a consistent CSV format."""
    if fixed_point.ndim != 2 or fixed_point.shape[1] != OUTPUT_SIZE:
        raise ValueError(f"expected outputs with shape (N, {OUTPUT_SIZE})")

    path = SOFTWARE_OUTPUT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    outputs_q8 = np.rint(fixed_point.astype(np.float64) * (1 << FRAC_BITS)).astype(np.int64)

    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sample", "output_0_q8", "output_1_q8", "output_0", "output_1"))
        for i, (q0, q1) in enumerate(outputs_q8):
            writer.writerow((i, int(q0), int(q1),
                             f"{q0 / 256:.8f}", f"{q1 / 256:.8f}"))

    print(f"output: {path} samples={len(fixed_point)}")


# ---------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------

def evaluate(
    name,
    prediction,
    target,
):

    error = (
        prediction
        - target
    )

    mae = np.mean(
        np.abs(error)
    )

    rmse = np.sqrt(
        np.mean(
            error ** 2
        )
    )

    max_error = np.max(
        np.abs(error)
    )

    print()
    print(name)
    print("-" * len(name))

    print(
        f"MAE       : {mae:.8f}"
    )

    print(
        f"RMSE      : {rmse:.8f}"
    )

    print(
        f"Max error : {max_error:.8f}"
    )


# ---------------------------------------------------------------------
# MATLAB output
# ---------------------------------------------------------------------

def save_output(
    x_test,
    y_test,
    floating,
    quantized,
    fixed_point,
    test_start,
):
    """Write two separate MAT files for ONNX FP32 and software Q8.8.

    The mixed-precision ``quantized`` predictions remain available for
    console evaluation but do not generate a third Python MAT file.
    Hardware results are exported separately by the C++ KV260 program.
    """
    if fixed_point is None:
        raise ValueError("Q8.8 software predictions are required for MAT export")

    sample_count = len(x_test)
    for name, data in (
        ("targets", y_test),
        ("ONNX FP32", floating),
        ("software Q8.8", fixed_point),
    ):
        if data.shape != (sample_count, OUTPUT_SIZE):
            raise ValueError(f"{name} shape must be {(sample_count, OUTPUT_SIZE)}")
        if not np.all(np.isfinite(data)):
            raise ValueError(f"{name} contains non-finite values")

    rx_before = x_test[:, 0] + 1j * x_test[:, 1]
    tx_reference = y_test[:, 0] + 1j * y_test[:, 1]
    test_index = np.array([[test_start]], dtype=np.int32)

    # The same reference samples and zero-based test split are saved in both
    # files, allowing MATLAB to validate alignment with the C++ FPGA output.
    common_data = {
        "rx_before": rx_before,
        "tx_reference": tx_reference,
        "test_start": test_index,
    }

    onnx_output = floating[:, 0] + 1j * floating[:, 1]
    software_output = fixed_point[:, 0] + 1j * fixed_point[:, 1]

    # Q8.8 values are represented as complex floats in the MAT file,
    # without changing the fixed-point arithmetic that produced them.
    scaled = fixed_point.astype(np.float64) * (1 << FRAC_BITS)
    if not np.all(scaled == np.rint(scaled)):
        raise ValueError("Software Q8.8 results are not on the Q8.8 grid")

    for path, key, result in (
        (ONNX_FP32_OUTPUT_PATH, "rx_equalized_onnx_fp32", onnx_output),
        (SOFTWARE_Q8_8_OUTPUT_PATH, "rx_equalized_software_q8_8", software_output),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        scipy.io.savemat(path, {**common_data, key: result}, do_compression=True)
        print(f"MATLAB output ({key}): {path}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def legacy_main():

    validate_configuration()
    seed_training()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    print(f"mode: {MODE}")
    print(f"network: {[INPUT_SIZE,*HIDDEN_SIZES,OUTPUT_SIZE]}")
    print(f"activation: {ACTIVATION}")
    print(f"optimizer: {OPTIMIZER.upper()}, learning_rate={LEARNING_RATE}")
    print(f"batch_size: {BATCH_SIZE}, shuffle={SHUFFLE}, epochs={EPOCHS}")
    print(
        f"quantization: "
        f"Q{TOTAL_BITS - FRAC_BITS}.{FRAC_BITS}"
    )

    x, y = load_dataset()

    x_train, y_train, x_test, y_test, split = split_dataset(x, y)

    print(
        f"dataset : {x.shape}"
    )

    print(
        f"training: {x_train.shape}"
    )

    print(
        f"testing : {x_test.shape}"
    )

    model = create_model(device)

    initial_alpha = None
    if ACTIVATION == "variable_alpha":
        initial_alpha = model.activation.alpha.detach().item()
        print(f"trainable alpha initial: {initial_alpha:.8f}")
    elif ACTIVATION == "stable_alpha":
        print(f"fixed alpha: {model.activation.alpha:.8f}")

    train(
        model,
        x_train,
        y_train,
        device,
    )

    if ACTIVATION == "variable_alpha":
        final_alpha = model.activation.alpha.detach().item()
        print(f"trainable alpha final  : {final_alpha:.8f}")
        print(f"trainable alpha change : {final_alpha - initial_alpha:+.8f}")

    export_onnx(
        model,
        device,
    )

    floating = floating_inference(
        x_test
    )

    evaluate(
        "Floating-point equalizer",
        floating,
        y_test,
    )

    if not EXPORT_FIXED_POINT:
        rx_before = x_test[:,0]+1j*x_test[:,1]
        scipy.io.savemat(ONNX_FP32_OUTPUT_PATH,dict(
            rx_before=rx_before, tx_reference=y_test[:,0]+1j*y_test[:,1],
            rx_equalized_onnx_fp32=floating[:,0]+1j*floating[:,1],
            test_start=np.array([[split]],dtype=np.int32)))
        return

    parameters = quantize_model(model)

    quantized = quantized_inference(
        x_test,
        parameters,
    )

    evaluate(
        "Quantized equalizer",
        quantized,
        y_test,
    )

    evaluate(
        "Floating vs quantized",
        quantized,
        floating,
    )

    # MATLAB is the sole producer of test_inputs.mem. Both Python and C++
    # use its signed Q8.8 words directly for fixed-point inference.
    # Only standard sigmoid supports bit-exact FPGA comparison.
    export_parameter_memory(parameters)
    q8_inputs = load_test_inputs(len(x_test))
    q8_parameters = load_parameter_memory(parameters)
    fixed_point = fixed_point_inference(q8_inputs, q8_parameters)
    evaluate("Q8.8 software equalizer", fixed_point, y_test)
    evaluate("Quantized vs Q8.8 software", fixed_point, quantized)
    evaluate("ONNX FP32 vs Q8.8 software", floating, fixed_point)

    # The fixed-point implementation must produce exact multiples of 1/256;
    # storing those values in float32 does not change their provenance.
    q8_scaled = fixed_point.astype(np.float64) * (1 << FRAC_BITS)
    if not np.all(q8_scaled == np.rint(q8_scaled)):
        raise ValueError("Software Q8.8 predictions are not on the Q8.8 grid")
    different_samples = int(np.count_nonzero(np.any(floating != fixed_point, axis=1)))
    print(f"ONNX FP32 vs software Q8.8: {different_samples}/{len(x_test)} "
          "samples numerically different")
    print(f"\nQ8.8 software: accumulator={ACC_BITS} bits, activation={ACTIVATION}")
    if not FPGA_INFERENCE_SUPPORTED:
        print("NOTE: The current FPGA RTL supports standard sigmoid only.")
        print("      Alpha-scaled activation outputs are Python Q8.8 estimates,")
        print("      not bit-exact predictions of the existing FPGA RTL.")
        print("      parameters.mem holds 86 weights/biases; the alpha")
        print("      coefficient is saved in equalizer_q8_8.npz, not in RTL memory.")
    export_python_outputs(fixed_point)

    save_output(
        x_test,
        y_test,
        floating,
        quantized,
        fixed_point,
        split,
    )


def train_bit_equalizer(x, bits, vx, vbits, spec, *, seed=42, epochs=600, patience=60, polish=10):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    mean = x.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = np.maximum(x.std(axis=0, dtype=np.float64), 1e-6).astype(np.float32)
    xt = torch.from_numpy(np.asarray((x-mean)/scale, dtype=np.float32))
    vt = torch.from_numpy(np.asarray((vx-mean)/scale, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(bits, dtype=np.float32))
    vy = torch.from_numpy(np.asarray(vbits, dtype=np.float32))
    net = RealEqualizer(x.shape[1], spec['widths'], bits.shape[1], spec['activation'])
    optimizer_name=spec.get('optimizer','adamw')
    optimizers=dict(adamw=torch.optim.AdamW,adam=torch.optim.Adam,sgd=torch.optim.SGD)
    if optimizer_name not in optimizers:
        raise ValueError(f'Unsupported research optimizer: {optimizer_name}')
    optimizer=optimizers[optimizer_name](net.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    best, state, best_step, history = math.inf, None, 0, []

    def capture(step, loss, phase):
        nonlocal best, state, best_step
        with torch.no_grad():
            logits = net(vt)
            val = nn.functional.binary_cross_entropy_with_logits(logits, vy).item()
            ber = (logits.ge(0) != vy.bool()).float().mean().item()
        history.append([step, loss, val, ber, phase])
        if val < best:
            best, state, best_step = val, copy.deepcopy(net.state_dict()), step

    for epoch in range(1, epochs+1):
        total = 0.
        for batch in torch.randperm(len(xt)).split(spec.get('batch_size',512)):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt[batch]), yt[batch])
            loss.backward(); optimizer.step()
            total += loss.item()*len(batch)
        capture(epoch, total/len(xt), 0)
        if epoch-best_step >= patience:
            break
    adam_steps = len(history)
    net.load_state_dict(state)
    lbfgs = torch.optim.LBFGS(net.parameters(), lr=.5, max_iter=10, line_search_fn='strong_wolfe')
    for step in range(polish):
        def closure():
            lbfgs.zero_grad(set_to_none=True)
            loss = nn.functional.binary_cross_entropy_with_logits(net(xt), yt)
            loss.backward()
            return loss
        loss = lbfgs.step(closure)
        capture(adam_steps+step+1, loss.item(), 1)
    net.load_state_dict(state); net.eval()
    dims = [x.shape[1], *spec['widths'], bits.shape[1]]
    hidden = sum(spec['widths'])
    nonlinear = hidden*(3 if 'multilevel' in spec['activation'] else 1)
    dense = sum(a*b for a,b in zip(dims[:-1], dims[1:]))
    extra = 4*hidden if 'multilevel' in spec['activation'] else hidden if spec['activation']=='silu' else 0
    info = dict(**spec, input_size=x.shape[1], output_bits=bits.shape[1],
                parameters=sum(p.numel() for p in net.parameters()),
                multiplications=dense+extra, dense_multiplications=dense,
                activation_multiplications=extra,
                nonlinear_evaluations=nonlinear, comparisons=bits.shape[1],
                best_step=best_step, validation_bce=best, epochs_completed=adam_steps,
                polish_steps=polish, seed=seed)
    return dict(net=net, mean=mean, scale=scale, info=info, history=np.asarray(history))


def predict_bits(model, x):
    with torch.no_grad():
        value = torch.from_numpy(np.asarray((x-model['mean'])/model['scale'], dtype=np.float32))
        return model['net'](value).numpy()


def save_checkpoint(model, path):
    torch.save(dict(state_dict=model['net'].state_dict(), mean=torch.from_numpy(model['mean']),
                    scale=torch.from_numpy(model['scale']), info=model['info']), path)


def load_checkpoint(path):
    """Restore a locally generated float checkpoint for frozen inference."""
    torch.set_num_threads(1)
    saved=torch.load(path,map_location='cpu',weights_only=True)
    info=saved['info']
    net=RealEqualizer(info['input_size'],info['widths'],info['output_bits'],info['activation'])
    state=saved['state_dict']
    if any(key.startswith('layers.') for key in state):
        converted={}
        for key,value in state.items():
            _,number,suffix=key.split('.',2); index=int(number); block=index//2
            if block==len(info['widths']): prefix='fc2'
            elif block==0: prefix='fc1' if index%2==0 else 'activation'
            else: prefix=f'hidden_layers.{block-1}.{index%2}'
            converted[prefix+'.'+suffix]=value
        state=converted
    net.load_state_dict(state); net.eval()
    return dict(net=net,mean=saved['mean'].numpy(),scale=saved['scale'].numpy(),info=info)


def main():
    """Preserve no-argument legacy launch; research uses this same entry point."""
    import argparse
    from config import WORKFLOW, RESEARCH_RUN_DIR, RESEARCH_PHASE
    parser=argparse.ArgumentParser(description="MATLAB/Python neural equalizer")
    parser.add_argument('--workflow',choices=['legacy','research','adaptive'],default=WORKFLOW)
    parser.add_argument('--run',type=Path,default=RESEARCH_RUN_DIR)
    parser.add_argument('--phase',choices=['discovery','extended-search','confirmation','report','snr-sweep'],default=RESEARCH_PHASE)
    parser.add_argument('--only',nargs='+')
    parser.add_argument('--preset',choices=['baseline','tuned'],default='baseline')
    parser.add_argument('--baseline-run',type=Path)
    args=parser.parse_args()
    if args.workflow=='adaptive':
        if args.run is None:
            parser.error('Pass --run with the MATLAB comparison export directory')
        from adaptive_comparison import compare, combine_reports
        if args.phase=='report':
            combine_reports(args.run)
            if args.baseline_run:
                from adaptive_comparison import compare_previous
                compare_previous(args.run,args.baseline_run)
        else:
            from config import ADAPTIVE, ADAPTIVE_TUNED
            compare(args.run,args.only,ADAPTIVE_TUNED if args.preset=='tuned' else ADAPTIVE)
        return
    if args.workflow=='legacy' and args.run is None:
        legacy_main()
        return
    if args.run is None:
        parser.error('Set RESEARCH_RUN_DIR in config.py or pass --run')
    if args.phase=='discovery':
        from tuning.pipeline import design
        design(args.run,args.only)
    elif args.phase=='extended-search':
        from tuning.extended_search import search
        search(args.run)
    elif args.phase=='confirmation':
        from tuning.confirmation import confirm
        confirm(args.run,args.only)
    elif args.phase=='snr-sweep':
        from tuning.snr_sweep import evaluate
        evaluate(args.run)
    else:
        from tuning.report import report
        report(args.run,args.run/'report')


if __name__ == '__main__':
    main()

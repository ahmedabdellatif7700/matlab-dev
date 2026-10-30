"""TrainingEngine: one training step in the order of main_controller + layer loops.

forward: layers 0..L-1 up | loss | backward: layers L-1..1 down | update: layers 0..L-1 up
RTL: main_controller and the forward/loss/backward/update controllers + layer_loop_fsm.
The activation is injected, so the engine does not depend on the activation function.
"""
from .backward_atomic import backward_atomic
from .forward_atomic import forward_atomic
from .loss_atomic import loss_atomic
from . import memory_map as mm
from .update_atomic import LR_Q24, update_atomic

PHASES = ("fwd", "loss", "bwd", "upd")


class TrainingEngine:

    def __init__(self, activation, lr_q24=LR_Q24):
        self.act = activation
        self.lr_q24 = lr_q24

    def init_ram(self, ram):
        """Load the activation's initial parameters into RAM[mm.ACT_P ...]."""
        for k, word in enumerate(self.act.init_params()):
            ram[mm.ACT_P + k] = word

    def run_phase(self, ram, phase):
        layers = len(mm.LAYERS)
        if phase == "fwd":
            for layer in range(layers):
                forward_atomic(ram, layer, self.act)
        elif phase == "loss":
            loss_atomic(ram)
        elif phase == "bwd":
            for layer in range(layers - 1, 0, -1):
                backward_atomic(ram, layer, self.act)
        elif phase == "upd":
            for layer in range(layers):
                update_atomic(ram, layer, self.act, self.lr_q24)
        else:
            raise ValueError(phase)

"""Neural-network activation functions."""


def make_activation(name, *, alpha=4.0):
    """One activation registry for software research and legacy models."""
    from torch import nn
    from .sigmoid import Sigmoid
    from .variable_alpha_sigmoid import VariableAlphaSigmoid
    from .stable_alpha_sigmoid import StableAlphaSigmoid
    from .multilevel import Multilevel
    factories = dict(sigmoid=Sigmoid, variable_alpha=lambda: VariableAlphaSigmoid(alpha=alpha),
                     stable_alpha=lambda: StableAlphaSigmoid(alpha=alpha), tanh=nn.Tanh,
                     silu=nn.SiLU, softsign=nn.Softsign, multilevel=lambda: Multilevel(False),
                     learned_multilevel=lambda: Multilevel(True))
    if name not in factories:
        raise ValueError(f"Unknown activation {name!r}; choose from {tuple(factories)}")
    return factories[name]()

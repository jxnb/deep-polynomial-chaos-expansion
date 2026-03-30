import numpy as np
import torch
from torch import nn
from collections.abc import Callable
from functools import partial


class MLP(nn.Module):
    def __init__(
        self,
        input_shape: tuple,
        n_outputs: int,
        n_hidden_layers: int | None = None,
        n_units_per_layer: int | None = None,
        batch_norm: bool = True,
        activation: Callable = torch.nn.ReLU,
        dtype=torch.float32,
    ):
        super().__init__()
        n_hidden_units = [n_units_per_layer] * n_hidden_layers

        n_inputs = np.prod(input_shape).item()
        layer_sizes = [n_inputs, *[n for n in n_hidden_units], n_outputs]
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1], dtype=dtype))
            if i <= (len(layer_sizes) - 3):
                layers.append(activation())
                if batch_norm:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
        self.layers = nn.ModuleList(layers)
        self.activation = activation if not isinstance(activation, partial) else activation.func

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)
        return out

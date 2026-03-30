"""Experiment configurations"""

from pydantic import BaseModel
import torch
from pathlib import Path
from collections.abc import Callable, Iterable
from functools import partial
from tensorchaos.circuits import build_tensorized_circuit

from src.baselines.pce import build_pce
from src.baselines.cnn import UNet
from src.baselines.mlp import MLP
from src.datasets.pde import load_darcyflow_dataset, load_steadystatediffusion_dataset
from src.datasets.test_functions import BratleySumFunc, SobolGStarFunc, XDBenchmarkFunc


# ==================================== Config templates ====================================

# === MODELS ===

class CircuitConfig(BaseModel):
    n_sums: int
    scope_size: int
    input_layer: str
    product_layer: str
    region_graph: str
    max_order: int
    batch_norm: bool
    param_init: Callable | dict | None
    pce_var_decay: float | None
    pce_var_rank: str | None

    def build_factory(self):
        return partial(build_tensorized_circuit, **self.model_dump())


class MLPConfig(BaseModel):
    n_hidden_layers: int | None
    n_units_per_layer: int | None
    batch_norm: bool
    activation: Callable

    def build_factory(self):
        return partial(MLP, **self.model_dump())


class UNetConfig(BaseModel):
    n_channels_in: int
    n_channels_out: int
    n_channels_max: int
    n_channels_first_step: int
    batch_norm: bool
    activation: Callable

    def build_factory(self):
        return partial(UNet, **self.model_dump())


class TorchPCEConfig(BaseModel):
    expansion: str
    max_order: int
    truncation: str
    truncation_args: dict | None
    index_set: None = None
    method: str
    eps: float = 0.0
    max_active_dims: int | None = None
    orthonormal: bool = True

    def build_factory(self):
        return partial(build_pce, **self.model_dump())


class OptimizerConfig(BaseModel):
    lr: float
    amsgrad: bool

    def build_factory(self):
        return partial(torch.optim.Adam, **self.model_dump())


# === DATASETS ===

class DarcyFlowConfig(BaseModel):
    x_shape: tuple = (64, 64)
    y_shape: tuple = (64, 64)
    n_val_samples: int = 2000
    data_dir: str | Path = "/data/jxnb/deepchaos/datasets/pde/cnn-surrogate/dataset"  # <-- set path to data here!
    random_subsets: bool = True

    def build_factory(self):
        return partial(load_darcyflow_dataset, **self.model_dump())


class SteadyStateDiffusionConfig(BaseModel):
    n_val_samples: int = 2000
    data_dir: str | Path = "./"  # <-- set path to data here!
    random_subsets: bool = True

    def build_factory(self):
        return partial(load_steadystatediffusion_dataset, **self.model_dump())


class SobolGStarConfig(BaseModel):
    n_dims: int = 100
    a: Iterable | None = None
    alpha: float = 0.5
    delta: Iterable | None = None

    n_train: int
    n_val: int
    n_test: int

    def build(self):
        return SobolGStarFunc(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k not in ["n_train", "n_val", "n_test"]
            }
        )


class BratleySumConfig(BaseModel):
    n_dims: int

    n_train: int
    n_val: int
    n_test: int

    def build(self):
        return BratleySumFunc(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k not in ["n_train", "n_val", "n_test"]
            }
        )


class XDBenchmarkConfig(BaseModel):
    n_dims: int = 100

    n_train: int
    n_val: int
    n_test: int

    def build(self):
        return XDBenchmarkFunc(
            **{
                k: v
                for k, v in self.model_dump().items()
                if k not in ["n_train", "n_val", "n_test"]
            }
        )


# ==========================================================================================
# ======================================= EXPERIMENTS ======================================
# ==========================================================================================


# ========================================= DeepPCE ========================================

DEEPPCE_CONFIG = dict(
    # DARCY FLOW
    deeppce_darcyflow=dict(
        model=CircuitConfig(
            n_sums=100,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=3,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.9),
            pce_var_decay=0.3,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.005, amsgrad=True),
        dataset=DarcyFlowConfig(),
        n_model_inits=10,
        model_init_epochs=50,
    ),
    # STEADY STATE DIFFUSION
    deeppce_steadystatediffusion=dict(
        model=CircuitConfig(
            n_sums=50,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=3,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.15),
            pce_var_decay=0.15,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=SteadyStateDiffusionConfig(),
        n_model_inits=10,
        model_init_epochs=50,
    ),
    # SOBOL G STAR
    deeppce_sobolgstar=dict(
        model=CircuitConfig(
            n_sums=20,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=6,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.7),
            pce_var_decay=0.15,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.01, amsgrad=True),
        dataset=SobolGStarConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    # XD BENCHMARK
    deeppce_xdbenchmark=dict(
        model=CircuitConfig(
            n_sums=20,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=6,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.8),
            pce_var_decay=0.2,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.005, amsgrad=True),
        dataset=XDBenchmarkConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
)

# BRATLEY FUNCTION
for n_dims in [100, 250]:
    DEEPPCE_CONFIG[f"deeppce_bratleysum{n_dims}"] = dict(
        model=CircuitConfig(
            n_sums=20,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=6,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.5),
            pce_var_decay=0.15,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=BratleySumConfig(n_dims=n_dims, n_train=8000, n_val=2000, n_test=2000),
        n_model_inits=10,
        model_init_epochs=50,
    )

for n_dims in [500, 750, 1000, 2000]:
    DEEPPCE_CONFIG[f"deeppce_bratleysum{n_dims}"] = dict(
        model=CircuitConfig(
            n_sums=75,
            scope_size=1,
            input_layer="pce-hermite",
            product_layer="hadamard",
            region_graph="random-binary-tree-td",
            max_order=6,
            batch_norm=True,
            param_init=partial(torch.nn.init.normal_, std=0.9),
            pce_var_decay=0.12,
            pce_var_rank="sum",
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=BratleySumConfig(n_dims=n_dims, n_train=8000, n_val=2000, n_test=2000),
        n_model_inits=10,
        model_init_epochs=50,
    )


# =========================================== UNet =========================================

UNET_CONFIG = dict(
    # DARCY FLOW
    unet_darcyflow=dict(
        model=UNetConfig(
            n_channels_in=1,
            n_channels_out=1,
            n_channels_max=1024,
            n_channels_first_step=64,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.005, amsgrad=True),
        dataset=DarcyFlowConfig(),
    ),
    # STEADY STATE DIFFUSION
    unet_steadystatediffusion=dict(
        model=UNetConfig(
            n_channels_in=1,
            n_channels_out=1,
            n_channels_max=1024,
            n_channels_first_step=64,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=SteadyStateDiffusionConfig(),
    ),
)


# =========================================== MLP ==========================================

MLP_CONFIG = dict(
    # DARCY FLOW
    mlp_darcyflow=dict(
        model=MLPConfig(
            n_hidden_layers=2,
            n_units_per_layer=2048,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=DarcyFlowConfig(),
    ),
    # STEADY STATE DIFFUSION
    mlp_steadystatediffusion=dict(
        model=MLPConfig(
            n_hidden_layers=2,
            n_units_per_layer=2048,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=SteadyStateDiffusionConfig(),
    ),
    # SOBOL G STAR
    mlp_sobolgstar=dict(
        model=MLPConfig(
            n_hidden_layers=3,
            n_units_per_layer=512,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=SobolGStarConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    mlp_xdbenchmark=dict(
        model=MLPConfig(
            n_hidden_layers=2,
            n_units_per_layer=256,
            batch_norm=True,
            activation=partial(torch.nn.ReLU),
        ),
        optimizer=OptimizerConfig(lr=0.001, amsgrad=True),
        dataset=XDBenchmarkConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
)


# ==================================== PCE baselines ====================================

PCE_CONFIG = dict(
    # SOBOL G STAR
    pceq_sobolgstar=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="lstsq",
        ),
        dataset=SobolGStarConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    pceomp_sobolgstar=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        dataset=SobolGStarConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    bapceomp_sobolgstar=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        basis_selection="fnbs",
        dataset=SobolGStarConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    # XD BENCHMARK
    pceq_xdbenchmark=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="lstsq",
        ),
        dataset=XDBenchmarkConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    pceomp_xdbenchmark=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        dataset=XDBenchmarkConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
    bapceomp_xdbenchmark=dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        basis_selection="fnbs",
        dataset=XDBenchmarkConfig(n_train=8000, n_val=2000, n_test=2000),
    ),
)


# BRATLEY FUNCTION
for n_dims in [100, 250, 500, 750, 1000, 2000]:
    PCE_CONFIG[f"pceq_bratleysum{n_dims}"] = dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="lstsq",
        ),
        dataset=BratleySumConfig(n_dims=n_dims, n_train=8000, n_val=2000, n_test=2000),
    )

    PCE_CONFIG[f"pceomp_bratleysum{n_dims}"] = dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        dataset=BratleySumConfig(n_dims=n_dims, n_train=8000, n_val=2000, n_test=2000),
    )

    PCE_CONFIG[f"pceomp_bratleysum{n_dims}"] = dict(
        model=TorchPCEConfig(
            expansion="legendre",
            max_order=5,
            truncation="hyperbolic",
            truncation_args=dict(q=0.6),
            method="omp",
        ),
        basis_selection="fnbs",
        dataset=BratleySumConfig(n_dims=n_dims, n_train=8000, n_val=2000, n_test=2000),
    )


EXPERIMENTS = {**DEEPPCE_CONFIG, **UNET_CONFIG, **MLP_CONFIG, **PCE_CONFIG}

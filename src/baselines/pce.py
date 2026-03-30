import torch
from torch import Tensor
import numpy as np
from copy import deepcopy
from collections import defaultdict

from tensorchaos.polynomials import hermite_polynomials, legendre_polynomials
from tensorchaos.indexgenerator import (
    MultiIndexGeneratorBase,
    MultiIndexGenerator,
    MultiIndexGeneratorHyperbolic,
)


class TorchPCE(torch.nn.Module):
    def __init__(
        self,
        index_set: MultiIndexGeneratorBase | Tensor | None = None,
        n_inputs: int | None = None,
        n_outputs: int = 1,
        expansion: str | None = "hermite",
        method: str | None = "lstsq",
        eps: float = 0.0,
        max_active_dims: int | None = None,
        orthonormal: bool = True,
        dtype=torch.float32,
    ):
        """Polynomial Chaos Expansion for PyTorch.

        Args:
            index_set (MultiIndexGeneratorBase | Tensor | None, optional): Set of multi-indices,
                either as generator or Tensor. If None, delays the expansion step to a later time -
                this is useful for initializing a TorchPCE and loading a state_dict. Defaults to None.
            n_inputs (int | None, optional): Number of function inputs. Is only required if index_set
                is an instance of MultiIndexGeneratorBase. Defaults to None.
            n_outputs (int, optional): Number of function outputs. Defaults to 1.
            expansion (str | None, optional): Polynomial class for the expansion. Can either be
                'hermite' for Hermite polynomials or 'legendre' for Legendre polynomials. Defaults 
                 to "hermite".
            method (str | None, optional): Method to fit the PCE.
                'lstsq': Fit a PCE using standard least-squares.
                'omp': Fit a sparse PCE using orthogonal matching pursuit based on [1].
                Defaults to "lstsq".
            eps (float, optional): Target threshold for squared residuals if using 'omp'.
                Defaults to 0.0.
            max_active_dims (int | None, optional): Maximum active dimensions if using 'omp'. 
                Defaults to None.
            orthonormal (bool, optional): Use orthonormal polynomials. Defaults to True.
            dtype (_type_, optional): Data type for model tensors. Defaults to torch.float32.

        [1] Baptista, R., Stolbunov, V., and Nair, P. B. Some greedy algorithms for sparse polynomial 
        chaos expansions. Journal of Computational Physics, 387:303-325,2019.
        """
        super().__init__()

        self.n_inputs = None
        self.n_outputs = None
        self.expansion = None
        self.max_order = None
        self.method = method
        self.eps = eps
        self.max_active_dims = max_active_dims

        if method == "lstsq":
            self.sol = self._lstsq
        elif method == "omp":
            self.sol = self._omp
        else:
            raise ValueError(f"Unknown method: {method}.")

        if type(dtype) == str:
            dtype = eval(dtype)

        if index_set is not None:
            self.build_expansion(
                index_set,
                n_inputs,
                n_outputs,
                expansion=expansion,
                orthonormal=orthonormal,
                dtype=dtype,
            )
        else:
            self.register_buffer("coeffs", None)
            self.register_buffer("exponents", None)
            self.register_buffer("tensor_product_idx", None)
            self.weights = None

    def build_expansion(
        self,
        index_set: MultiIndexGeneratorBase | Tensor,
        n_inputs: int | None = None,
        n_outputs: int = 1,
        expansion: str = "hermite",
        orthonormal: bool = True,
        dtype=torch.float32,
    ):

        if isinstance(index_set, MultiIndexGeneratorBase):
            if n_inputs is None:
                raise ValueError("n_inputs cannot be None if index_set is a MultiIndexGenerator!")
            self.max_order = index_set.max_order
            self.n_inputs = n_inputs
        elif isinstance(index_set, Tensor):
            self.max_order = int(torch.max(index_set).item())
            self.n_inputs = index_set.shape[1]

        self.register_buffer("exponents", torch.arange(self.max_order + 1))

        if expansion == "hermite":
            polynomial_coeffs = torch.tensor(
                hermite_polynomials(self.max_order, orthonormal=orthonormal), dtype=dtype
            )
        elif expansion == "legendre":
            polynomial_coeffs = torch.tensor(
                legendre_polynomials(self.max_order, orthonormal=orthonormal), dtype=dtype
            )
        else:
            raise AssertionError(f"Unknown orthogonal polynomial expansion: {expansion}")
        self.expansion = expansion
        self.register_buffer("coeffs", polynomial_coeffs)

        if isinstance(index_set, MultiIndexGeneratorBase):
            multiindex_list = torch.tensor(np.array(list(index_set(n_inputs))), dtype=torch.long)
        else:
            multiindex_list = index_set

        self.register_buffer("tensor_product_idx", multiindex_list)
        self.weights = torch.nn.Parameter(
            torch.zeros(len(self.tensor_product_idx), n_outputs, dtype=dtype), requires_grad=False
        )

    def fit(self, x, y, batch_size=None):
        TP = self.tensor_product_matrix(x, batch_size=batch_size)
        self.weights.copy_(self.sol(TP, y))
        return torch.sum(self.weights.unsqueeze(0) * TP.unsqueeze(-1), dim=1)

    @staticmethod
    def _lstsq(TP, y):
        """Fit PCE using standard least squares."""
        return torch.linalg.lstsq(TP, y).solution

    def _omp(self, TP, y):
        """Fit sparse PCE using orthogonal matching pursuit. The algorithm is based on [1].
        
        [1] Baptista, R., Stolbunov, V., and Nair, P. B. Some greedy algorithms for sparse polynomial 
        chaos expansions. Journal of Computational Physics, 387:303-325,2019.
        """
        n_obs, n_terms = TP.shape
        max_active_dims = (
            round(n_terms / 10) if self.max_active_dims is None else self.max_active_dims
        )

        residuals = y.clone()
        active_dim_indices = []
        w = torch.zeros(n_terms, 1, device=TP.device, dtype=TP.dtype)

        while (
            torch.sqrt(torch.sum(residuals**2)) > self.eps
            or len(active_dim_indices) < max_active_dims
        ):
            correlations = torch.matmul(TP.T, residuals).squeeze()
            idx = torch.argmax(torch.abs(correlations)).item()

            if idx in active_dim_indices:
                break

            active_dim_indices.append(idx)

            TP_active = TP[:, active_dim_indices]
            weights_active = torch.linalg.lstsq(TP_active, y).solution
            residuals = torch.matmul(TP_active, weights_active) - y

        w[active_dim_indices] = weights_active
        return w

    def predict(self, x, batch_size=None):
        TP = self.tensor_product_matrix(x, batch_size=batch_size)
        pred = torch.sum(self.weights.unsqueeze(0) * TP.unsqueeze(-1), dim=1)
        return pred

    def sobol_indices_first_order(self, indices=None):
        """Compute first-order Sobol indices"""
        if indices is None:
            indices = torch.arange(self.n_inputs).unsqueeze(1)
        elif isinstance(indices, torch.Tensor):
            indices = [indices]

        sobol_indices = torch.zeros(len(indices), self.weights.shape[1], device=self.weights.device)
        for i in indices:
            # include only expansion terms where the order of the basis function for X_i is > 0
            weight_mask = (self.tensor_product_idx[:, i] > 0).flatten()

            # exclude all interaction terms; include only the 0th-order basis function for all j != i
            exclude_idx = torch.ones(self.n_inputs, dtype=torch.bool, device=self.weights.device)
            exclude_idx[i] = 0
            other_idx_zero = torch.all(
                self.tensor_product_idx[:, exclude_idx] == 0, dim=1
            ).flatten()
            weight_mask = weight_mask & other_idx_zero
            sobol_indices[i, :] = torch.sum(self.weights[weight_mask, :] ** 2, dim=0)
        return sobol_indices / self.var

    def sobol_indices_total_effect(self, indices=None):
        """Compute total effect Sobol indices."""
        if indices is None:
            indices = torch.arange(self.n_inputs).unsqueeze(1)
        elif isinstance(indices, torch.Tensor):
            indices = [indices]

        sobol_indices = torch.zeros(len(indices), self.weights.shape[1], device=self.weights.device)
        for i in indices:
            # include only expansion terms where the order of the basis function for X_i is > 0
            weight_mask = (self.tensor_product_idx[:, i] > 0).flatten()
            sobol_indices[i, :] = torch.sum(self.weights[weight_mask, :] ** 2, dim=0)
        return sobol_indices / self.var

    def tensor_product_matrix(self, x: Tensor, batch_size: int | None = None) -> Tensor:
        """Computes tensor product matrix of polynomial expansion.

        Args:
            x (Tensor): Input data of shape (B, N) with batch size B and input dimensionality N.
            batch_size (int | None): Compute tensor product matrix batchwise to reduce memory. If
                None, computes full tensor product matrix in a single step. Defaults to None.

        Returns:
            Tensor: Tensor product matrix of shape (B, C) with B = batch size and C = cardinality.
        """
        x_powers = self.expand(x)
        poly_matrix = x_powers @ torch.t(self.coeffs).unsqueeze(0)
        if not batch_size:
            TP = torch.prod(
                poly_matrix[:, torch.arange(poly_matrix.shape[1]), self.tensor_product_idx], dim=-1
            )
        else:
            TP = torch.zeros(
                poly_matrix.shape[0],
                len(self.tensor_product_idx),
                dtype=poly_matrix.dtype,
                device=poly_matrix.device,
            )

            for b in range(int(np.ceil(poly_matrix.shape[0] / batch_size))):
                s = b * batch_size
                e = s + batch_size
                TP[s:e, ...] = torch.prod(
                    poly_matrix[s:e, torch.arange(poly_matrix.shape[1]), self.tensor_product_idx],
                    dim=-1,
                )
        return TP

    def expand(self, x):
        num_axis = len(x.shape)
        return (
            torch.tile(x.unsqueeze(-1), (1,) * num_axis + (len(self.exponents),)) ** self.exponents
        )

    def polynomial_matrix(self, x):
        x_powers = self.expand(x)
        return x_powers @ torch.t(self.coeffs).unsqueeze(0)

    @property
    def E(self):
        return self.weights[0, :]

    @property
    def var(self):
        return torch.sum(self.weights[1:, :] ** 2, dim=0)


def build_pce(
    n_inputs: int | None,
    n_outputs: int = 1,
    expansion: str = "hermite",
    max_order: int = 3,
    truncation: str = "total_order",
    truncation_args: dict | None = None,
    index_set: Tensor | None = None,
    method: str | None = "lstsq",
    eps: float = 0.0,
    max_active_dims: int | None = None,
    orthonormal: bool = True,
    dtype=torch.float32,
):

    if index_set is None:
        truncation_args = dict() if truncation_args is None else truncation_args
        if truncation == "total_order":
            index_set = MultiIndexGenerator(max_order=max_order, **truncation_args)
        elif truncation == "hyperbolic":
            index_set = MultiIndexGeneratorHyperbolic(max_order=max_order, **truncation_args)

    pce = TorchPCE(
        index_set=index_set,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
        expansion=expansion,
        method=method,
        eps=eps,
        max_active_dims=max_active_dims,
        orthonormal=orthonormal,
        dtype=dtype,
    )
    return pce


def forward_neighbor_basis_selection(
    pce: TorchPCE,
    x_train: Tensor,
    y_train: Tensor,
    x_val: Tensor,
    y_val: Tensor,
    expand_active_dims: int = 3,
):
    """Forward neighbor adaptive basis selection algorithm for PCEs based on the paper [1].

    [1] J.D. Jakeman, M.S. Eldred, K. Sargsyan. Enhancig l1-minimization estimates of polynomial 
    chaos expansions using basis selection. Journal of Computational Physics 289, 2015.

    expand_active_dims corresponds to the term T in Algorithm 1 on page 25 in the paper.

    """
    expansion = pce.expansion
    method = pce.method
    n_dims = pce.tensor_product_idx.shape[1]
    device = pce.weights.device

    pce.fit(x_train, y_train)
    pce_predictions = pce.predict(y_val)

    best_pce_loss = torch.mean((pce_predictions - y_val) ** 2)
    best_pce = deepcopy(pce.state_dict())

    term_tree = defaultdict(list)
    for t in pce.tensor_product_idx:
        rank = torch.sum(t).item()
        term_tree[rank].append(t.clone())

    n_iter = 1
    while True:
        print(f"Start basis adaption round {n_iter}...")
        best_loss_run = torch.inf
        best_pce_run = None
        terms_added = 0

        pce_weights = best_pce["weights"]
        pce_basis = best_pce["tensor_product_idx"]
        n_terms, n_out_dims = pce_weights.shape
        if n_out_dims > 1:
            raise AssertionError(
                "Forward neighbor basis adaptivity not implemented for models with output dimensions > 1!"
            )
        pce_weights = pce_weights.flatten()
        active_terms = torch.nonzero(pce_weights)
        t_iter = 0
        for t in torch.argsort(pce_weights[active_terms.flatten()])[:expand_active_dims]:
            term_total_degree = torch.sum(pce_basis[t]).item()
            if term_total_degree > 1:
                forward_neighbors = []
                for i in range(n_dims):
                    forward_n = pce_basis[t].clone()
                    forward_n[i] += 1
                    backward_neighbors = []
                    for j in range(n_dims):
                        if pce_basis[t][j] > 0:
                            backward_n = pce_basis[t].clone()
                            backward_n[j] -= 1
                            backward_neighbors.append(backward_n)
                    if all(
                        [
                            any(torch.equal(b, c) for c in term_tree[term_total_degree - 1])
                            for b in backward_neighbors
                        ]
                    ):
                        forward_neighbors.append(forward_n.unsqueeze(0))
                basis_expanded = torch.concat([pce_basis, *forward_neighbors])

                pce = TorchPCE(expansion=expansion, method=method, index_set=basis_expanded).to(
                    device
                )
                pce.fit(x_train, y_train)
                pce_predictions = pce.predict(x_val)
                loss = torch.mean((pce_predictions - y_val) ** 2)

                if loss < best_loss_run:
                    terms_added = len(forward_neighbors)
                    best_loss_run = loss
                    best_pce_run = deepcopy(pce.state_dict())
            t_iter += 1

        if best_loss_run < best_pce_loss:
            best_pce_loss = best_loss_run
            best_pce = best_pce_run
            print(f"added {terms_added} terms.")
        else:
            break
        n_iter += 1

    print("Basis adaptation done.")
    tp = best_pce["tensor_product_idx"]
    pce = TorchPCE(index_set=tp, expansion=expansion, method=method)
    pce.load_state_dict(best_pce)
    pce.to(device)
    return pce

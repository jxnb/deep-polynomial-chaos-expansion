import numpy as np
from pathlib import Path


class UQFunc:
    """UQ test function base class"""

    def __init__(self, n_dims: int, **parameters):
        self.n_dims = n_dims
        self.parameters = parameters

    @classmethod
    def f(cls, x, **parameters):
        raise NotImplementedError

    def sample(self, n_samples, transform_x=True, seed=None):
        rng = np.random.default_rng(seed)
        x = self._sample_inputs(n_samples, rng)
        if transform_x:
            return self.transform_inputs(x), self.f(x, **self.parameters)
        else:
            return x, self.f(x, **self.parameters)

    def _sample_inputs(self, n_samples, rng):
        return rng.random((n_samples, self.n_dims))

    @classmethod
    def transform_inputs(cls, x):
        """Transform inputs from a uniform distribution on the interval [0. 1] to the interval [-1, 1]"""
        return 2 * x - 1


class SobolGStarFunc(UQFunc):
    def __init__(self, n_dims, a=None, alpha=0.5, delta=None):
        """Sobol G Star function [1]

        [1] Saltelli, A. and Sobol', I. M. (1995). About the use of rank transformation in sensitivity
        analysis of model output. Reliability Engineering & System Safety, 50(3):225-239.

        The analytical solutions for the Sobol' indices are based on the UQTestFuns library [2], see:

        https://uqtestfuns.readthedocs.io/en/latest/test-functions/sobol-g-star.html

        [2] Wicaksono, D. and Hecht, M. (2023). UQTestFuns: A Python3 library of uncertainty
        quantication (UQ) test functions. Journal of Open Source Software, 8(90).
        """
        a = np.linspace(0, 10, n_dims) if a is None else a
        delta = np.random.rand(n_dims) if delta is None else delta
        super().__init__(n_dims=n_dims, a=a, alpha=alpha, delta=delta)

    @classmethod
    def f(cls, x, a, alpha, delta):
        t = np.abs(2 * (x + delta - np.floor(x + delta)) - 1) ** alpha
        y = ((1 + alpha) * t + a) / (1 + a)
        return np.prod(y, axis=1)

    @property
    def E(self):
        return 1.0

    @property
    def var(self):
        return np.prod(self.var_cond_E + 1) - 1

    @property
    def var_cond_E(self):
        a = self.parameters["a"]
        alpha = self.parameters["alpha"]
        return (alpha**2) / ((1 + 2 * alpha) * (1 + a) ** 2)

    @property
    def sobol_indices(self):
        v_i = self.var_cond_E
        s_i = v_i / self.var
        s_t = (
            v_i * np.prod(np.eye(self.n_dims) + (1 - np.eye(self.n_dims)) * (1 + v_i), axis=1)
        ) / self.var
        return s_i, s_t


class BratleySumFunc(UQFunc):
    def __init__(self, n_dims):
        """Function first introduced by Bratley [1].

        [1] Bratley, P., Fox, B. L., and Niederreiter, H. (1992). Implementation and tests of
        low-discrepancy sequences. ACM Trans. Model. Comput. Simul., 2(3):195-213.

        The function is also part of the UQTestFuns library [2]:

        https://uqtestfuns.readthedocs.io/en/latest/test-functions/bratley1992d.html

        [2] Wicaksono, D. and Hecht, M. (2023). UQTestFuns: A Python3 library of uncertainty
        quantication (UQ) test functions. Journal of Open Source Software, 8(90).
        """
        super().__init__(n_dims=n_dims)

    @classmethod
    def f(cls, x):
        y = np.zeros(x.shape[0])
        for d in range(x.shape[1]):
            y += (-1) ** (d + 1) * np.prod(x[:, : d + 1], axis=1)
        return y
    

class XDBenchmarkFunc(UQFunc):
    def __init__(self, n_dims: int = 100):
        """Polynomial Chaos benchmark function with arbitrary input dimensions and single output
        dimension based on [1]. The expectation of the function is computed analytically, while the
        (conditional) variances for the Sobol' indices were estimated with Monte Carlo for the case
        where n_dims = 100.

        [1] Lüthen, Nora, Stefano Marelli, and Bruno Sudret. 2021. “Sparse Polynomial Chaos Expansions:
        Literature Survey and Benchmark.” SIAM/ASA Journal on Uncertainty Quantification 9 (2):
        593-649.
        """
        super().__init__(n_dims=n_dims)

    @classmethod
    def f(cls, x):
        n_dims = x.shape[1]
        k = np.arange(1, n_dims + 1).reshape(1, -1)

        y = (
            3
            - (5 / n_dims) * np.sum(k * x, axis=1)
            + (1 / n_dims) * np.sum(k * x**3, axis=1)
            + (1 / (3 * n_dims)) * np.sum(k * np.log(x**2 + x**4), axis=1)
        )

        if n_dims >= 2:
            y += x[:, 0] * x[:, 1] ** 2
        if n_dims >= 4:
            y += x[:, 1] * x[:, 3]
        if n_dims >= 5:
            y -= x[:, 2] * x[:, 4]
        if n_dims >= 51:
            y += x[:, 50]
        if n_dims >= 54:
            y += x[:, 49] * x[:, 53] ** 2

        return y

    def _sample_inputs(self, n_samples, rng):
        # Inputs are sampled from a uniform distribution on the interval [1, 2], except X_20 which is
        # on the interval [1, 3].
        x = rng.random((n_samples, self.n_dims)) + 1
        x[:, 19] = (3 - 1) * (x[:, 19] - 1) / (2 - 1) + 1
        return x

    @classmethod
    def transform_inputs(cls, x):
        """Transform inputs from a uniform distribution on the interval [1, 2] and [1, 3] to the
        interval [-1, 1]"""
        u = 2 * (x - 1) / (2 - 1) - 1  # [1, 2] -> [-1, 1]  (bn - an) * (x - a) / (b - a) + an
        u[:, 19] = 2 * (x[:, 19] - 1) / (3 - 1) - 1  # [1, 3] -> [-1, 1]
        return u

    @property
    def E(self, domain, conditionals=None):
        """Analytical solution to (conditional) expectations"""

        def log_term(a, b):
            return (
                b * np.log(b**4 + b**2)
                + 2 * np.arctan(b)
                - 4 * b
                - a * np.log(a**4 + a**2)
                - 2 * np.arctan(a)
                + 4 * a
            )

        a = domain[:, 0]
        b = domain[:, 1]

        marginals_mask = np.ones(self.n_dims, dtype=np.bool)
        ind = np.arange(self.n_dims) + 1

        if conditionals:
            c_idx, c_vals = conditionals
            if c_vals.ndim != 2:
                raise AssertionError(f"Expected conditional values with 2 dims, got {c_vals.ndim}")
            marginals_mask[c_idx] = 0
            conditionals_mask = ~marginals_mask

        normalizing_const = np.prod(1 / (b[marginals_mask] - a[marginals_mask]))
        n_marginal_dims = len(marginals_mask[marginals_mask == True])

        # Term 1
        t1 = 3 * np.prod(b[marginals_mask] - a[marginals_mask])

        # Term 2
        const_part = b[marginals_mask] - a[marginals_mask]
        integrated_part = b[marginals_mask] ** 2 - a[marginals_mask] ** 2
        x_marginal = np.prod(
            (1 - np.eye(n_marginal_dims)) * const_part[None, ...]
            + (np.eye(n_marginal_dims) * integrated_part),
            axis=1,
        )
        x_marginal = (1 / 2) * np.sum(x_marginal * ind[marginals_mask])

        if conditionals is not None:
            const_part = b - a
            conditional_part = c_vals[:, None, :]
            xc1 = (
                np.ones((len(c_idx), self.n_dims))
                * marginals_mask[None, ...]
                * const_part[None, ...]
            )
            xc21 = ((1 - np.eye(self.n_dims))[conditionals_mask] * conditionals_mask[None, ...])[
                ..., None
            ]
            xc22 = np.eye(self.n_dims)[conditionals_mask][..., None] * conditional_part
            xc2 = xc21 + xc22
            x_conditional = np.sum(
                np.prod((xc1[..., None] + xc2), axis=1) * ind[conditionals_mask][..., None], axis=0
            )
        else:
            x_conditional = 0

        t2 = (5 / self.n_dims) * (x_marginal + x_conditional)

        # Term 3
        const_part = b[marginals_mask] - a[marginals_mask]
        integrated_part = b[marginals_mask] ** 4 - a[marginals_mask] ** 4
        x_marginal = np.prod(
            (1 - np.eye(n_marginal_dims)) * const_part[None, ...]
            + (np.eye(n_marginal_dims) * integrated_part),
            axis=1,
        )
        x_marginal = (1 / 4) * np.sum(x_marginal * ind[marginals_mask])

        if conditionals is not None:
            const_part = b - a
            conditional_part = c_vals[:, None, :] ** 3
            xc1 = (
                np.ones((len(c_idx), self.n_dims))
                * marginals_mask[None, ...]
                * const_part[None, ...]
            )
            xc21 = ((1 - np.eye(self.n_dims))[conditionals_mask] * conditionals_mask[None, ...])[
                ..., None
            ]
            xc22 = np.eye(self.n_dims)[conditionals_mask][..., None] * conditional_part
            xc2 = xc21 + xc22
            x_conditional = np.sum(
                np.prod((xc1[..., None] + xc2), axis=1) * ind[conditionals_mask][..., None], axis=0
            )
        else:
            x_conditional = 0

        t3 = (1 / self.n_dims) * (x_marginal + x_conditional)

        # Term 4
        const_part = b[marginals_mask] - a[marginals_mask]
        integrated_part = log_term(a[marginals_mask], b[marginals_mask])
        x_marginal = np.prod(
            (1 - np.eye(n_marginal_dims)) * const_part[None, ...]
            + (np.eye(n_marginal_dims) * integrated_part),
            axis=1,
        )
        x_marginal = np.sum(x_marginal * ind[marginals_mask])

        if conditionals is not None:
            const_part = b - a
            conditional_part = np.log(c_vals[:, None, :] ** 2 + c_vals[:, None, :] ** 4)
            xc1 = (
                np.ones((len(c_idx), self.n_dims))
                * marginals_mask[None, ...]
                * const_part[None, ...]
            )
            xc21 = ((1 - np.eye(self.n_dims))[conditionals_mask] * conditionals_mask[None, ...])[
                ..., None
            ]
            xc22 = np.eye(self.n_dims)[conditionals_mask][..., None] * conditional_part
            xc2 = xc21 + xc22
            x_conditional = np.sum(
                np.prod((xc1[..., None] + xc2), axis=1) * ind[conditionals_mask][..., None], axis=0
            )
        else:
            x_conditional = 0

        t4 = (1 / (3 * self.n_dims)) * (x_marginal + x_conditional)

        y = t1 - t2 + t3 + t4

        if self.n_dims >= 2:  # x1 * x2 ** 2
            interaction_indices = [0, 1]
            interaction_mask = np.copy(marginals_mask)
            if np.all(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                x2_int = (1 / 3) * (b[j] ** 3 - a[j] ** 3)
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * x2_int * const

            elif not np.any(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                cj = np.where(c_idx == j)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci] * c_vals[cj] ** 2

            elif marginals_mask[interaction_indices[0]]:
                i, j = interaction_indices
                cj = np.where(c_idx == j)[0][0]
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                interaction_mask[i] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[cj] ** 2
                t = x1_int * const

            elif marginals_mask[interaction_indices[1]]:
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                x2_int = (1 / 3) * (b[j] ** 3 - a[j] ** 3)
                interaction_mask[j] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[ci]
                t = x2_int * const
            y += t

        if self.n_dims >= 4:  # x2 * x4
            interaction_indices = [1, 3]
            interaction_mask = np.copy(marginals_mask)
            if np.all(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                x2_int = (1 / 2) * (b[j] ** 2 - a[j] ** 2)
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * x2_int * const

            elif not np.any(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                cj = np.where(c_idx == j)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci] * c_vals[cj]

            elif marginals_mask[interaction_indices[0]]:
                i, j = interaction_indices
                cj = np.where(c_idx == j)[0][0]
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                interaction_mask[i] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[cj]
                t = x1_int * const

            elif marginals_mask[interaction_indices[1]]:
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                x2_int = (1 / 2) * (b[j] ** 2 - a[j] ** 2)
                interaction_mask[j] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[ci]
                t = x2_int * const
            y += t

        if self.n_dims >= 5:  # x3 * x5
            interaction_indices = [2, 4]
            interaction_mask = np.copy(marginals_mask)
            if np.all(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                x2_int = (1 / 2) * (b[j] ** 2 - a[j] ** 2)
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * x2_int * const

            elif not np.any(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                cj = np.where(c_idx == j)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci] * c_vals[cj]

            elif marginals_mask[interaction_indices[0]]:
                i, j = interaction_indices
                cj = np.where(c_idx == j)[0][0]
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                interaction_mask[i] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[cj]
                t = x1_int * const

            elif marginals_mask[interaction_indices[1]]:
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                x2_int = (1 / 2) * (b[j] ** 2 - a[j] ** 2)
                interaction_mask[j] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[ci]
                t = x2_int * const
            y -= t

        if self.n_dims >= 51:  # x51
            interaction_indices = 50
            interaction_mask = np.copy(marginals_mask)
            if marginals_mask[interaction_indices]:
                i = interaction_indices
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * const
            else:
                ci = np.where(c_idx == interaction_indices)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci]
            y += t

        if self.n_dims >= 54:  # x50 * x54 ** 2
            interaction_indices = [49, 53]
            interaction_mask = np.copy(marginals_mask)
            if np.all(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                x2_int = (1 / 3) * (b[j] ** 3 - a[j] ** 3)
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * x2_int * const

            elif not np.any(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                cj = np.where(c_idx == j)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci] * c_vals[cj] ** 2

            elif marginals_mask[interaction_indices[0]]:
                i, j = interaction_indices
                cj = np.where(c_idx == j)[0][0]
                x1_int = (1 / 2) * (b[i] ** 2 - a[i] ** 2)
                interaction_mask[i] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[cj] ** 2
                t = x1_int * const

            elif marginals_mask[interaction_indices[1]]:
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                x2_int = (1 / 3) * (b[j] ** 3 - a[j] ** 3)
                interaction_mask[j] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[ci]
                t = x2_int * const
            y += t

        for interaction_indices, pow, sign in self.additional_interactions:
            interaction_mask = np.copy(marginals_mask)
            if np.all(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                x1_int = (1 / (pow[0] + 1)) * (b[i] ** (pow[0] + 1) - a[i] ** (pow[0] + 1))
                x2_int = (1 / (pow[1] + 1)) * (b[j] ** (pow[1] + 1) - a[j] ** (pow[1] + 1))
                interaction_mask[interaction_indices] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask])
                t = x1_int * x2_int * const

            elif not np.any(marginals_mask[interaction_indices]):
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                cj = np.where(c_idx == j)[0][0]
                t = np.prod(b[marginals_mask] - a[marginals_mask]) * c_vals[ci] * c_vals[cj] ** 2

            elif marginals_mask[interaction_indices[0]]:
                i, j = interaction_indices
                cj = np.where(c_idx == j)[0][0]
                x1_int = (1 / (pow[0] + 1)) * (b[i] ** (pow[0] + 1) - a[i] ** (pow[0] + 1))
                interaction_mask[i] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[cj] ** 2
                t = x1_int * const

            elif marginals_mask[interaction_indices[1]]:
                i, j = interaction_indices
                ci = np.where(c_idx == i)[0][0]
                x2_int = (1 / (pow[1] + 1)) * (b[j] ** (pow[1] + 1) - a[j] ** (pow[1] + 1))
                interaction_mask[j] = 0
                const = np.prod(b[interaction_mask] - a[interaction_mask]) * c_vals[ci]
                t = x2_int * const
            y += t * sign

        return normalizing_const * y

    @property
    def sobol_indices(self):
        var_conditional_expectations = np.load(
            Path(Path(__file__).parent, f"xdbenchmark_var_cond_exp_MC_{self.ndims}.npy")
        )
        var_total = np.load(
            Path(Path(__file__).parent, f"xdbenchmark_variance_MC_{self.ndims}.npy")
        )
        return None, var_conditional_expectations / var_total


def xdbenchmark_monte_carlo(n_dims: int, n_samples: int, out_dir: Path | str):
    """Compute variance Var(Y) and variances of conditional expectations Var(E[Y | X]) for the
    XD benchmark function."""

    # Inputs X are sampled from a uniform distribution on the interval [1, 2], X_20 on the interval [1, 3].
    domain = np.ones((n_dims, 2))
    domain[:, 1] = 2
    if n_dims >= 19:
        domain[19, 1] = 3

    Path(out_dir).mkdir(parents=True, exist_ok=True)

    xdbenchmark = XDBenchmarkFunc(n_dims=n_dims)
    y_mean = xdbenchmark.E(domain, conditionals=None)
    out_var_E_cond = np.zeros(n_dims)
    for i in range(n_dims):
        print(f"dim {i + 1}...")
        a, b = domain[i]
        x = np.linspace(a, b, n_samples)[None, ...]
        y = xdbenchmark.E(domain, conditionals=(np.array([[i]]), np.array(x)))
        out_var_E_cond[i] = np.sum((y - y_mean) ** 2) / (n_samples - 1)

    np.save(Path(out_dir, f"xd_mean_{n_dims}.npy"), y_mean)
    np.save(Path(out_dir, f"xd_var_cond_exp_{n_dims}.npy"), out_var_E_cond)
    np.save(Path(out_dir, f"xd_var_cond_exp_{n_dims}_domain.npy"), domain)

    print("Monte Carlo Variance estimation")
    x = (
        np.random.sample((n_samples, len(domain))) * (domain[:, 1] - domain[:, 0])[None, ...]
        + domain[:, 0]
    )
    batches = 1000
    print(f"n samples: {batches * n_samples:.0e}")
    y_var = 0
    for i in range(batches):
        print(f"batch {i + 1}...")
        y = xdbenchmark.f(x)
        y_var += np.sum((y - y_mean) ** 2)
    y_var = y_var / (n_samples * batches - 1)
    np.save(Path(out_dir, f"xd_variance_MC_{n_dims}.npy"), y_var)

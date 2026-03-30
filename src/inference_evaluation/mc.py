import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable, Callable

from tensorchaos.circuits import TensorCircuit
from tensorchaos.circuits.inference import expectation, covariance


def E_pX(y_mc, **kwargs):
    return torch.mean(y_mc, dim=0).detach().cpu().numpy()


def var_pX(y_mc, **kwargs):
    return torch.var(y_mc, dim=0).detach().cpu().numpy()


def cov_pX(y_mc, **kwargs):
    return torch.cov(y_mc).detach().cpu().numpy()


def var_pX_batch(y_mc, y_mean, **kwargs):
    return torch.sum((y_mc - y_mean) ** 2, dim=0).detach().cpu().numpy()


def cov_pX_batch(y_mc, y_mean, var_dim=1, **kwargs):
    x = y_mc - y_mean
    return torch.sum(x.unsqueeze(var_dim + 1) * x.unsqueeze(var_dim), dim=0).detach().cpu().numpy()


def E_var_pX_Y_batch(y_mc, y_mean, **kwargs):
    return (1 / (y_mc.shape[0] - 1)) * torch.sum((y_mc - y_mean) ** 2, dim=0).detach().cpu().numpy()


def E_cov_pX_Y_batch(y_mc, y_mean, var_dim=1, **kwargs):
    x = y_mc - y_mean
    return (1 / (y_mc.shape[0] - 1)) * torch.sum(
        x.unsqueeze(var_dim + 1) * x.unsqueeze(var_dim), dim=0
    ).detach().cpu().numpy()


def var_E_pX_Y_batch(y_mc, y_mean, **kwargs):
    return (
        torch.sum((torch.mean(y_mc, dim=0).unsqueeze(0) - y_mean) ** 2, dim=0)
        .detach()
        .cpu()
        .numpy()
    )


def cov_E_pX_Y_batch(y_mc, y_mean, var_dim=1, **kwargs):
    x = torch.mean(y_mc, dim=0).unsqueeze(0) - y_mean
    return torch.sum(x.unsqueeze(var_dim + 1) * x.unsqueeze(var_dim), dim=0).detach().cpu().numpy()


def _mc_single_batch(
    model: TensorCircuit,
    f_model: Callable,
    f_E: Callable,
    f_var: Callable,
    y_mean: Callable,
    n_samples: int,
    sample_distribution: torch.distributions.Distribution,
    fix_inputs: int | Iterable | None = None,
    fix_values: int | Iterable | None = None,
    device: str = "cpu",
    all_layers: bool = False,
):

    samples = sample_distribution.sample((n_samples,)).to(device)
    if fix_inputs is not None:
        if fix_values is not None:
            samples[:, fix_inputs] = fix_values
        else:
            samples[:, fix_inputs] = torch.randn_like(fix_inputs, dtype=samples.dtype)

    if all_layers:
        model_preds = f_model(model=model, x=samples, fix_inputs=fix_inputs, fix_values=fix_values)
        preds_mean = [] if f_E else None
        preds_var = [] if f_var else None
        for l, model_preds_layer in enumerate(model_preds):
            if f_E:
                preds_mean.append(f_E(model_preds_layer))
            if f_var:
                preds_var.append(f_var(model_preds_layer, y_mean=y_mean, var_dim=1))

    else:
        model_preds = f_model(model=model, x=samples, fix_inputs=fix_inputs, fix_values=fix_values)
        if f_E:
            preds_mean = f_E(model_preds)
        else:
            preds_mean = None
        if f_var:
            preds_var = f_var(model_preds)
        else:
            preds_var = None

    return preds_mean, preds_var


def _mc_multi_batch(
    model: TensorCircuit,
    f_model: Callable,
    f_E: Callable,
    f_var: Callable,
    y_mean: Callable,
    batch_size: int,
    n_samples: int,
    sample_distribution: torch.distributions.Distribution,
    fix_inputs: int | Iterable | None = None,
    fix_values: int | Iterable | None = None,
    device: str = "cpu",
    all_layers: bool = False,
):
    n_batches = int(n_samples / batch_size)

    if all_layers:
        preds_mean = [0 for _ in range(len(model.get_layers()))] if f_E else None
        preds_var = [0 for _ in range(len(model.get_layers()))] if f_var else None
    else:
        preds_mean = 0 if f_E else None
        preds_var = 0 if f_var else None

    pbar = (
        tqdm(range(n_batches), leave=False)
        if n_batches > 100 or n_samples >= 10**4
        else range(n_batches)
    )
    for _ in pbar:
        batch_s = sample_distribution.sample((batch_size,)).to(device)

        if fix_inputs is not None:
            if fix_values is not None:
                batch_s[:, fix_inputs] = fix_values
                batch_fix_values = fix_values
            else:
                batch_fix_values = torch.randn_like(fix_inputs, dtype=batch_s.dtype)
                batch_s[:, fix_inputs] = batch_fix_values
        else:
            batch_fix_values = None

        if y_mean is not None:
            y_mean_batch = y_mean(
                model=model,
                fix_inputs=fix_inputs,
                fix_values=batch_fix_values,
                return_all_outputs=all_layers,
            )
        else:
            y_mean_batch = None

        if all_layers:
            model_preds_batch = f_model(
                model=model, x=batch_s, fix_inputs=fix_inputs, fix_values=batch_fix_values
            )
            for i, (model_preds_layer, y_mean_layer) in enumerate(
                zip(model_preds_batch, y_mean_batch)
            ):
                if f_E:
                    preds_mean[i] += f_E(model_preds_layer, y_mean=y_mean_layer)
                if f_var:
                    preds_var[i] += f_var(model_preds_layer, y_mean=y_mean_layer)

        else:
            model_preds_batch = f_model(
                model=model, x=batch_s, fix_inputs=fix_inputs, fix_values=batch_fix_values
            )
            if f_E:
                preds_mean += f_E(model_preds_batch, y_mean=y_mean_batch)
            if f_var:
                preds_var += f_var(model_preds_batch, y_mean=y_mean_batch)

    if all_layers:
        if f_E:
            preds_mean = [x / n_batches for x in preds_mean]
        if f_var:
            preds_var = [x / (n_batches * batch_size - 1) for x in preds_var]
    else:
        if f_E:
            preds_mean = preds_mean / n_batches
        if f_var:
            preds_var = preds_var / (n_batches * batch_size - 1)

    return preds_mean, preds_var


def monte_carlo_pX(
    model: TensorCircuit,
    sample_distribution: torch.distributions.Distribution,
    n_samples: int,
    batch_size: int = 1000,
    device: str = "cpu",
    return_cov: bool = True,
    all_layers: bool = False,
):
    model.eval()
    with torch.no_grad():
        if all_layers:

            def f_model(model, x, **kwargs):
                return model.get_layerwise_outputs(x)

        else:

            def f_model(model, x, **kwargs):
                return model.predict(x)

        if n_samples <= batch_size:

            E, var = _mc_single_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=cov_pX if return_cov else var_pX,
                y_mean=None,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=None,
                fix_values=None,
                device=device,
                all_layers=all_layers,
            )

        else:
            y_mean = expectation(model, return_all_outputs=all_layers)

            def get_y_mean(**kwargs):
                return y_mean

            E, var = _mc_multi_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=cov_pX_batch if return_cov else var_pX_batch,
                y_mean=get_y_mean,
                batch_size=batch_size,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=None,
                fix_values=None,
                device=device,
                all_layers=all_layers,
            )

        return E, var


def monte_carlo_pX_y(
    model: TensorCircuit,
    fix_inputs: int | Iterable,
    fix_values: int | Iterable,
    sample_distribution: torch.distributions.Distribution,
    n_samples: int,
    batch_size: int = 1000,
    device: str = "cpu",
    return_cov: bool = True,
    all_layers: bool = False,
):
    model.eval()
    with torch.no_grad():
        if all_layers:

            def f_model(model, x, **kwargs):
                return model.get_layerwise_outputs(x)

        else:

            def f_model(model, x, **kwargs):
                return model.predict(x)

        if n_samples <= batch_size:
            E_cond, var_cond = _mc_single_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=cov_pX if return_cov else var_pX,
                y_mean=None,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=fix_values,
                device=device,
                all_layers=all_layers,
            )

        else:
            y_mean = expectation(model, fix_inputs, fix_values, return_all_outputs=all_layers)

            def get_y_mean(**kwargs):
                return y_mean

            E_cond, var_cond = _mc_multi_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=cov_pX_batch if return_cov else var_pX_batch,
                y_mean=get_y_mean,
                batch_size=batch_size,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=fix_values,
                device=device,
                all_layers=all_layers,
            )
        return E_cond, var_cond


def monte_carlo_E_var_pX_Y(
    model: TensorCircuit,
    fix_inputs: int | Iterable,
    sample_distribution: torch.distributions.Distribution,
    n_samples: int,
    batch_size: int = 1000,
    device: str = "cpu",
    return_cov: bool = True,
    all_layers: bool = False,
):
    model.eval()
    with torch.no_grad():
        if all_layers:

            def f_model(model, x, **kwargs):
                return model.get_layerwise_outputs(x)

        else:

            def f_model(model, x, **kwargs):
                return model.predict(x)

        if n_samples <= batch_size:
            fix_values = torch.randn(fix_inputs.shape).to(device)
            E_var_cond, _ = _mc_single_batch(
                model=model,
                f_model=f_model,
                f_E=cov_pX if return_cov else var_pX,
                f_var=None,
                y_mean=None,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=fix_values,
                device=device,
                all_layers=all_layers,
            )

        else:

            def get_y_mean(model, fix_inputs, fix_values, return_all_outputs, **kwargs):
                return expectation(model, fix_inputs, fix_values, return_all_outputs)

            E_var_cond, _ = _mc_multi_batch(
                model=model,
                f_model=f_model,
                f_E=E_cov_pX_Y_batch if return_cov else E_var_pX_Y_batch,
                f_var=None,
                y_mean=get_y_mean,
                batch_size=batch_size,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=None,
                device=device,
                all_layers=all_layers,
            )
        return None, E_var_cond


def monte_carlo_E_var_pX_Y_2(
    model: TensorCircuit,
    fix_inputs: int | Iterable,
    sample_distribution: torch.distributions.Distribution,
    n_samples: int,
    batch_size: int = 1000,
    device: str = "cpu",
    return_cov: bool = True,
    all_layers: bool = False,
):
    model.eval()
    with torch.no_grad():
        if all_layers:

            def f_model(model, fix_inputs, fix_values, **kwargs):
                return covariance(model, fix_inputs, fix_values, return_all_outputs=True)

        else:

            def f_model(model, fix_inputs, fix_values, **kwargs):
                return covariance(model, fix_inputs, fix_values)

        if n_samples <= batch_size:
            fix_values = torch.randn(n_samples, len(fix_inputs)).to(device)
            fix_inputs = torch.ones_like(fix_values, dtype=fix_inputs.dtype) * fix_inputs.unsqueeze(
                0
            )
            E_var_cond, _ = _mc_single_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=None,
                y_mean=None,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=fix_values,
                device=device,
                all_layers=all_layers,
            )

        else:
            fix_inputs = torch.ones(batch_size, len(fix_inputs), dtype=fix_inputs.dtype).to(
                fix_inputs.device
            ) * fix_inputs.unsqueeze(0)
            E_var_cond, _ = _mc_multi_batch(
                model=model,
                f_model=f_model,
                f_E=E_pX,
                f_var=None,
                y_mean=None,
                batch_size=batch_size,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=None,
                device=device,
                all_layers=all_layers,
            )

        if not return_cov:
            E_var_cond = np.diagonal(E_var_cond)
        return None, E_var_cond


def monte_carlo_var_E_pX_Y(
    model: TensorCircuit,
    fix_inputs: int | Iterable,
    sample_distribution: torch.distributions.Distribution,
    n_samples: int,
    batch_size: int = 1000,
    device: str = "cpu",
    return_cov: bool = True,
    all_layers: bool = False,
):
    model.eval()
    with torch.no_grad():
        if n_samples <= batch_size:
            fix_values = torch.randn(fix_inputs.shape).to(device)
            _, var_E_cond = _mc_single_batch(
                model=model,
                f_E=None,
                f_var=E_pX,
                y_mean=None,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=fix_values,
                device=device,
                all_layers=all_layers,
            )

        else:
            y_mean = expectation(model, return_all_outputs=all_layers)

            def get_y_mean(**kwargs):
                return y_mean

            _, var_E_cond = _mc_multi_batch(
                model=model,
                f_E=None,
                f_var=cov_E_pX_Y_batch if return_cov else var_E_pX_Y_batch,
                y_mean=get_y_mean,
                batch_size=batch_size,
                n_samples=n_samples,
                sample_distribution=sample_distribution,
                fix_inputs=fix_inputs,
                fix_values=None,
                device=device,
                all_layers=all_layers,
            )
        return None, var_E_cond

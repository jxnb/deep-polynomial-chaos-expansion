import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
import json

from tensorchaos.circuits import TensorCircuit, build_tensorized_circuit
from tensorchaos.circuits.inference import (
    expectation,
    covariance,
    expectation_conditional_covariances,
    covariance_conditional_expectations,
)
from src.inference_evaluation import mc


def plot_expectations(
    E_monte_carlo, E_model, var_idx, title="model_E", label="E[X]", out_path="./E_X.png"
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
    ax.axhline(E_model[var_idx], color="grey", label=label)
    for i, n_samples in enumerate(mc_samples):
        ax.scatter(
            [n_samples] * E_monte_carlo.shape[0],
            E_monte_carlo[:, i, var_idx],
            c="coral",
            alpha=0.6,
            label="$\\bar{x}_{MC}$",
            zorder=3,
        )

    ax.set_xscale("log")
    ymin, ymax = ax.get_ylim()
    if abs(ymax) - abs(ymin) < 0.0001:
        ax.set_ylim((E_model[var_idx] - 0.005, E_model[var_idx] + 0.005))
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_covariances(
    cov_monte_carlo,
    cov_model_Exy,
    cov_model_cov,
    var_idx,
    title="model_cov",
    label="cov(X, X)",
    out_path="./cov_X.png",
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
    ax.axhline(cov_model_Exy[var_idx], color="grey", label=f"{label} (mode: 'Exy')")
    ax.axhline(cov_model_cov[var_idx], color="grey", linestyle="--", label=f"{label} (mode: 'cov')")
    for i, n_samples in enumerate(mc_samples):
        ax.scatter(
            [n_samples] * cov_monte_carlo.shape[0],
            cov_monte_carlo[:, i, var_idx],
            c="coral",
            alpha=0.6,
            label="$\\text{Var}(x)_{MC}$",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_variances(
    var_monte_carlo,
    var_model_Exy,
    var_model_cov,
    var_idx,
    title="model_Var",
    label="Var(X)",
    out_path="./Var_X.png",
):
    fig, ax = plt.subplots(1, 1, figsize=(8, 5), sharex=True)
    ax.axhline(var_model_Exy[var_idx], color="grey", label=f"{label} (mode: 'Exy')")
    ax.axhline(var_model_cov[var_idx], color="grey", linestyle="--", label=f"{label} (mode: 'cov')")
    for i, n_samples in enumerate(mc_samples):
        ax.scatter(
            [n_samples] * var_monte_carlo.shape[0],
            var_monte_carlo[:, i, var_idx],
            c="coral",
            alpha=0.6,
            label="$\\text{Var}(x)_{MC}$",
            zorder=3,
        )

    ax.set_xscale("log")
    ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[:2], labels[:2])

    fig.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def plot_expectations_all_layers(
    E_monte_carlo,
    E_model,
    var_idx,
    layer_names,
    plot_layer_idx=None,
    plot_sums=None,
    title="model_E",
    label="E[X]",
    out_path="./E_X_all_layers.png",
):
    plot_sums = (
        np.arange(np.max([e.shape[-1] for e in E_model])) if plot_sums is None else plot_sums
    )
    n_cols = len(plot_sums)
    plot_layer_idx = np.arange(len(E_monte_carlo)) if plot_layer_idx is None else plot_layer_idx
    for r in range(E_model[0].shape[1]):
        ax_idx = -1
        fig, ax = plt.subplots(
            len(plot_layer_idx), n_cols, figsize=(8 * n_cols, 5 * len(plot_layer_idx)), sharex=True
        )
        for l, (mc_layer, layer_E) in enumerate(zip(E_monte_carlo, E_model)):
            if l in plot_layer_idx:
                ax_idx += 1
                for ai, a in enumerate(ax[ax_idx]):
                    if l == 0:
                        s_idx = ai
                    else:
                        if ai < len(plot_sums):
                            s_idx = plot_sums[ai]
                        else:
                            s_idx = 10000
                    if layer_E.ndim == 2:
                        layer_out = layer_E[var_idx]
                        mc_layer_out = mc_layer[:, :, var_idx, :]
                    else:
                        layer_out = layer_E[var_idx, r]
                        mc_layer_out = mc_layer[:, :, var_idx, r, :]

                    if s_idx < len(layer_out):
                        a.axhline(layer_out[s_idx], color="grey", label=label)
                        for i, n_samples in enumerate(mc_samples):
                            a.scatter(
                                [n_samples] * mc_layer_out.shape[0],
                                mc_layer_out[:, i, s_idx],
                                c="coral",
                                alpha=0.6,
                                label="$\\bar{x}_{MC}$",
                                zorder=3,
                            )
                        a.set_xscale("log")
                        ymin, ymax = a.get_ylim()
                        if abs(ymax) - abs(ymin) < 0.0001:
                            a.set_ylim((layer_out[s_idx] - 0.005, layer_out[s_idx] + 0.005))
                        title_add = f"(sum {s_idx})"
                        a.set_title(f"{l+1} {layer_names[l]} {title_add} rep {r}")
                        handles, labels = a.get_legend_handles_labels()
                        a.legend(handles[:2], labels[:2])
                    else:
                        a.remove()

        fig.tight_layout()
        figname = out_path.stem
        suffix = out_path.suffix
        figname += f"_r{r}"
        plt.savefig(Path(out_path.parent, figname + suffix), dpi=300)
        plt.close()


def plot_variances_all_layers(
    var_monte_carlo,
    var_model_Exy,
    var_model_cov,
    var_idx,
    layer_names,
    plot_layer_idx=None,
    title="model_Var",
    label="Var(X)",
    out_path="./Var_X_all_layers.png",
    plot_sums=None,
):
    plot_sums = (
        np.arange(np.max([v.shape[-1] for v in var_model_Exy])) if plot_sums is None else plot_sums
    )
    n_cols = len(plot_sums)
    plot_layer_idx = (
        torch.arange(len(var_monte_carlo)) if plot_layer_idx is None else plot_layer_idx
    )
    for r in range(var_model_Exy[0].shape[1]):
        ax_idx = -1
        fig, ax = plt.subplots(
            len(plot_layer_idx), n_cols, figsize=(8 * n_cols, 5 * len(plot_layer_idx)), sharex=True
        )
        for l, (mc_layer, layer_var_exy, layer_var_cov) in enumerate(
            zip(var_monte_carlo, var_model_Exy, var_model_cov)
        ):
            if l in plot_layer_idx:
                ax_idx += 1
                for ai, a in enumerate(ax[ax_idx]):
                    if l == 0:
                        s_idx = ai
                    else:
                        if ai < len(plot_sums):
                            s_idx = plot_sums[ai]
                        else:
                            s_idx = 10000
                    if layer_var_exy.ndim == 2:
                        layer_out_var_exy = layer_var_exy[var_idx]
                        layer_out_var_cov = layer_var_cov[var_idx]
                        mc_layer_out = mc_layer[:, :, var_idx, :]
                    else:
                        layer_out_var_exy = layer_var_exy[var_idx, r]
                        layer_out_var_cov = layer_var_cov[var_idx, r]
                        mc_layer_out = mc_layer[:, :, var_idx, r, :]

                    if s_idx < len(layer_out_var_exy):
                        a.axhline(
                            layer_out_var_exy[s_idx], color="grey", label=f"{label} (mode: 'Exy')"
                        )
                        a.axhline(
                            layer_out_var_cov[s_idx],
                            color="grey",
                            linestyle="--",
                            label=f"{label} (mode: 'cov')",
                        )
                        for i, n_samples in enumerate(mc_samples):
                            a.scatter(
                                [n_samples] * mc_layer.shape[0],
                                mc_layer_out[:, i, s_idx],
                                c="coral",
                                alpha=0.6,
                                label="$\\text{Var}(x)_{MC}$",
                                zorder=3,
                            )

                        a.set_xscale("log")
                        title_add = f"(sum {s_idx})"
                        a.set_title(f"{l+1} {layer_names[l]} {title_add}")
                        handles, labels = a.get_legend_handles_labels()
                        a.legend(handles[:2], labels[:2])
                    else:
                        a.remove()

        fig.tight_layout()
        figname = out_path.stem
        suffix = out_path.suffix
        figname += f"_r{r}"
        plt.savefig(Path(out_path.parent, figname + suffix), dpi=300)
        plt.close()


def inference_end_to_end_test(
    model,
    sample_distribution,
    mc_samples,
    n_sums,
    target="p(X)",
    conditionals=None,
    plot_var=None,
    mc_runs=1,
    batch_size=1000,
    test_cov=False,
    device="cpu",
    plot_E=True,
    plot_variance=True,
    plot_covariance=True,
    save_output=True,
    print_output=True,
    out_dir="./",
    plot_layer_idx=None,
    plot_sums=None,
):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    model_layers = model.get_layers()

    if target == "E[Y]":
        model_E_out = expectation(model, return_all_outputs=True)
        model_Cov_out_Exy = covariance(model, return_all_outputs=True, mode="Exy")
        model_Cov_out_cov = covariance(model, return_all_outputs=True, mode="cov")
        exclude_E = False

    elif target == "E[Y|x]":
        fix_inputs, fix_values = conditionals
        model_E_out = expectation(
            model, fix_inputs=fix_inputs, fix_values=fix_values, return_all_outputs=True
        )
        model_Cov_out_Exy = covariance(
            model, fix_inputs=fix_inputs, fix_values=fix_values, return_all_outputs=True, mode="Exy"
        )
        model_Cov_out_cov = covariance(
            model, fix_inputs=fix_inputs, fix_values=fix_values, return_all_outputs=True, mode="cov"
        )
        exclude_E = False

    elif target == "E[var(Y|X)]":
        fix_inputs, _ = conditionals
        model_Cov_out_Exy = expectation_conditional_covariances(
            model, fix_inputs=fix_inputs, return_all_outputs=True, mode="Exy"
        )
        model_Cov_out_cov = expectation_conditional_covariances(
            model, fix_inputs=fix_inputs, return_all_outputs=True, mode="cov"
        )
        exclude_E = True

    elif target == "var(E[Y|X])":
        fix_inputs, _ = conditionals
        model_Cov_out_Exy = covariance_conditional_expectations(
            model, fix_inputs=fix_inputs, return_all_outputs=True, mode="Exy"
        )
        model_Cov_out_cov = covariance_conditional_expectations(
            model, fix_inputs=fix_inputs, return_all_outputs=True, mode="cov"
        )
        exclude_E = True

    n_inputs = model_Cov_out_Exy[-1].shape[-1]
    n_repetitions = model_Cov_out_Exy[0].shape[1]
    plot_var = np.arange(n_inputs) if plot_var is None else plot_var
    test_var = np.arange(n_inputs)

    model_Var_Exy = [
        np.zeros((len(test_var), n_repetitions, n_sums)) for _ in range(len(model_Cov_out_Exy[:-1]))
    ]
    model_Var_Exy.append(np.zeros((len(test_var), 1)))

    model_Var_cov = deepcopy(model_Var_Exy)
    if not exclude_E:
        model_E = deepcopy(model_Var_Exy)

    n_layers = len(model_Var_Exy)

    for p, tvar in enumerate(test_var):
        if tvar in plot_var:
            pvar = tvar
            plt_idx = pvar
            for i in range(n_layers):

                if not exclude_E:
                    e = model_E_out[i]
                cov = model_Cov_out_Exy[i]
                cov2 = model_Cov_out_cov[i]

                if p == 0:
                    print(f"--- Layer {i} ---")
                    print("Cov1 == Cov2:", torch.allclose(cov, cov2, atol=1e-6))
                    print("covs max diff:", torch.max(cov - cov2).item())

                if not exclude_E:
                    e_np = e.detach().cpu().numpy()
                diag_ax1, diag_ax2 = (1, 2) if cov.ndim == 3 else (2, 3)
                var_np = np.diagonal(
                    cov.detach().cpu().numpy(), axis1=diag_ax1, axis2=diag_ax2
                ).swapaxes(-2, -1)
                var2_np = np.diagonal(
                    cov2.detach().cpu().numpy(), axis1=diag_ax1, axis2=diag_ax2
                ).swapaxes(-2, -1)

                if hasattr(model_layers[i], "scopes"):
                    plt_idx = torch.where(model_layers[i].scopes == pvar)[1][0]

                if i == n_layers - 1:
                    if not exclude_E:
                        model_E[i][p] = e_np[0, plt_idx]
                    model_Var_Exy[i][p] = var_np[0, plt_idx]
                    model_Var_cov[i][p] = var2_np[0, plt_idx]
                else:
                    if not exclude_E:
                        model_E[i][p, ...] = e_np[0, ..., plt_idx]
                    model_Var_Exy[i][p, ...] = var_np[0, ..., plt_idx]
                    model_Var_cov[i][p, ...] = var2_np[0, ..., plt_idx]

    if not exclude_E:
        E_mc = []
    Var_mc = []
    for i in range(n_layers):
        if i == n_layers - 1:
            if not exclude_E:
                E_mc.append(np.zeros((mc_runs, len(mc_samples), len(test_var), 1)))
            Var_mc.append(np.zeros((mc_runs, len(mc_samples), len(test_var), 1)))
        else:
            if not exclude_E:
                E_mc.append(
                    np.zeros((mc_runs, len(mc_samples), len(test_var), n_repetitions, n_sums))
                )
            Var_mc.append(
                np.zeros((mc_runs, len(mc_samples), len(test_var), n_repetitions, n_sums))
            )

    for i, n_samples in enumerate(mc_samples):
        print(f"MC, samples: {n_samples:.0e}")
        runs_iter = tqdm(range(mc_runs)) if mc_runs > 1 else range(mc_runs)

        for j in runs_iter:
            if target == "E[Y]":
                E_mc_layers, var_mc_layers = mc.monte_carlo_pX(
                    model,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=True,
                )
            elif target == "E[Y|x]":
                E_mc_layers, var_mc_layers = mc.monte_carlo_pX_y(
                    model,
                    fix_inputs,
                    fix_values,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=True,
                )
            elif target == "E[var(Y|X)]":
                _, var_mc_layers = mc.monte_carlo_E_var_pX_Y(
                    model,
                    fix_inputs,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=True,
                )

            for p, tvar in enumerate(test_var):
                if tvar in plot_var:
                    pvar = tvar
                    plt_idx = pvar
                    for l in range(n_layers):
                        if not exclude_E:
                            layer_e = E_mc_layers[l]
                        layer_var = var_mc_layers[l]
                        if hasattr(model_layers[l], "scopes"):
                            plt_idx = torch.where(model_layers[l].scopes == pvar)[1][0]
                        if l == n_layers - 1:
                            if not exclude_E:
                                E_mc[l][j, i, p, :] = layer_e[plt_idx]
                            Var_mc[l][j, i, p, :] = layer_var[plt_idx]
                        else:
                            if not exclude_E:
                                E_mc[l][j, i, p, ...] = layer_e[..., plt_idx]
                            Var_mc[l][j, i, p, ...] = layer_var[..., plt_idx]

    if print_output:
        print()
        print("=== RESULTS ===")
        print("conditionals: ", conditionals)
        print("scope_size: ", scope_size)
        print()

        for r, rep_scopes in enumerate(model.input_layer.scopes):
            print(f"=== Circuit repetition {r + 1} ===")
            for s in rep_scopes:
                if conditionals is None:
                    desc = "free"
                else:
                    fix_inputs, fix_values = conditionals
                    if torch.all(torch.isin(s, fix_inputs)):
                        desc = "fixed"
                    elif torch.any(torch.isin(s, fix_inputs)):
                        desc = "mixed"
                    else:
                        desc = "free"
                s = s.detach().cpu().numpy()

                print(f"---- S{s} ({desc}) ----")
                if not exclude_E:
                    print(f"{target}: {model_E[0][s[0], r, 0]:.4f}")
                    print(f"{target} MC: {E_mc[0][..., s[0], r, 0].mean(axis=(0, 1)):.4f}")
                    print()
                if target in ["E[Y]", "E[Y|X]"]:
                    vartarget = f"var({target[2:-1]})"
                else:
                    vartarget = target
                print(f"{vartarget} (Exy): {model_Var_Exy[0][s[0], r, 0]:.4f}")
                print(f"{vartarget} (cov): {model_Var_cov[0][s[0], r, 0]:.4f}")
                print(f"{vartarget} MC: {Var_mc[0][..., s[0], r, 0].mean(axis=(0, 1)):.4f}")
                print()

    if save_output:
        data_out_dir = Path(out_dir, "output")
        data_out_dir.mkdir(parents=True, exist_ok=True)
        for i in range(n_layers):
            if not exclude_E:
                np.save(Path(data_out_dir, f"E_{i}"), model_E[i])
                np.save(Path(data_out_dir, f"E_MC_{i}"), E_mc[i])
            np.save(Path(data_out_dir, f"var_Exy_{i}"), model_Var_Exy[i])
            np.save(Path(data_out_dir, f"var_cov_{i}"), model_Var_cov[i])
            np.save(Path(data_out_dir, f"var_MC_{i}"), Var_mc[i])

    if plot_E and not exclude_E:
        print("plot E...")
        for p, pvar in enumerate(plot_var):
            print(f"{p+1}/{len(plot_var)}...")
            if target == "E[Y]":
                title = f"model_E_Y_{pvar}"
                label = target
                fig_path = Path(out_dir, f"E_Y_{pvar}.png")
            elif target == "E[Y|x]":
                title = f"model_E_Y_x_{pvar}"
                label = target
                fig_path = Path(out_dir, f"E_Y_x_{pvar}.png")

            plot_expectations_all_layers(
                E_mc,
                model_E,
                pvar,
                layer_names=[type(l).__name__ for l in model_layers],
                title=title,
                label=label,
                out_path=fig_path,
                plot_layer_idx=plot_layer_idx,
                plot_sums=plot_sums,
            )

    if plot_variance:
        print("plot var...")
        for p, pvar in enumerate(plot_var):
            print(f"{p+1}/{len(plot_var)}...")

            if target == "E[Y]":
                title = f"model_var_Y_{pvar}"
                label = "var(Y)"
                fig_path = Path(out_dir, f"var_Y_{pvar}.png")
            elif target == "E[Y|x]":
                title = f"model_var_Y_x_{pvar}"
                label = "var(Y|x)"
                fig_path = Path(out_dir, f"var_Y_x_{pvar}.png")
            elif target == "E[var(Y|X)]":
                title = f"model_E_var_Y_X_{pvar}"
                label = target
                fig_path = Path(out_dir, f"E_var_Y_X_{pvar}.png")
            elif target == "var(E[Y|X])":
                title = f"model_var_E_Y_X_{pvar}"
                label = target
                fig_path = Path(out_dir, f"var_E_Y_X_{pvar}.png")

            plot_variances_all_layers(
                Var_mc,
                model_Var_Exy,
                model_Var_cov,
                pvar,
                layer_names=[type(l).__name__ for l in model_layers],
                title=title,
                label=label,
                out_path=fig_path,
                plot_layer_idx=plot_layer_idx,
                plot_sums=plot_sums,
            )


def inference_evaluation(
    model: TensorCircuit,
    sample_distribution,
    mc_samples,
    target,
    conditionals=None,
    plot_var=None,
    mc_runs=1,
    batch_size=1000,
    test_cov=False,
    device="cpu",
    plot_E=True,
    plot_variance=True,
    plot_covariance=True,
    save_output=True,
    print_output=True,
    out_dir="./",
):

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    if save_output:
        data_out_dir = Path(out_dir, "output")
        data_out_dir.mkdir(parents=True, exist_ok=True)

    if target == "E[Y]":
        model_E = expectation(model).detach().cpu().numpy()
        model_Cov_Exy = covariance(model, mode="Exy").detach().cpu().numpy()
        model_Cov_cov = covariance(model, mode="cov").detach().cpu().numpy()
        exclude_E = False

    elif target == "E[Y|x]":
        fix_inputs, fix_values = conditionals
        model_E = (
            expectation(model, fix_inputs=fix_inputs, fix_values=fix_values).detach().cpu().numpy()
        )
        model_Cov_Exy = (
            covariance(model, fix_inputs=fix_inputs, fix_values=fix_values, mode="Exy")
            .detach()
            .cpu()
            .numpy()
        )
        model_Cov_cov = (
            covariance(model, fix_inputs=fix_inputs, fix_values=fix_values, mode="cov")
            .detach()
            .cpu()
            .numpy()
        )
        exclude_E = False

    elif target == "E[var(Y|X)]":
        fix_inputs, fix_values = conditionals
        model_Cov_Exy = (
            expectation_conditional_covariances(model, fix_inputs=fix_inputs, mode="Exy")
            .detach()
            .cpu()
            .numpy()
        )
        model_Cov_cov = (
            expectation_conditional_covariances(model, fix_inputs=fix_inputs, mode="cov")
            .detach()
            .cpu()
            .numpy()
        )
        exclude_E = True

    elif target == "var(E[Y|X])":
        fix_inputs, fix_values = conditionals
        model_Cov_Exy = (
            covariance_conditional_expectations(model, fix_inputs=fix_inputs, mode="Exy")
            .detach()
            .cpu()
            .numpy()
        )
        model_Cov_cov = (
            covariance_conditional_expectations(model, fix_inputs=fix_inputs, mode="cov")
            .detach()
            .cpu()
            .numpy()
        )
        exclude_E = True

    n_inputs = model_Cov_Exy.shape[1]
    if not test_cov:
        model_Cov_Exy = np.diagonal(model_Cov_Exy, axis1=1, axis2=2)[None, ...]
        model_Cov_cov = np.diagonal(model_Cov_cov, axis1=1, axis2=2)[None, ...]

    if not exclude_E:
        E_mc = np.zeros((mc_runs, len(mc_samples), *model_E.shape[1:]))
    Var_mc = np.zeros((mc_runs, len(mc_samples), *model_Cov_Exy.shape[1:]))

    for i, n_samples in enumerate(mc_samples):
        print(f"MC, samples: {n_samples:.0e}")
        runs_iter = tqdm(range(mc_runs)) if mc_runs > 1 else range(mc_runs)

        for j in runs_iter:
            if target == "E[Y]":
                e_mc, var_mc = mc.monte_carlo_pX(
                    model,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=False,
                )

            elif target == "E[Y|x]":
                e_mc, var_mc = mc.monte_carlo_pX_y(
                    model,
                    fix_inputs,
                    fix_values,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=False,
                )

            elif target == "E[var(Y|X)]":
                _, var_mc = mc.monte_carlo_E_var_pX_Y_2(
                    model,
                    fix_inputs,
                    sample_distribution,
                    n_samples,
                    batch_size=batch_size,
                    device=device,
                    return_cov=test_cov,
                    all_layers=False,
                )

            if not exclude_E:
                E_mc[j, i, :] = e_mc
            Var_mc[j, i, ...] = var_mc

        if save_output:
            if not exclude_E:
                np.save(Path(data_out_dir, f"E_MC_{n_samples:.0e}"), E_mc[:, i : i + 1, :])
            fname = "cov" if test_cov else "var"
            np.save(Path(data_out_dir, f"{fname}_MC_{n_samples:.0e}"), Var_mc[:, i : i + 1, ...])

    if conditionals:
        if fix_inputs is not None:
            fix_inputs = fix_inputs.detach().cpu().numpy()
        if fix_values is not None:
            fix_values = fix_values.detach().cpu().numpy()

    if not exclude_E:
        E_mse = np.mean((E_mc - model_E[None, ...]) ** 2, axis=(0, 2))
    cov_Exy_mse = np.mean((Var_mc - model_Cov_cov[None, ...]) ** 2, axis=(0, 2, 3))
    cov_cov_mse = np.mean((Var_mc - model_Cov_cov[None, ...]) ** 2, axis=(0, 2, 3))

    if print_output:
        print()
        print("=== RESULTS ===")
        print(target)
        if conditionals:
            print("conditionals: ", conditionals)
        print()
        for n, n_samples in enumerate(mc_samples):
            print(f"--- MC samples: {n_samples:.0E} ---")
            if not exclude_E:
                print(f"E mse: {E_mse[n]}")
            print(f"cov (Exy) mse: {cov_Exy_mse[n]}")
            print(f"cov (cov) mse: {cov_cov_mse[n]}")
            print()

    if save_output:
        if not exclude_E:
            np.save(Path(data_out_dir, "E"), model_E)
        fname = "cov" if test_cov else "var"
        np.save(Path(data_out_dir, f"{fname}_Exy"), model_Cov_Exy)
        np.save(Path(data_out_dir, f"{fname}_cov"), model_Cov_cov)
        if conditionals:
            if fix_inputs is not None:
                np.save(Path(data_out_dir, "fix_inputs"), fix_inputs)
            if fix_values is not None:
                np.save(Path(data_out_dir, "fix_values"), fix_values)

    plot_var = np.arange(n_inputs) if plot_var is None else plot_var
    if plot_E and not exclude_E:
        print("plot E...")
        for p, pvar in enumerate(plot_var):
            print(f"{p+1}/{len(plot_var)}...")
            title = f"E[Y{pvar}]"
            if target == "E[Y]":
                label = target
                fig_path = Path(out_dir, f"E_Y_{pvar}.png")
            elif target == "E[Y|x]":
                title += f" conditionals: {fix_inputs}, {fix_values}"
                label = target
                fig_path = Path(out_dir, f"E_Y_x_{pvar}.png")
            plot_expectations(
                E_mc, model_E[0, ...], pvar, title=title, label=label, out_path=fig_path
            )
        print()

    if plot_variance:
        print("plot var...")
        for p, pvar in enumerate(plot_var):
            print(f"{p+1}/{len(plot_var)}...")
            title = f"var(Y{pvar})"
            if target == "E[Y]":
                label = "var(Y)"
                fig_path = Path(out_dir, f"var_Y_{pvar}.png")
            elif target == "E[Y|x]":
                title += f" conditionals: {fix_inputs}, {fix_values}"
                label = "var(Y|x)"
                fig_path = Path(out_dir, f"var_Y_x_{pvar}.png")
            elif target == "E[var(Y|X)]":
                title += f" conditionals: {fix_inputs}"
                label = target
                fig_path = Path(out_dir, f"E_var_Y_X_{pvar}.png")
            elif target == "var(E[Y|X])":
                title += f" conditionals: {fix_inputs}"
                label = target
                fig_path = Path(out_dir, f"var_E_Y_X_{pvar}.png")
            if test_cov:
                plot_variances(
                    np.diagonal(Var_mc, axis1=2, axis2=3),
                    np.diagonal(model_Cov_Exy[0, ...]),
                    np.diagonal(model_Cov_cov[0, ...]),
                    pvar,
                    title=title,
                    label=label,
                    out_path=fig_path,
                )
            else:
                plot_variances(
                    Var_mc[..., 0, :],
                    model_Cov_Exy[0, 0, ...],
                    model_Cov_cov[0, 0, ...],
                    pvar,
                    title=title,
                    label=label,
                    out_path=fig_path,
                )
        print()

    if plot_covariance and test_cov:
        print("plot cov antidiag...")
        for p, pvar in enumerate(plot_var):
            print(f"{p+1}/{len(plot_var)}...")
            title = f"Cov(Y{pvar}, Y{Var_mc.shape[-1] - 1 - pvar})"
            if target == "E[Y]":
                label = "cov(Y)"
                fig_path = Path(out_dir, f"cov_Y_{pvar}_antidiag.png")
            elif target == "E[Y|x]":
                title += f" conditionals: {fix_inputs}, {fix_values}"
                label = "cov(Y|x)"
                fig_path = Path(out_dir, f"cov_Y_x_{pvar}_antidiag.png")
            elif target == "E[var(Y|X)]":
                title += f" conditionals: {fix_inputs}"
                label = target
                fig_path = Path(out_dir, f"E_cov_Y_X_{pvar}_antidiag.png")
            elif target == "var(E[Y|X])":
                title += f" conditionals: {fix_inputs}"
                label = target
                fig_path = Path(out_dir, f"cov_E_Y_X_{pvar}_antidiag.png")
            plot_variances(
                np.diagonal(np.flip(Var_mc, axis=3), axis1=2, axis2=3),
                np.diagonal(np.fliplr(model_Cov_Exy[0, ...])),
                np.diagonal(np.fliplr(model_Cov_cov[0, ...])),
                pvar,
                title=title,
                label=label,
                out_path=fig_path,
            )
        print()


def convert_to_json_serializable(x):
    return x.cpu().numpy().tolist() if x is not None else None


if __name__ == "__main__":
    mc_samples = [10**i for i in range(5, 8)]
    mc_runs = 30
    batch_size = 10000

    test_covariance = True
    input_dist = "normal"

    print_output = True
    plot_E = True
    plot_variance = True
    plot_covariance = True

    test_01 = True
    test_02 = True
    test_03 = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    np.random.seed(42)
    torch.manual_seed(42)

    results_dir = "./inference_test"
    fix_inputs = torch.tensor([0, 1, 2, 5], dtype=torch.long).to(device)
    fix_values = torch.empty(len(fix_inputs)).uniform_(-2, 2).to(device)
    conditionals = torch.stack([fix_inputs, fix_values], dim=0)

    input_dist = "normal"

    n_sums = 3
    n_inputs = 8
    max_order = 3
    scope_size = 2
    plot_layer_idx = None
    plot_sums = None

    plot_var = np.arange(n_inputs)
    deep_chaos_model, _ = build_tensorized_circuit(
        input_shape=(n_inputs, 1),
        n_outputs=n_inputs,
        n_sums=n_sums,
        scope_size=scope_size,
        circuit_depth=None,
        input_layer="pce-hermite" if input_dist == "normal" else "pce-legendre",
        product_layer="hadamard",
        region_graph="binary-tree-td",
        max_order=max_order,
        batch_norm=False,
        param_init=None,
        pce_var_decay=(1 / torch.e),
        pce_var_rank="sum",
        rng_seed=None,
        dtype=torch.float32,
    )

    if input_dist == "normal":
        input_distribution = torch.distributions.MultivariateNormal(
            torch.zeros(n_inputs), covariance_matrix=torch.eye(n_inputs)
        )
    elif input_dist == "uniform":
        input_distribution = torch.distributions.Uniform(
            torch.ones(n_inputs) * (-1), torch.ones(n_inputs)
        )

    if test_01:
        # TEST 1: NO CONDITIONED VARIABLES
        print("=== INFERENCE TEST 01 ===")
        print("E[Y], Var(Y)\n")

        test_name = "E_Y"
        out_dir = Path(results_dir, test_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        conditionals = None

        model = deepcopy(deep_chaos_model)

        with open(Path(out_dir, "test_config.json"), "w") as f:
            json.dump(
                dict(
                    n_inputs=n_inputs,
                    n_sums=n_sums,
                    max_order=max_order,
                    scope_size=scope_size,
                    test_cov=test_covariance,
                    input_dist=input_dist,
                    fix_inputs=None,
                    fix_values=None,
                    scopes=convert_to_json_serializable(model.input_layer.scopes[0]),
                ),
                f,
                indent=4,
            )

        model.to(device)
        model.eval()

        sample_dist = deepcopy(input_distribution)
        inference_evaluation(
            model,
            sample_dist,
            mc_samples,
            target="E[Y]",
            conditionals=conditionals,
            plot_var=plot_var,
            mc_runs=mc_runs,
            batch_size=batch_size,
            test_cov=test_covariance,
            device=device,
            plot_E=plot_E,
            plot_variance=plot_variance,
            plot_covariance=plot_covariance,
            print_output=print_output,
            out_dir=out_dir,
        )

    if test_02:
        # TEST 2: CONDITIONED VARIABLES, FIXED VALUES
        print("=== INFERENCE TEST 02 ===")
        print("E(Y | X_c = x), Var(Y | X_c = x)\n")

        test_name = "E_Y_x"
        out_dir = Path(results_dir, test_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        conditionals = (fix_inputs, fix_values)

        print(convert_to_json_serializable(fix_inputs), convert_to_json_serializable(fix_values))

        model = deepcopy(deep_chaos_model)

        with open(Path(out_dir, "test_config.json"), "w") as f:
            json.dump(
                dict(
                    n_inputs=n_inputs,
                    n_sums=n_sums,
                    max_order=max_order,
                    scope_size=scope_size,
                    test_cov=test_covariance,
                    input_dist=input_dist,
                    fix_inputs=convert_to_json_serializable(fix_inputs),
                    fix_values=convert_to_json_serializable(fix_values),
                    scopes=convert_to_json_serializable(model.input_layer.scopes[0]),
                ),
                f,
                indent=4,
            )

        model.to(device)
        model.eval()

        sample_dist = deepcopy(input_distribution)
        inference_evaluation(
            model,
            sample_dist,
            mc_samples,
            target="E[Y|x]",
            conditionals=conditionals,
            plot_var=plot_var,
            mc_runs=mc_runs,
            batch_size=batch_size,
            test_cov=test_covariance,
            device=device,
            plot_E=plot_E,
            plot_variance=plot_variance,
            plot_covariance=plot_covariance,
            print_output=print_output,
            out_dir=out_dir,
        )

    if test_03:
        # TEST 3: CONDITIONED VARIABLES, NO FIXED VALUES
        print("=== INFERENCE TEST 03 ===")
        print("E_m(Y | X_c), E_m[Var(Y | X_c)]\n")

        test_name = "E_Y_X"
        out_dir = Path(results_dir, test_name)
        out_dir.mkdir(parents=True, exist_ok=True)

        conditionals = (fix_inputs, None)
        model = deepcopy(deep_chaos_model)

        with open(Path(out_dir, "test_config.json"), "w") as f:
            json.dump(
                dict(
                    n_inputs=n_inputs,
                    n_sums=n_sums,
                    max_order=max_order,
                    scope_size=scope_size,
                    test_cov=test_covariance,
                    input_dist=input_dist,
                    fix_inputs=convert_to_json_serializable(fix_inputs),
                    fix_values=None,
                    scopes=convert_to_json_serializable(model.input_layer.scopes[0]),
                ),
                f,
                indent=4,
            )

        model.to(device)
        model.eval()
        sample_dist = deepcopy(input_distribution)
        inference_evaluation(
            model,
            sample_dist,
            mc_samples,
            target="E[var(Y|X)]",
            conditionals=conditionals,
            plot_var=plot_var,
            mc_runs=mc_runs,
            batch_size=10000,
            test_cov=test_covariance,
            device=device,
            plot_E=plot_E,
            plot_variance=plot_variance,
            plot_covariance=plot_covariance,
            print_output=print_output,
            out_dir=out_dir,
        )

"""This module includes all functions to reproduce the paper experiments."""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from tqdm import tqdm
from argparse import ArgumentParser
from functools import partial
import json
from copy import deepcopy
from pathlib import Path
from timeit import default_timer
from collections.abc import Callable

from tensorchaos.circuits import build_tensorized_circuit
from src.baselines.pce import build_pce, forward_neighbor_basis_selection
from src.baselines.cnn import UNet
from src.baselines.mlp import MLP
from src.datasets.test_functions import UQFunc
from src.config import EXPERIMENTS, DarcyFlowConfig, SteadyStateDiffusionConfig


def train_loop(
    model,
    optimizer,
    criterion,
    n_epochs,
    train_loader,
    val_loader,
    early_stopping=True,
    patience=10,
    early_stopping_grace_period=100,
    nested_progbar=False
):
    best_model_weights = deepcopy(model.state_dict())
    current_best_epoch = 0
    best_val_loss = torch.inf

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    train_loss_list = []
    val_loss_list = []

    es_patience_count = 0
    es_target = torch.inf

    prog_bar = tqdm(range(n_epochs), leave=not nested_progbar)
    for epoch_count in prog_bar:
        model.train()
        running_loss = 0.0
        for idx, (batch_x, batch_y) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            objective = criterion(outputs, batch_y)
            loss = objective.item()
            running_loss += loss

            objective.backward()
            optimizer.step()

        train_loss = running_loss / len(train_loader)

        # --- validation step ---
        model.eval()
        with torch.no_grad():
            val_loss = 0.0
            for idx, (batch_x, batch_y) in enumerate(val_loader):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()
            val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_model_weights = deepcopy(model.state_dict())
            best_val_loss = val_loss
            current_best_epoch = epoch_count

        pbar_dict = {"loss": train_loss}
        if val_loader:
            pbar_dict["val_loss"] = val_loss
            pbar_dict["val_loss_best"] = best_val_loss
            pbar_dict["best_ep"] = current_best_epoch
        prog_bar.set_postfix(**pbar_dict)

        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)

        if early_stopping:
            if val_loss < es_target:
                es_patience_count = 0
                es_target = val_loss
            else:
                if epoch_count >= early_stopping_grace_period:
                    es_patience_count += 1
                    if es_patience_count > patience:
                        break

    return best_model_weights, train_loss_list, val_loss_list


def model_selection(
    n_warmup_epochs, n_trials, model, optimizer, criterion, train_loader, val_loader
):
    best_model_weights = None
    best_val_loss = torch.inf
    model_seed = None
    best_trial = None

    rng = np.random.default_rng()
    test_seeds = rng.integers(0, 2**32 - 1, n_trials - 1)
    print("Starting weight initialization...")
    for r in tqdm(range(n_trials)):
        if r == 0:
            run_seed = None
        else:
            run_seed = test_seeds[r - 1]
            torch.manual_seed(run_seed)
            model.reset_parameters()
            optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)

        model_weights, train_loss_list, val_loss_list = train_loop(
            model,
            optimizer,
            criterion,
            n_epochs=n_warmup_epochs,
            train_loader=train_loader,
            val_loader=val_loader,
            early_stopping=False,
            nested_progbar=True
        )

        val_loss = val_loss_list[-1]

        if r == 0 or val_loss < best_val_loss:
            if val_loss < best_val_loss:
                best_val_loss = val_loss
            best_model_weights = deepcopy(model.state_dict())
            model_seed = run_seed
            best_trial = r + 1

    print(f"Best trial: {best_trial} (val_loss: {best_val_loss})")
    model.load_state_dict(best_model_weights)
    optimizer = optimizer.__class__(model.parameters(), **optimizer.defaults)
    return model_seed


def run_experiment(
    model: partial[torch.nn.Module] | partial[Callable],
    optimizer: partial[torch.optim.Optimizer],
    dataset: partial[Callable] | UQFunc,
    criterion,
    n_epochs: int | None,
    batch_size: int | None,
    results_dir: str | Path,
    n_train: None,
    n_val: None,
    n_test: None,
    scale_targets: bool = True,
    early_stopping: bool = True,
    patience: int = 10,
    early_stopping_grace_period: int = 100,
    basis_selection: str | None = None,
    n_model_inits: int | None = None,
    model_init_epochs: int = 10,
    store_model: bool = True,
    n_runs: int = 1,
    load_dataset_to_device: bool = True,
    dataset_seed: int | None = None,
    torch_seed: int | None = None,
    device: str | None = None,
    plot_results=True,
    dtype=torch.float32,
):

    model_init_epochs = 0 if model_init_epochs is None else model_init_epochs
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results dir: {str(results_dir)}")
    dataset_device = device if load_dataset_to_device else None

    if dataset_seed is None:
        dataset_seed = np.random.randint(2**32 - 1 - n_runs)
    if torch_seed is None:
        torch_seed = np.random.randint(2**32 - 1 - n_runs)

    ## N iterations
    for r in range(n_runs):
        print(f"Experiment run {r + 1}/{n_runs}")
        dataset_seed += r
        torch_seed += r

        out_dir = Path(results_dir, f"run_{str(r + 1).zfill(3)}")
        out_dir.mkdir(parents=True, exist_ok=True)

        np.save(Path(out_dir, "dataset_seed"), dataset_seed)

        if isinstance(dataset, UQFunc):
            n_samples = n_train + n_val + n_test
            x, y = dataset.sample(n_samples, transform_x=True, seed=dataset_seed)
            x_train, y_train = x[:n_train], y[:n_train]
            x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
            x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]
        else:
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = dataset(rng_seed=dataset_seed)

        if len(y_train.shape) < 2:
            y_train = y_train.reshape(-1, 1)
            y_val = y_val.reshape(-1, 1)
            y_test = y_test.reshape(-1, 1)

        x_shape = x_train.shape[1:]
        y_shape = y_train.shape[1:]

        if scale_targets:
            y_train = y_train.reshape(y_train.shape[0], -1)
            y_val = y_val.reshape(y_val.shape[0], -1)
            y_test = y_test.reshape(y_test.shape[0], -1)

            scaler = StandardScaler()
            y_train = scaler.fit_transform(y_train)
            y_val = scaler.transform(y_val)
            y_test = scaler.transform(y_test)

        n_outputs = y_train[0].size

        # === DEEPPCE, NEURAL NETWORK MODELS ===
        if model.func in [build_tensorized_circuit, UNet, MLP]:
            model_instantiated = model(input_shape=x_shape, n_outputs=n_outputs, dtype=dtype)
            if isinstance(model_instantiated, tuple):
                model_instantiated = model_instantiated[0]
            torch_generator = torch.manual_seed(torch_seed)
            torch.save(torch_seed, str(Path(out_dir, "torch_seed.pt")))
            model_instantiated.reset_parameters()

            if isinstance(model_instantiated, UNet):
                y_train = y_train.reshape(-1, *y_shape)
                y_val = y_val.reshape(-1, *y_shape)
                y_test = y_test.reshape(-1, *y_shape)
            else:
                x_train = x_train.reshape(x_train.shape[0], -1)
                x_val = x_val.reshape(x_val.shape[0], -1)
                x_test = x_test.reshape(x_test.shape[0], -1)

            train_loader = to_dataloader(
                x_train,
                y_train,
                batch_size,
                to_device=dataset_device,
                shuffle=True,
                drop_last=True,
                dtype=dtype,
                generator=torch_generator,
            )

            val_loader = to_dataloader(
                x_val, y_val, batch_size=100, to_device=dataset_device, shuffle=False, dtype=dtype
            )

            model_instantiated.to(device)
            optimizer_instantiated = optimizer(params=model_instantiated.parameters())

            # === Run training ===
            train_start = default_timer()

            # Initialize weights with multiple seeds and choose best one
            if n_model_inits is not None and n_model_inits > 1:
                timer = default_timer()
                model_selection(
                    n_warmup_epochs=model_init_epochs,
                    n_trials=n_model_inits,
                    model=model_instantiated,
                    optimizer=optimizer_instantiated,
                    criterion=criterion,
                    train_loader=train_loader,
                    val_loader=val_loader,
                )
                duration = default_timer() - timer
                print(f"weight init time: {duration / 60} min")

            model_weights, train_loss, val_loss = train_loop(
                model_instantiated,
                optimizer_instantiated,
                criterion,
                n_epochs - model_init_epochs,
                train_loader,
                val_loader,
                early_stopping=early_stopping,
                patience=patience,
                early_stopping_grace_period=early_stopping_grace_period,
            )
            model_instantiated.load_state_dict(model_weights)

            best_epoch = np.argmin(val_loss)
            print(f"best model: epoch: {best_epoch}, val loss: {val_loss[best_epoch]:.4f}")
            train_time = default_timer() - train_start
            print(f"train time: {train_time:.0f} sec")

            if store_model:
                torch.save(model_instantiated.state_dict(), Path(out_dir, f"model_trained.pt"))

            if plot_results:
                # Plot train loss
                fig, ax = plt.subplots(1, 2, figsize=(12, 5))
                ax[0].plot(train_loss, color="plum")
                ax[0].set_title("Train")
                ax[0].set_yscale("log")
                ax[1].plot(val_loss, color="lightseagreen")
                ax[1].set_title("Val")
                ax[1].set_yscale("log")
                fig.suptitle(f"Loss")
                fig.tight_layout()
                plt.savefig(Path(out_dir, "train_loss.png"), dpi=300)
                plt.close()

            with open(Path(out_dir, "train_time.json"), "w") as f:
                json.dump({f"train_time": train_time}, f, indent=4)

            # === Run evaluation on test set ===

            # Evaluate model
            test_loader = to_dataloader(
                x_test, y_test, batch_size=100, to_device=dataset_device, shuffle=False
            )

            test_predictions = np.zeros(y_test.shape)
            test_loss = 0.0
            start_idx = 0
            model_instantiated.eval()
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    if device:
                        batch_x = batch_x.to(device)
                        batch_y = batch_y.to(device)
                    outputs = model_instantiated(batch_x)

                    test_predictions[start_idx : start_idx + len(batch_x), ...] = (
                        outputs.detach().cpu().numpy()
                    )
                    start_idx += len(batch_x)
                    test_loss += criterion(outputs, batch_y).item()

                test_loss /= len(test_loader)


        # == BASELINE PCE ===
        elif model.func == build_pce:

            x_train = x_train.reshape(x_train.shape[0], -1)
            x_val = x_val.reshape(x_val.shape[0], -1)
            x_test = x_test.reshape(x_test.shape[0], -1)

            # === Fit PCE ===
            print("fit pce ...")
            expansion_start = default_timer()
            model_instantiated = model(n_inputs=np.prod(x_shape), n_outputs=n_outputs, dtype=dtype)
            time_expansion = default_timer() - expansion_start
            start = default_timer()
            model_instantiated.to(device)
            if basis_selection == "fnbs":
                x_train = torch.tensor(x_train, dtype=dtype).to(device)
                y_train = torch.tensor(y_train, dtype=dtype).to(device)
                x_val = torch.tensor(x_val, dtype=dtype).to(device)
                y_val = torch.tensor(y_val, dtype=dtype).to(device)
                model_instantiated = forward_neighbor_basis_selection(
                    model_instantiated, x_train, y_train, x_val, y_val, T=3
                )
                train_out = model_instantiated.predict(x_train).cpu().numpy()
            else:
                x_train = torch.tensor(np.concatenate([x_train, x_val], axis=0), dtype=dtype)
                x_train = x_train.to(device)
                y_train = torch.tensor(np.concatenate([y_train, y_val], axis=0), dtype=dtype)
                y_train = y_train.to(device)
                train_out = (
                    model_instantiated.fit(x_train, y_train, batch_size=batch_size).cpu().numpy()
                )
            time_fit = default_timer() - start

            train_time = time_expansion + time_fit
            train_loss = np.mean((train_out - y_train.cpu().numpy()) ** 2)

            print("train loss:", train_loss)

            # === Run evaluation on test set ===
            start = default_timer()
            test_predictions = (
                model_instantiated.predict(torch.tensor(x_test, dtype=DTYPE).to(device))
                .cpu()
                .numpy()
            )

            test_loss = np.mean((test_predictions - y_test) ** 2)



        if scale_targets:
            if isinstance(model_instantiated, UNet):
                y_test = y_test.reshape(y_test.shape[0], -1)
                test_predictions = test_predictions.reshape(test_predictions.shape[0], -1)
            y_test = scaler.inverse_transform(y_test)
            test_predictions = scaler.inverse_transform(test_predictions)

        # compute mse for unscaled output data
        mse = np.mean((test_predictions - y_test) ** 2)
        relative_mse = mse / (np.mean(y_test**2))

        y_test = y_test.reshape(y_test.shape[0], *y_shape)
        test_predictions = test_predictions.reshape(test_predictions.shape[0], *y_shape)

        # plot results
        if plot_results:
            if len(y_shape) < 2:
                test_predictions = test_predictions.flatten()
                y_test = y_test.flatten()
                n_rows = 1
                n_cols = 2
                fig, ax = plt.subplots(n_rows, 2, figsize=(8, 4.5), sharex="col")

                ax[0].hist(
                    y_test, bins=50, density=True, color="lightgrey", alpha=0.5, label="true"
                )
                ax[0].hist(
                    test_predictions, density=True, bins=50, color="coral", alpha=0.5, label="pred"
                )
                ax[0].set_ylabel("p(y)")
                ax[0].set_xlim(y_test.min() - y_test.std(), y_test.max() + y_test.std())
                ax[0].set_title(f"Output distribution")
                ax[0].set_xlabel("y")

                ax[0].legend()
                ax[1].scatter(y_test, test_predictions, s=10, alpha=0.5, color="purple")
                ax[1].axline(
                    (y_test.min(), y_test.min()), slope=1, linewidth=1, color="grey", linestyle="--"
                )
                ax[1].set_ylabel("y pred")
                ax[1].set_ylim(y_test.min() - y_test.std(), y_test.max() + y_test.std())
                ax[1].set_title("Predictions")
                ax[1].set_xlabel("y true")

                fig.tight_layout()
                plt.savefig(Path(out_dir, "test_outputs.png"), dpi=300)
                plt.close()

            else:
                x_test = x_test.reshape(x_test.shape[0], *x_shape)

                n_rows, n_cols = (5, 3)
                xvmin, xvmax = (np.percentile(x_test, 1), np.percentile(x_test, 99))
                yvmin, yvmax = (np.percentile(y_test, 1), np.percentile(y_test, 99))
                fig, ax = plt.subplots(
                    n_rows, n_cols, sharex=True, sharey=True, figsize=(2.5 * n_cols, 2.5 * n_rows)
                )
                rand_idx = np.random.choice(np.arange(len(y_test)), size=n_rows, replace=False)
                for i, idx in enumerate(rand_idx):
                    ximg = ax[i, 0].imshow(x_test[idx, ...], vmin=xvmin, vmax=xvmax, cmap="plasma")
                    if i == 0:
                        ax[i, 0].set_title("input")
                    fig.colorbar(ximg, ax=ax[i, 0], location="left", shrink=0.6)

                    ytrueimg = ax[i, 1].imshow(y_test[idx, ...], vmin=yvmin, vmax=yvmax)
                    if i == 0:
                        ax[i, 1].set_title("true")
                    fig.colorbar(ytrueimg, ax=ax[i, 1], location="right", shrink=0.6)
                    ypredimg = ax[i, 2].imshow(test_predictions[idx, ...], vmin=yvmin, vmax=yvmax)
                    if i == 0:
                        ax[i, 2].set_title("pred")
                    fig.colorbar(ypredimg, ax=ax[i, 2], location="right", shrink=0.6)

                for a in ax.flatten():
                    a.set_xticklabels([])
                    a.set_xticks([])
                    a.set_yticklabels([])
                    a.set_yticks([])

                fig.tight_layout()
                plt.savefig(Path(out_dir, "test_outputs.png"), dpi=300)
                plt.close()

        print()
        pheading = f"########### Test results {out_dir.name} ###########"
        print(pheading)
        print(f"Test loss: {test_loss:.4f}")
        print(f"MSE: {mse:.8f}")
        print(f"Relative MSE: {relative_mse:.8f}")
        print("#" * len(pheading))
        print()

    print(f"=== Experiment finished ===")


def to_dataloader(x, y, batch_size, to_device=None, shuffle=True, dtype=torch.float32, **kwargs):
    x_tensor = x if type(x) == torch.Tensor else torch.tensor(x, dtype=dtype)
    y_tensor = y if type(y) == torch.Tensor else torch.tensor(y, dtype=dtype)
    if to_device:
        x_tensor = x_tensor.to(to_device)
        y_tensor = y_tensor.to(to_device)
    dataset = TensorDataset(x_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, **kwargs)
    return dataloader


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("experiment", type=str)
    args = parser.parse_args()
    DTYPE = torch.float32

    experiment_dir = Path("experiments")
    experiment = args.experiment

    try:
        cfg = EXPERIMENTS[experiment]
    except KeyError as e:
        msg = "\n".join(sorted([k for k in EXPERIMENTS.keys()]))
        raise Exception("Available experiments:\n" + msg)

    results_dir = Path(experiment_dir, experiment)
    results_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(cfg["dataset"], (DarcyFlowConfig, SteadyStateDiffusionConfig)):
        dataset = cfg["dataset"].build_factory()
        n_train, n_val, n_test = None, None, None
    else:
        dataset = cfg["dataset"].build()
        n_train = cfg["dataset"].n_train
        n_val = cfg["dataset"].n_val
        n_test = cfg["dataset"].n_test

    run_experiment(
        model=cfg["model"].build_factory(),
        optimizer=cfg["optimizer"].build_factory() if "optimizer" in cfg.keys() else None,
        criterion=torch.nn.MSELoss(),
        dataset=dataset,
        n_epochs=500,
        batch_size=128,
        results_dir=results_dir,
        early_stopping=True,
        patience=10,
        early_stopping_grace_period=100,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        basis_selection=cfg["basis_selection"] if "basis_selection" in cfg.keys() else None,
        n_model_inits=cfg["n_model_inits"] if "n_model_inits" in cfg.keys() else None,
        model_init_epochs=cfg["model_init_epochs"] if "model_init_epochs" in cfg.keys() else None,
        store_model=True,
        n_runs=10,
        load_dataset_to_device=True,
        dataset_seed=1234,
        torch_seed=5678,
        device="cuda" if torch.cuda.is_available() else "cpu",
        plot_results=True,
        dtype=DTYPE,
    )

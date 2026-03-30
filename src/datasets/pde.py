import numpy as np
from pathlib import Path
import h5py


def load_darcyflow_dataset(
    x_shape: tuple[int, int],
    y_shape: tuple[int, int],
    n_val_samples: int,
    data_dir: str | Path,
    random_subsets: bool = True,
    rng_seed: int | None = None,
):
    """Import helper function to load the Darcy flow PDE dataset from the paper [1]. First download
    the data from the paper github repo:

    https://github.com/cics-nd/cnn-surrogate

    Yinhao Zhu, Nicolas Zabaras. Bayesian deep convolutional encoder-decoder networks for surrogate
    modeling and uncertainty quantification. Journal of Computational Phyics 366 (2018).

    Args:
        x_shape (tuple): Input data shape
        y_shape (tuple): Output data shape
        n_val_samples (int): Number of validation samples
        data_dir (str | Path): Path to the data
        random_subsets (bool, optional): Shuffle train set after import. Defaults to True.
        rng_seed (int | None, optional): Seed for shuffle. Defaults to None.
    """
    train_data_dir = Path(data_dir, f"kle50_mc10000.hdf5")
    test_data_dir1 = Path(data_dir, f"kle50_lhs*.hdf5")
    test_data_dir2 = Path(data_dir, f"kle50_mc500.hdf5")

    with h5py.File(train_data_dir, "r") as f:
        x_train = f["input"][()]
        y_train = f["output"][()]

    x_test = []
    y_test = []
    for d in [test_data_dir1, test_data_dir2]:
        ddir = Path(d)
        fname = ddir.stem
        fname = fname.split("_")

        if fname[-1].startswith("lhs"):
            dpaths = list(ddir.parent.rglob(f"{fname[0]}_lhs*"))
            for dpath in dpaths:
                with h5py.File(dpath, "r") as f:
                    x_test.append(f["input"][()])
                    y_test.append(f["output"][()])
        else:
            with h5py.File(ddir, "r") as f:
                x_test.append(f["input"][()])
                y_test.append(f["output"][()])

    x_test = np.concatenate(x_test, axis=0)
    y_test = np.concatenate(y_test, axis=0)

    rng = np.random.default_rng(rng_seed) if random_subsets else None

    height_in, width_in = x_shape
    height_out, width_out = y_shape

    data_h, data_w = x_train.shape[-2:]
    if height_in < data_h or width_in < data_w:
        x_train = center_crop(x_train[:, 0, ...], height_in, width_in)
        y_train = center_crop(y_train[:, 1, :, :], height_out, width_out)
        x_test = center_crop(x_test[:, 0, ...], height_in, width_in)
        y_test = center_crop(y_test[:, 1, :, :], height_out, width_out)
    elif height_in > x_train.shape[-2] or width_in > x_train.shape[-1]:
        msg = f"Image subset dimensions larger than image! ({height_in}x{width_in} vs. {data_h}x{data_w})"
        raise AssertionError(msg)

    # transform log-normal distributed inputs to standard normal (gaussian random field)
    x_train = np.log(x_train)
    x_test = np.log(x_test)

    if random_subsets:
        x_train, y_train = shuffle_dataset(x_train, y_train, rng)

    if n_val_samples is not None and n_val_samples > 0:
        n_train = x_train.shape[0] - n_val_samples
        x_val = x_train[n_train:]
        y_val = y_train[n_train:]
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
    else:
        x_val = None
        y_val = None

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def load_steadystatediffusion_dataset(
    n_val_samples: int,
    data_dir: str | Path,
    random_subsets: bool = True,
    rng_seed: int | None = None,
):
    """Import helper function to load the Steady State Diffusion PDE dataset from the paper [1].
    First download the data from the paper github repo:

    https://github.com/cics-nd/cnn-surrogate

    Yinhao Zhu, Nicolas Zabaras. Bayesian deep convolutional encoder–decoder networks for surrogate
    modeling and uncertainty quantification. Journal of Computational Phyics 366 (2018).

    Args:
        x_shape (tuple): Input data shape
        y_shape (tuple): Output data shape
        n_val_samples (int): Number of validation samples
        data_dir (str | Path): Path to the data
        random_subsets (bool, optional): Shuffle train set after import. Defaults to True.
        rng_seed (int | None, optional): Seed for shuffle. Defaults to None.
    """
    train_data_dir = Path(data_dir, f"data_train.npz")
    test_data_dir = Path(data_dir, f"data_test.npz")

    rng = np.random.default_rng(rng_seed) if random_subsets else None

    with np.load(train_data_dir) as f:
        x = f["inputs"]
        y = f["outputs"]
        if random_subsets:
            x_train, y_train = shuffle_dataset(x_train, y_train, rng=rng)

        with np.load(test_data_dir) as f:
            x_test = f["inputs"]
            y_test = f["outputs"]

    if n_val_samples is not None and n_val_samples > 0:
        n_train = x_train.shape[0] - n_val_samples
        x_val = x_train[n_train:]
        y_val = y_train[n_train:]
        x_train = x_train[:n_train]
        y_train = y_train[:n_train]
    else:
        x_val = None
        y_val = None

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def shuffle_dataset(x, y, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    rand_idx = rng.permutation(np.arange(x.shape[0]))
    return x[rand_idx, ...], y[rand_idx, ...]


def center_crop(x, height, width):
    h, w = x.shape[-2:]
    if height < h:
        hidx1 = int(h // 2 - height // 2)
    else:
        hidx1 = 0
    if width < w:
        widx1 = int(w // 2 - width // 2)
    else:
        widx1 = 0
    return x[..., hidx1 : hidx1 + height, widx1 : widx1 + width]

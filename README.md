# Deep Polynomial Chaos Expansion (DeepPCE)

This repository contains the experiment code for the paper

[Johannes Exenberger, Sascha Ranftl, Robert Peharz. *Deep Polynomial Chaos Expansion*.
29th International Conference on Artificial Intelligence and Statistics (AISTATS) 2026.](https://arxiv.org/abs/2507.21273)

The main code base for Deep Polynomial Chaos Expansions is released alongside this repository as the standalone library [`tensorchaos`](https://github.com/jxnb/tensorchaos).

## Install

Clone the repository from github and `cd` into the directory

```
git clone git@github.com:jxnb/deep-polynomial-chaos-expansion.git
```
```
cd deep-polynomial-chaos-expansion
```

The code in this repo is based on `tensorchaos` version `0.1.0`.
If you install the dependencies from the provided `pyproject.toml`, the correct version of `tensorchaos` will be installed from Github.
We recommend the package manager [`uv`](https://docs.astral.sh/uv/) - if you use `uv`, simply install the dependencies with

```
uv sync
```

Dependencies can also be installed using `pip`:

```
python -m venv .venv
source .venv/bin/activate
pip install .
```

## Running experiments

The experiments described in the paper can be reproduced using the `run_experiment.py` script.
The `config.py` file includes the configurations for all experiments as `pydantic BaseModels`.

**Example:** To perform the experiment for the DeepPCE model on the XDBenchmark dataset, simply run

```
python run_experiments.py deeppce_xdbenchmark
```

## Datasets

The repository includes the three benchmark UQ functions described in the paper.
The PDE datasets have to be downloaded or generated. 
After that, point the dataset configuration in `config.py` to the location of the dataset.

### Darcy flow dataset

The Darcy flow dataset is based on [1] and can be downloaded from the [Github paper repository](https://github.com/cics-nd/cnn-surrogate).

[1] [Zhu, Y. and Zabaras, N. *Bayesian deep convolutional encoder–decoder networks for surrogate modeling and uncertainty quantification.* Journal of Computational Physics, 366:415–447. 2018](https://www.sciencedirect.com/science/article/pii/S0021999118302341?via%3Dihub)

### Steady state diffusion dataset

The steady state diffusion data is based on [2] and can be generated using the solver provided in the
[Github paper repository](https://github.com/rohitkt10/deep-uq-paper).
To reproduce the dataset used in the paper, use the following settings for the Gaussian random field in the `generate_data.py` script:

| param |  val  |
| :---: | :---: |
| N     | 10000 |
| nx    | 64    |
| ny    | 64    |
| lx    | 10    |
| ly    | 10    |
| var   | 1     |
| k     | rbf   |

[2] [Tripathy, R. K. and Bilionis, I. *Deep UQ: Learning deep neural network surrogate models for high dimensional uncertainty quantification.* Journal of Computational Physics, 375:565–588, 2018](https://www.sciencedirect.com/science/article/pii/S0021999118305655#bl0010)

## Cite

```
@inproceedings{exenberger2026Deep,
  title = {Deep {{Polynomial Chaos Expansion}}},
  booktitle = {Proceedings of the {{29th International Conference}} on {{Artificial Intelligence}} and {{Statistics}} ({{AISTATS}})},
  author = {Exenberger, Johannes and Ranftl, Sascha and Peharz, Robert},
  year = 2026,
  publisher = {PMLR}
}
```
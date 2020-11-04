# Hydra Optuna Sweeper plugin

Provides a plugin for Hydra applications to utilize [Optuna](https://optuna.org) for the optimization of the parameters of experiments.

# Installation

You can install Optuna plugin by using pip:

```consle
pip install -e .
```

# Usage

Please set `hydra/sweeper` to `optuna` in your config file.

```yaml
defaults:
  - hydra/sweeper: optuna
```

Alternatively, add `hydra/sweeper=optuna` option to your command line.

You can see an example in this directory. `example/quadratic.py` implements a simple quadratic function to be minimized.

```console
python example/quadratic.py -m 'x=interval(-5.0, 5.0)' 'y=interval(1, 10)'
```

By default, interval is converted to `UniformDistribution`. You can use `IntUniformDistribution` or `LogUniformDistribution` by specifying the tags:

```console
python example/quadratic.py -m 'x=tag(int, interval(-5.0, 5.0))' 'y=tag(log, interval(1, 10))'
```

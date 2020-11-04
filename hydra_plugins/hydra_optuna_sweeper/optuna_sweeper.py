import logging
from typing import Any
from typing import List

from hydra.core.config_loader import ConfigLoader
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import ChoiceSweep
from hydra.core.override_parser.types import IntervalSweep
from hydra.core.override_parser.types import Override
from hydra.core.override_parser.types import Transformer
from hydra.core.plugins import Plugins
from hydra.plugins.sweeper import Sweeper
from hydra.types import TaskFunction
from omegaconf import DictConfig
from omegaconf import OmegaConf
import optuna
from optuna.distributions import CategoricalDistribution
from optuna.distributions import IntLogUniformDistribution
from optuna.distributions import IntUniformDistribution
from optuna.distributions import LogUniformDistribution
from optuna.distributions import UniformDistribution

from .config import OptunaConfig


log = logging.getLogger(__name__)


def create_optuna_distribution_from_override(override: Override) -> Any:
    value = override.value()
    if not override.is_sweep_override():
        return value

    if override.is_choice_sweep():
        assert isinstance(value, ChoiceSweep)
        choices = [x for x in override.sweep_iterator(transformer=Transformer.encode)]
        return CategoricalDistribution(choices)

    if override.is_range_sweep():
        choices = [x for x in override.sweep_iterator(transformer=Transformer.encode)]
        return CategoricalDistribution(choices)

    if override.is_interval_sweep():
        assert isinstance(value, IntervalSweep)
        if "log" in value.tags:
            if "int" in value.tags:
                return IntLogUniformDistribution(value.start, value.end)
            return LogUniformDistribution(value.start, value.end)
        else:
            if "int" in value.tags:
                return IntUniformDistribution(value.start, value.end)
            return UniformDistribution(value.start, value.end)

    raise NotImplementedError("{} is not supported by Optuna sweeper.".format(override))


class OptunaSweeper(Sweeper):

    def __init__(self, optuna_config: OptunaConfig) -> None:
        self.optuna_config = optuna_config

    def setup(
        self: DictConfig,
        config: DictConfig,
        config_loader: ConfigLoader,
        task_function: TaskFunction,
    ) -> None:

        self.config = config
        self.config_loader = config_loader
        self.launcher = Plugins.instance().instantiate_launcher(
            config=config, config_loader=config_loader, task_function=task_function
        )
        self.sweep_dir = config.hydra.sweep.dir

    def sweep(self, arguments: List[str]) -> None:
        parser = OverridesParser.create()
        parsed = parser.parse_overrides(arguments)

        search_space = {}
        for override in parsed:
            search_space[override.get_key_element()] = create_optuna_distribution_from_override(override)

        study = optuna.create_study(
            study_name=self.optuna_config.study_name,
            storage=self.optuna_config.storage,
            direction=self.optuna_config.direction
        )

        batch_size = self.optuna_config.n_jobs
        n_trials_to_go = self.optuna_config.n_trials

        while n_trials_to_go > 0:
            batch_size = min(n_trials_to_go, batch_size)

            trials = [study._ask() for _ in range(batch_size)]
            for trial in trials:
                for param_name, distribution in search_space.items():
                    trial._suggest(param_name, distribution)

            overrides = []
            for trial in trials:
                params = trial.params
                overrides.append(
                    tuple(f"{name}={val}" for name, val in params.items())
                )

            returns = self.launcher.launch(overrides, initial_job_idx=trials[0].number)
            for trial, ret in zip(trials, returns):
                study._tell(trial, optuna.trial.TrialState.COMPLETE, ret.return_value)
            n_trials_to_go -= batch_size

        best_trial = study.best_trial
        results_to_serialize = {
            "name": "optuna",
            "best_params": best_trial.params,
            "best_value": best_trial.value,
        }
        OmegaConf.save(
            OmegaConf.create(results_to_serialize),
            f"{self.config.hydra.sweep.dir}/optimization_results.yaml",
        )
        log.info(f"Best parameters: {best_trial.params}")
        log.info(f"Best value: {best_trial.value}")
        log.info(f"Storage: {self.optuna_config.storage}")
        log.info(f"Study name: {study.study_name}")

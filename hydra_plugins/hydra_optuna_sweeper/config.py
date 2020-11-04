from dataclasses import dataclass
from typing import Optional

from hydra.core.config_store import ConfigStore


@dataclass
class OptunaConfig:

    direction: str = "minimize"
    storage: Optional[str] = None
    study_name: Optional[str] = None
    n_trials: Optional[int] = 20
    n_jobs: int = 1
    timeout: Optional[float] = None

    # TODO(yanase): Configure sampler and pruner.

@dataclass
class OptunaSweeperConf:
    _target_: str = "hydra_plugins.hydra_optuna_sweeper.optuna_sweeper.OptunaSweeper"
    optuna_config: OptunaConfig = OptunaConfig()


ConfigStore.instance().store(
    group="hydra/sweeper", name="optuna", node=OptunaSweeperConf, provider="optuna_sweeper"
)

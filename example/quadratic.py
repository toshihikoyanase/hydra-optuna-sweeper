from typing import Any

import hydra
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def evaluate(cfg: DictConfig) -> Any:
    x = cfg.x
    y = cfg.y
    return (x - 2) ** 2 + y


if __name__ == "__main__":
    evaluate()

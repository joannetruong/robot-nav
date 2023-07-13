import hydra
from habitat.config.default import patch_config
from habitat_baselines.run import execute_exp
from omegaconf import DictConfig


@hydra.main(
    version_base=None,
    config_path="../",
    config_name="ver_hm3d_robot_nav",
)
def main(cfg: DictConfig):
    cfg = patch_config(cfg)
    execute_exp(cfg, "eval" if cfg.habitat_baselines.evaluate else "train")


if __name__ == "__main__":
    main()

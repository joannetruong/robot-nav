from habitat.config.read_write import read_write
from habitat_baselines.config.default import get_config
from habitat_baselines.run import build_parser, execute_exp
from omegaconf import OmegaConf
import kin_nav.kinematic_velocity_action  # necessary for updating registry!!


def main():
    parser = build_parser()
    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Same as habitat-lab, except we filter out actions that aren't kinematic velocity
    control.

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval".
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    with read_write(config):
        for k in list(config.habitat.task.actions.keys()):
            if k != "kinematic_velocity_control":
                del config.habitat.task.actions[k]
    # print(OmegaConf.to_yaml(config))  # good for debugging
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()

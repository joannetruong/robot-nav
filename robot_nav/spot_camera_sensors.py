import copy
from typing import Dict

import torch
from gym import spaces
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import ObservationTransformer
from habitat_baselines.config.default_structured_configs import ObsTransformConfig
from omegaconf import DictConfig

from dataclasses import dataclass, field
from typing import List

from habitat.config.default_structured_configs import HabitatSimDepthSensorConfig
from habitat.core.registry import registry
from habitat.sims.habitat_simulator.habitat_simulator import (
    HabitatSimDepthSensor,
)
from hydra.core.config_store import ConfigStore


class SpotDepthSensor(HabitatSimDepthSensor):
    obs_uuid: str = "spot_depth"

    def _get_uuid(self, *args, **kwargs):
        return self.obs_uuid


@registry.register_sensor
class SpotLeftDepthSensor(SpotDepthSensor):
    obs_uuid: str = "spot_left_depth"


@registry.register_sensor
class SpotRightDepthSensor(SpotDepthSensor):
    obs_uuid: str = "spot_right_depth"


class SpotDepthSensorConfig(HabitatSimDepthSensorConfig):
    height: int = 212
    width: int = 120
    hfov: float = 58.286
    min_depth: float = 0.0
    max_depth: float = 3.5


@dataclass
class SpotLeftDepthSensorConfig(SpotDepthSensorConfig):
    type: str = "SpotLeftDepthSensor"
    position: List[float] = field(
        default_factory=lambda: [-0.03740343144695029, 0.5, -0.4164822634134684]
    )
    # Euler's angles:
    orientation: List[float] = field(
        default_factory=lambda: [-0.4415683, -0.57648225, 0.270526]
    )


@dataclass
class SpotRightDepthSensorConfig(SpotDepthSensorConfig):
    type: str = "SpotRightDepthSensor"
    position: List[float] = field(
        default_factory=lambda: [0.03614789234067159, 0.5, -0.4164822634134684]
    )
    # Euler's angles:
    orientation: List[float] = field(
        default_factory=lambda: [-0.4415683, 0.57648225, -0.270526]
    )


@baseline_registry.register_obs_transformer()
class StitchSpotFrontVision(ObservationTransformer):
    def __init__(self):
        super().__init__()

    def transform_observation_space(self, observation_space: spaces.Dict, **kwargs):
        observation_space = copy.deepcopy(observation_space)

        for uuid in [SpotLeftDepthSensor.obs_uuid, SpotRightDepthSensor.obs_uuid]:
            if uuid not in observation_space.spaces:
                raise ValueError(
                    f"Observation space must contain {uuid} for StitchSpotFrontVision"
                )

        # Remove these two sensors from the observation space
        left_box = observation_space.spaces.pop(SpotLeftDepthSensor.obs_uuid)
        right_box = observation_space.spaces.pop(SpotRightDepthSensor.obs_uuid)

        # Assert that both left and right sensors had the same shape
        assert (
            left_box.shape == right_box.shape
        ), "Left and right sensors must have the same shape"

        # Compute the shape of the new sensor, which is just twice the width of the
        # original sensors
        new_shape = list(left_box.shape)
        new_shape[1] *= 2

        # Add a just 'depth' sensor to the observation space
        observation_space.spaces["depth"] = spaces.Box(
            low=-float("inf"), high=float("inf"), shape=new_shape, dtype=left_box.dtype
        )

        return observation_space

    @torch.no_grad()
    def forward(self, observations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Concatenate the left and right tensors along the width dimension; the LEFT
        # sensor is on the RIGHT side of the concatenated tensor, and the RIGHT sensor
        # is on the LEFT side of the concatenated tensor (this is because the
        # Spot robot is cross-eyed)

        left = observations.pop(SpotLeftDepthSensor.obs_uuid)
        right = observations.pop(SpotRightDepthSensor.obs_uuid)
        observations["depth"] = torch.cat((right, left), dim=2)

        return observations

    @classmethod
    def from_config(cls, config: "DictConfig"):
        return cls()


@dataclass
class StitchSpotFrontVisionConfig(ObsTransformConfig):
    type: str = StitchSpotFrontVision.__name__


# Register the sensors and observation transformer to the config store for Hydra to find

cs = ConfigStore.instance()

cs.store(
    group="habitat/simulator/sim_sensors",
    name="spot_left_depth_sensor",
    node=SpotLeftDepthSensorConfig,
)
cs.store(
    group="habitat/simulator/sim_sensors",
    name="spot_right_depth_sensor",
    node=SpotRightDepthSensorConfig,
)


cs.store(
    package="habitat_baselines.rl.policy.obs_transforms.stitch_spot_front_vision",
    group="habitat_baselines/rl/policy/obs_transforms",
    name="stitch_spot_front_vision",
    node=StitchSpotFrontVisionConfig,
)

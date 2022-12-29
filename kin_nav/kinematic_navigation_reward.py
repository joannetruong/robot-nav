from dataclasses import dataclass
from typing import Any

from habitat import EmbodiedTask, Measure, Simulator, registry
from habitat.config.default_structured_configs import MeasurementConfig
from habitat.tasks.nav.nav import DistanceToGoalReward
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig

""" Register full kinematic navigation reward measure below """


@dataclass
class KinematicNavRewardMeasurementConfig(MeasurementConfig):
    type: str = "KinematicNavReward"  # must match class name below
    collision_penalty: float = -0.003
    backwards_penalty: float = -0.003


@registry.register_measure
class KinematicNavReward(Measure):
    cls_uuid: str = "kinematic_nav_reward"

    def __init__(self, sim: Simulator, config: "DictConfig", *args: Any, **kwargs: Any):
        self._sim = sim
        self._config = config
        self.collision_penalty = config.collision_penalty
        self.backwards_penalty = config.backwards_penalty
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid,
            [DistanceToGoalReward.cls_uuid, BackwardsMotionCount.cls_uuid],
        )
        self.update_metric(episode=episode, task=task, *args, **kwargs)

    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_goal_reward = task.measurements.measures[
            DistanceToGoalReward.cls_uuid
        ].get_metric()
        collision_penalty = (
            self.collision_penalty if self._sim.kin_nav_metrics["collided"] else 0
        )
        backwards_penalty = (
            self.backwards_penalty
            if self._sim.kin_nav_metrics["backwards_motion"]
            else 0
        )
        self._metric = distance_to_goal_reward + collision_penalty + backwards_penalty


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.kinematic_nav_reward",
    group="habitat/task/measurements",
    name="kinematic_nav_reward",
    node=KinematicNavRewardMeasurementConfig,
)


class CounterMeasure(Measure):
    """Abstract measure class for counting"""

    cls_uuid: str = "counter"
    count_key: str = "None"

    def __init__(self, sim, *args: Any, **kwargs: Any):
        self._sim = sim
        self._metric = 0
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._metric = 0

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        if self._sim.kin_nav_metrics[self.count_key]:
            self._metric += 1


""" Register backwards motion counter measure below """


@dataclass
class BackwardsMotionCountMeasurementConfig(MeasurementConfig):
    type: str = "BackwardsMotionCount"  # must match class name below


@registry.register_measure
class BackwardsMotionCount(CounterMeasure):
    cls_uuid: str = "backwards_motion_count"
    count_key: str = "backwards_motion"


cs.store(
    package="habitat.task.measurements.backwards_motion_count",
    group="habitat/task/measurements",
    name="backwards_motion_count",
    node=BackwardsMotionCountMeasurementConfig,
)


""" Register collision counter measure below """


@dataclass
class CollisionCountMeasurementConfig(MeasurementConfig):
    type: str = "CollisionCount"  # must match class name below


@registry.register_measure
class CollisionCount(CounterMeasure):
    cls_uuid: str = "collision_count"
    count_key: str = "collided"


cs.store(
    package="habitat.task.measurements.collision_count",
    group="habitat/task/measurements",
    name="collision_count",
    node=CollisionCountMeasurementConfig,
)

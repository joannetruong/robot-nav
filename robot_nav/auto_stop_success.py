from typing import Any


from habitat.config.default_structured_configs import SuccessMeasurementConfig
from habitat.core.embodied_task import (
    EmbodiedTask,
)
from habitat.core.registry import registry
from habitat.tasks.nav.nav import Success, DistanceToGoal
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass


@dataclass
class AutoStopSuccessMeasurementConfig(SuccessMeasurementConfig):
    type: str = "AutoStopSuccess"  # must match class name below


@registry.register_measure
class AutoStopSuccess(Success):
    def update_metric(self, episode, task: EmbodiedTask, *args: Any, **kwargs: Any):
        distance_to_target = task.measurements.measures[
            DistanceToGoal.cls_uuid
        ].get_metric()
        if distance_to_target < self._success_distance:
            self._metric = 1.0
        else:
            self._metric = 0.0


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.measurements.auto_stop_success",
    group="habitat/task/measurements",
    name="auto_stop_success",
    node=AutoStopSuccessMeasurementConfig,
)

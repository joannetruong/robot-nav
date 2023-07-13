from dataclasses import dataclass, field
from typing import Any, List

import magnum as mn
import numpy as np
from gym import spaces
from habitat import registry
from habitat.config.default_structured_configs import ActionConfig
from habitat.core.embodied_task import EmbodiedTask, SimulatorTaskAction
from habitat.core.spaces import ActionSpace
from habitat.tasks.nav.nav import NavigationEpisode
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
from habitat_sim import RigidState
from habitat_sim._ext.habitat_sim_bindings import VelocityControl
from hydra.core.config_store import ConfigStore

from robot_nav.collision_checker import EmbodimentCollisionChecker

NOMINAL_JOINTS = [
    0, -180, 0, 135, 90, 0, -90, 0, 0, 60, -120, 0, 60, -120, 0, 60, -120, 0, 60, -120  # fmt: skip
]
NOMINAL_POSITION = [0.0, 0.5, 0.0]
NOMINAL_ROTATION = [0.0, 0.0, 180.0]  # roll, pitch, yaw in degrees


@dataclass
class KinematicVelocityControlActionConfig(ActionConfig):
    type: str = "KinematicVelocityAction"  # must be same name as class defined below!!
    # meters/sec:
    lin_vel_range: List[float] = field(default_factory=lambda: [0.0, 0.25])
    # deg/sec:
    ang_vel_range: List[float] = field(default_factory=lambda: [-10.0, 10.0])
    min_abs_lin_speed: float = 0.025  # meters/sec
    min_abs_ang_speed: float = 1.0  # deg/sec
    time_step: float = 1.0  # seconds
    robot_urdf: str = "data/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
    nominal_joints: List[float] = field(default_factory=lambda: NOMINAL_JOINTS)
    nominal_position: List[float] = field(default_factory=lambda: NOMINAL_POSITION)
    nominal_rotation: List[float] = field(default_factory=lambda: NOMINAL_ROTATION)


@registry.register_task_action
class KinematicVelocityAction(SimulatorTaskAction):
    name: str = "kinematic_velocity_control"

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.vel_control = VelocityControl()
        self.vel_control.controlling_lin_vel = True
        self.vel_control.controlling_ang_vel = True
        self.vel_control.lin_vel_is_local = True
        self.vel_control.ang_vel_is_local = True

        self.min_lin_vel, self.max_lin_vel = self._config.lin_vel_range
        self.min_ang_vel, self.max_ang_vel = self._config.ang_vel_range
        self.min_abs_lin_speed = self._config.min_abs_lin_speed
        self.min_abs_ang_speed = self._config.min_abs_ang_speed
        self.time_step = self._config.time_step
        self.robot_urdf = self._config.robot_urdf

        # Collision checking
        self.robot_collider = EmbodimentCollisionChecker(
            self._sim,
            self._config.robot_urdf,
            self._config.nominal_joints,
            self._config.nominal_position,
            self._config.nominal_rotation,
            robot_id=-1,  # robot_id and file will be manually set by reset()
        )

        self.robot_id = None

    @property
    def action_space(self):
        return ActionSpace(
            {
                "angular_velocity": spaces.Box(
                    low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
                ),
                "linear_velocity": spaces.Box(
                    low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32
                ),
            }
        )

    def reset(
        self, task: EmbodiedTask, episode: NavigationEpisode, *args: Any, **kwargs: Any
    ):
        task.is_stop_called = False

        # If robot hasn't spawned yet, or was removed when scene was reloaded
        if self.robot_id is None or self.robot_id.object_id == -1:
            ao_mgr = self._sim.get_articulated_object_manager()
            self.robot_id = ao_mgr.add_articulated_object_from_urdf(
                self.robot_urdf, fixed_base=False
            )
            self.robot_collider.robot_id = self.robot_id

        # Move the robot urdf to the start position and rotation
        collided = self.robot_collider.collided(
            episode.start_position, episode.info["start_yaw"]
        )
        if collided:
            print(
                "BAD EPISODE: Robot collided with start position. "
                "Details of episode below:"
            )
            print(episode)
            raise RuntimeError("Robot collided with something at start position!")
        self.update_metrics(collided=False, backwards_motion=False)

    def step(
        self,
        *args: Any,
        task: EmbodiedTask,
        angular_velocity: np.ndarray,  # SPELLING MUST MATCH KEY IN self.action_space
        linear_velocity: np.ndarray,  # SPELLING MUST MATCH KEY IN self.action_space
        **kwargs: Any,
    ):
        r"""Moves the agent with a provided linear and angular velocity for the
        provided amount of time

        Args:
            linear_velocity: between [-1,1], scaled according to
                             config.lin_vel_range
            angular_velocity: between [-1,1], scaled according to
                             config.ang_vel_range
        """
        # Extract from single-value array and convert from [-1, 1] to [0, 1] range
        linear_velocity = (linear_velocity[0] + 1.0) / 2.0
        angular_velocity = (angular_velocity[0] + 1.0) / 2.0

        # Scale actions
        linear_velocity = self.min_lin_vel + linear_velocity * (
            self.max_lin_vel - self.min_lin_vel
        )
        angular_velocity = self.min_ang_vel + angular_velocity * (
            self.max_ang_vel - self.min_ang_vel
        )

        final_pos, final_rot, backwards, collided, task.is_stop_called = self.teleport(
            linear_velocity, angular_velocity
        )
        self.update_metrics(collided, backwards)
        return self._sim.get_observations_at(
            position=final_pos,
            rotation=final_rot,
            keep_agent_at_new_pose=not (collided or task.is_stop_called),
        )

    def update_metrics(self, collided, backwards_motion):
        """Hacky, but simple way to keep track of metrics needed for reward penalties"""
        if not hasattr(self._sim, "kin_nav_metrics"):
            self._sim.kin_nav_metrics = {}
        self._sim.kin_nav_metrics["collided"] = collided
        self._sim.kin_nav_metrics["backwards_motion"] = backwards_motion

    def teleport(self, linear_velocity: float, angular_velocity: float):
        # Stop is called if both linear/angular speed are below their threshold
        stop = (
            abs(linear_velocity) < self.min_abs_lin_speed
            and abs(angular_velocity) < self.min_abs_ang_speed
        )
        if stop:
            final_position, final_rotation, backwards, collided = (
                None, None, False, False,  # fmt: skip
            )
        else:
            agent_state = self._sim.get_agent_state()
            agent_magnum_quat = mn.Quaternion(
                agent_state.rotation.imag, agent_state.rotation.real
            )
            current_rigid_state = RigidState(agent_magnum_quat, agent_state.position)
            angular_velocity = np.deg2rad(angular_velocity)
            # negative linear velocity is forward
            self.vel_control.linear_velocity = np.array([0.0, 0.0, -linear_velocity])
            self.vel_control.angular_velocity = np.array([0.0, angular_velocity, 0.0])
            goal_rigid_state = self.vel_control.integrate_transform(
                self.time_step, current_rigid_state
            )
            final_position, final_rotation = self.get_next_pos_rot(goal_rigid_state)
            # negative linear velocity is forward
            backwards = linear_velocity > 0.0
            collided = final_position is None

        return final_position, final_rotation, backwards, collided, stop

    def get_next_pos_rot(self, goal_rigid_state: RigidState):
        """
        Returns Nones if the agent would collide with an object. Otherwise, will adjust
        goal rigid state based on navmesh and return the new position and rotation.
        :param goal_rigid_state:
        :return: (None, None) or (position, rotation)
        """

        """ 1. Cancel if the point isn't even a valid point on the navmesh """
        if not self._sim.pathfinder.is_navigable(goal_rigid_state.translation):
            return None, None
        """ 2. Adjust height of goal state based on navmesh """
        navmesh_pos = self._sim.pathfinder.snap_point(goal_rigid_state.translation)
        """ 3. Check if collision would occur at adjusted goal """
        yaw = np.quaternion(
            goal_rigid_state.rotation.scalar, *goal_rigid_state.rotation.vector
        )
        yaw = -quat_to_rad(yaw) + np.pi / 2
        if self.robot_collider.collided(navmesh_pos, yaw, revert_on_collision=True):
            # Collision would occur, so return Nones
            return None, None

        final_rotation = [
            *goal_rigid_state.rotation.vector,
            goal_rigid_state.rotation.scalar,
        ]

        return navmesh_pos, final_rotation


cs = ConfigStore.instance()
cs.store(
    package="habitat.task.actions.kinematic_velocity_control",
    group="habitat/task/actions",
    name="kinematic_velocity_control",
    node=KinematicVelocityControlActionConfig,
)


def quat_to_rad(rotation):
    heading_vector = quaternion_rotate_vector(rotation.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi

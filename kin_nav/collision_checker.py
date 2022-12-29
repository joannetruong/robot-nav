from typing import Tuple, Union

import numpy as np
import magnum as mn


class EmbodimentCollisionChecker:
    def __init__(
        self,
        sim,
        robot_urdf,
        nominal_joints,
        nominal_position,
        nominal_rotation,
        robot_id=None,
    ):
        self.sim = sim

        # Extract data from Hydra config of embodiment
        if robot_id is None:
            ao_mgr = sim.get_articulated_object_manager()
            self.robot_id = ao_mgr.add_articulated_object_from_urdf(
                robot_urdf, fixed_base=False
            )
        else:
            self.robot_id = robot_id
        self.nominal_joints = np.deg2rad(nominal_joints)
        self.nominal_position = np.array(nominal_position)
        roll, pitch, yaw = np.deg2rad(nominal_rotation)
        # Habitat's positive y-axis is the conventional vertical positive z-axis
        self.nominal_rotation = (
            mn.Matrix4.rotation_y(mn.Rad(yaw))
            @ mn.Matrix4.rotation_x(mn.Rad(pitch))
            @ mn.Matrix4.rotation_z(mn.Rad(roll))
        )

    def check_position(self, position) -> Tuple[bool, Union[None, float]]:
        """Checks if the robot could be placed at position without colliding for at
        least one yaw value"""
        for _ in range(1000):
            yaw = np.random.uniform(0, 2 * np.pi)
            if not self.collided(position, yaw, revert_on_collision=False):
                return True, yaw
        return False, None

    def collided(self, position, yaw, revert_on_collision=True):
        """Moves the robot to the given position and yaw and returns whether a
        collision has occurred"""
        orig_transformation = self.robot_id.transformation
        self.robot_id.joint_positions = self.nominal_joints

        # Magnum's negative y-axis is the positive z-axis in Habitat convention
        robot_rigid_state = self.nominal_rotation @ mn.Matrix4.rotation_y(mn.Rad(-yaw))
        robot_rigid_state.translation = np.array(position) + self.nominal_position

        self.robot_id.transformation = robot_rigid_state  # move robot to position+yaw
        collided = self.sim.contact_test(self.robot_id.object_id)

        if collided and revert_on_collision:
            self.robot_id.transformation = orig_transformation

        return collided

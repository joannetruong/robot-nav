import argparse
import gzip
import random

import pathos.multiprocessing as multiprocessing
import os
import os.path as osp
from typing import Generator, List

import habitat
import numpy as np
import tqdm
import yaml
from habitat.datasets.pointnav.pointnav_generator import (
    ISLAND_RADIUS_LIMIT,
    _create_episode,
    is_compatible_episode,
)
from habitat.datasets.utils import get_action_shortest_path
from habitat.tasks.nav.nav import NavigationEpisode
from habitat_sim.errors import GreedyFollowerError
from omegaconf import DictConfig

from kin_nav.collision_checker import EmbodimentCollisionChecker

parent_dir = osp.dirname(osp.dirname(osp.abspath(__file__)))
with open(osp.join(parent_dir, "train_val_splits.yaml"), "r") as f:
    TRAIN_VAL_SPLITS = yaml.safe_load(f)


def generate_pointnav_episode(
    sim: "HabitatSim",
    robot_urdf: str,
    nominal_joints: list,
    nominal_position: list,
    nominal_rotation: list,
    num_episodes: int = -1,
    is_gen_shortest_path: bool = True,
    shortest_path_success_distance: float = 0.2,
    shortest_path_max_steps: int = 500,
    closest_dist_limit: float = 1,
    furthest_dist_limit: float = 50,
    geodesic_to_euclid_min_ratio: float = 1.1,
    number_retries_per_target: int = 1e6,
) -> Generator[NavigationEpisode, None, None]:
    r"""Generator function that generates PointGoal navigation episodes.

    An episode is trivial if there is an obstacle-free, straight line between
    the start and goal positions. A good measure of the navigation
    complexity of an episode is the ratio of
    geodesic shortest path position to Euclidean distance between start and
    goal positions to the corresponding Euclidean distance.
    If the ratio is nearly 1, it indicates there are few obstacles, and the
    episode is easy; if the ratio is larger than 1, the
    episode is difficult because strategic navigation is required.
    To keep the navigation complexity of the precomputed episodes reasonably
    high, we perform aggressive rejection sampling for episodes with the above
    ratio falling in the range [1, 1.1].
    Following this, there is a significant decrease in the number of
    straight-line episodes.


    :param robot_urdf: path to urdf file for the robot
    :param nominal_joints: joint configuration to use when checking for collisions
    :param nominal_position: local position of robot to use when checking for collisions
    (relative to a 3D coordinate lying on the ground of the navmesh)
    :param nominal_rotation: local rotation of robot to use when checking for collisions
    :param sim: simulator with loaded scene for generation.
    :param num_episodes: number of episodes needed to generate
    :param is_gen_shortest_path: option to generate shortest paths
    :param shortest_path_success_distance: success distance when agent should
    stop during shortest path generation
    :param shortest_path_max_steps maximum number of steps shortest path
    expected to be
    :param closest_dist_limit episode geodesic distance lowest limit
    :param furthest_dist_limit episode geodesic distance highest limit
    :param geodesic_to_euclid_min_ratio geodesic shortest path to Euclid
    distance ratio upper limit till aggressive sampling is applied.
    :return: navigation episode that satisfy specified distribution for
    currently loaded into simulator scene.
    """
    episode_count = 0
    collision_checker = EmbodimentCollisionChecker(
        sim, robot_urdf, nominal_joints, nominal_position, nominal_rotation
    )
    while episode_count < num_episodes or num_episodes < 0:
        (
            source_position,
            start_yaw,
            target_position,
            target_yaw,
            dist,
        ) = try_generate_episode(
            number_retries_per_target,
            sim,
            collision_checker,
            closest_dist_limit,
            furthest_dist_limit,
            geodesic_to_euclid_min_ratio,
        )
        source_rotation = [0, np.sin(start_yaw / 2), 0, np.cos(start_yaw / 2)]

        if is_gen_shortest_path:
            try:
                shortest_paths = [
                    get_action_shortest_path(
                        sim,
                        source_position=source_position,
                        source_rotation=source_rotation,
                        goal_position=target_position,
                        success_distance=shortest_path_success_distance,
                        max_episode_steps=shortest_path_max_steps,
                    )
                ]
            # Throws an error when it can't find a path
            except GreedyFollowerError:
                continue
        else:
            shortest_paths = None

        episode = _create_episode(
            episode_id=episode_count,
            scene_id=sim.habitat_config.scene,
            start_position=source_position,
            start_rotation=source_rotation,
            target_position=target_position,
            shortest_paths=shortest_paths,
            radius=shortest_path_success_distance,
            info={
                "geodesic_distance": dist,
                "start_yaw": start_yaw,
                "target_yaw": target_yaw,
            },
        )

        episode_count += 1
        yield episode


def try_generate_episode(
    number_retries_per_target: int,
    sim,
    collision_checker,
    closest_dist_limit,
    furthest_dist_limit,
    geodesic_to_euclid_min_ratio,
):
    for _retry in range(int(number_retries_per_target)):
        target_position = sim.sample_navigable_point()

        if sim.island_radius(target_position) < ISLAND_RADIUS_LIMIT:
            continue

        # Check if robot would collide with something at desired target pose
        compatible_target, target_yaw = collision_checker.check_position(
            target_position
        )
        if not compatible_target:
            continue

        source_position = sim.sample_navigable_point()
        is_compatible, dist = is_compatible_episode(
            source_position,
            target_position,
            sim,
            near_dist=closest_dist_limit,
            far_dist=furthest_dist_limit,
            geodesic_to_euclid_ratio=geodesic_to_euclid_min_ratio,
        )
        if is_compatible:
            # Validate source position
            is_compatible, start_yaw = collision_checker.check_position(source_position)
            if is_compatible:
                return source_position, start_yaw, target_position, target_yaw, dist

    raise RuntimeError("Could not generate a compatible episode!")


def _generate_fn(
    scene: str,
    scene_dir: str,
    cfg: DictConfig,
    out_dir: str,
    robot_urdf: str,
    nominal_joints: List[float],
    nominal_position: List[float],
    nominal_rotation: List[float],
    num_episodes_per_scene: int,
    is_hm3d: bool = False,
    split="train",
):
    if is_hm3d:
        scene_name = scene.split("-")[-1]
        scene_path = f"hm3d/{split}/{scene}/{scene_name}.basis.glb"
    else:
        scene_name = scene
        scene_path = f"gibson/{scene}.glb"

    # Skip this scene if a dataset was or is being generated for it
    out_file = osp.join(out_dir, f"{split}/content/{scene_name}.json.gz")
    if osp.exists(out_file) or osp.exists(out_file + ".incomplete"):
        return
    # Create an empty file so other processes know this scene is being processed
    with open(out_file + ".incomplete", "w") as f:
        f.write("")

    # Insert path to scene into config so it gets loaded
    full_scene_path = osp.join(scene_dir, scene_path)
    with habitat.config.read_write(cfg):
        cfg.habitat.simulator.scene = full_scene_path
        # Physics MUST BE ENABLED for EmbodimentCollisionChecker to work (contact_test)
        cfg.habitat.simulator.habitat_sim_v0.enable_physics = True
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.habitat.simulator)

    dset = habitat.datasets.make_dataset("PointNav-v1")
    dset.episodes = list(
        generate_pointnav_episode(
            sim,
            robot_urdf,
            nominal_joints,
            nominal_position,
            nominal_rotation,
            num_episodes_per_scene,
            is_gen_shortest_path=False,
        )
    )

    for ep in dset.episodes:
        ep.scene_id = scene_path

    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write(dset.to_json())
    os.remove(out_file + ".incomplete")


def generate_dataset(
    config_path: str,
    scene_dir: str,
    split: str,
    out_dir: str,
    robot_urdf: str,
    nominal_joints: List[float],
    nominal_position: List[float],
    nominal_rotation: List[float],
    dataset_type: str,
    overrides: list,
    num_episodes_per_scene: int,
):
    cfg = habitat.get_config(config_path=config_path, overrides=overrides)
    is_hm3d = dataset_type == "hm3d"
    scenes = TRAIN_VAL_SPLITS[dataset_type][split]
    out_file = osp.join(out_dir, f"{split}/{split}.json.gz")
    os.makedirs(osp.dirname(out_file), exist_ok=True)
    with gzip.open(out_file, "wt") as f:
        f.write('{"episodes": []}')

    _generate_fn_partial = lambda x: _generate_fn(
        x,
        scene_dir,
        cfg,
        out_dir,
        robot_urdf,
        nominal_joints,
        nominal_position,
        nominal_rotation,
        num_episodes_per_scene,
        is_hm3d,
        split,
    )
    # Shuffle order of elements in scenes
    random.shuffle(scenes)
    with multiprocessing.Pool(27) as pool, tqdm.tqdm(total=len(scenes)) as pbar:
        for _ in pool.imap_unordered(_generate_fn_partial, scenes):
            pbar.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--exp-config",
        required=True,
        help="Path to config yaml used to setup the simulator",
    )
    parser.add_argument(
        "--dataset-type",
        required=True,
        help="Dataset type to generate. One of [hm3d, gibson]",
    )
    parser.add_argument(
        "--split",
        required=True,
        help="Which dataset split to generate. One of [train, val]",
    )
    parser.add_argument(
        "--out-dir",
        required=True,
        type=str,
        help="Path to output directory for dataset",
    )
    parser.add_argument(
        "--robot-urdf",
        required=True,
        type=str,
        help="Path to robot's urdf file",
    )
    parser.add_argument(
        "--nominal-joints",
        required=True,
        type=str,
        help="Joint configuration used for collision checking",
    )
    parser.add_argument(
        "--nominal-position",
        required=True,
        type=str,
        help="Local position used for collision checking",
    )
    parser.add_argument(
        "--nominal-rotation",
        required=True,
        type=str,
        help="Local roll pitch yaw used for collision checking",
    )
    parser.add_argument(
        "-s",
        "--scenes-dir",
        help="Path to the scene directory",
        default="data/scene_datasets",
    )
    parser.add_argument(
        "-o",
        "--overrides",
        nargs="*",
        help="Modify config options from command line",
    )
    parser.add_argument(
        "-n",
        "--num_episodes_per_scene",
        type=int,
        help="Number of episodes per scene",
        default=1e3,
    )
    args = parser.parse_args()
    assert args.dataset_type in [
        "hm3d",
        "gibson",
    ], f"Invalid dataset type {args.dataset_type}"
    generate_dataset(
        args.exp_config,
        args.scenes_dir,
        args.split,
        args.out_dir,
        args.robot_urdf,
        eval(args.nominal_joints),
        eval(args.nominal_position),
        eval(args.nominal_rotation),
        args.dataset_type,
        args.overrides,
        args.num_episodes_per_scene,
    )

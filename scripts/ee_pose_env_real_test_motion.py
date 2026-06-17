import time
from dataclasses import dataclass
from typing import Sequence

import numpy as np

from i2rt.ee_pose_env_real import EnvConfig, RealRobotEndEffectorEnv


@dataclass
class Args:
    delta_xyz: Sequence[float] = (0.03, 0.03, 0.03)
    delta_rpy: Sequence[float] = (0.3, 0.3, 0.3)
    dwell_s: float = 1.0


def main() -> None:
    args = Args()
    env = RealRobotEndEffectorEnv(EnvConfig(control_mode="absolute"))

    deltas = [
        (np.array([args.delta_xyz[0], 0.0, 0.0]), np.zeros(3)),
        (np.array([-args.delta_xyz[0], 0.0, 0.0]), np.zeros(3)),
        (np.array([0.0, args.delta_xyz[1], 0.0]), np.zeros(3)),
        (np.array([0.0, -args.delta_xyz[1], 0.0]), np.zeros(3)),
        (np.array([0.0, 0.0, args.delta_xyz[2]]), np.zeros(3)),
        (np.array([0.0, 0.0, -args.delta_xyz[2]]), np.zeros(3)),
        # (np.zeros(3), np.array([args.delta_rpy[0], 0.0, 0.0])),
        # (np.zeros(3), np.array([-args.delta_rpy[0], 0.0, 0.0])),
        # (np.zeros(3), np.array([0.0, args.delta_rpy[1], 0.0])),
        # (np.zeros(3), np.array([0.0, -args.delta_rpy[1], 0.0])),
        # (np.zeros(3), np.array([0.0, 0.0, args.delta_rpy[2]])),
        # (np.zeros(3), np.array([0.0, 0.0, -args.delta_rpy[2]])),
    ]

    # try:
    #     for delta_xyz, delta_rpy in deltas:
    #         action = np.concatenate([delta_xyz, delta_rpy, [0.0]])
    #         obs = env.get_observation()
    #         action
    #         print("Previous Position:", obs["robot_state"]["cartesian_position"])
    #         env.step(action)
    #         time.sleep(0.2)
    #         obs = env.get_observation()
    #         print("Current Position:", obs["robot_state"]["cartesian_position"])
    #         breakpoint()
    #         time.sleep(args.dwell_s)
    # finally:
    #     env.close()

    try:
        action = np.array([ 1.41213149e-01, 0.1,  1.36971214e-01, 0, 0, 0, 1])
        env.step(action)
        breakpoint()
    finally:
        env.close()

if __name__ == "__main__":

    main()

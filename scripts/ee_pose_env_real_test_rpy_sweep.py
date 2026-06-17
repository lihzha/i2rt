import time
from dataclasses import dataclass

import numpy as np

from scripts.ee_pose_env_real import EnvConfig, RealRobotEndEffectorEnv


@dataclass
class Args:
    step_rpy: float = 0.05
    steps: int = 30
    dwell_s: float = 0.05


def sweep_axis(env: RealRobotEndEffectorEnv, axis: int, step: float, steps: int, dwell_s: float) -> None:
    delta = np.zeros(3)
    for _ in range(steps):
        delta[:] = 0.0
        delta[axis] = -step
        action = np.concatenate([np.zeros(3), delta, [0.0]])
        env.step(action)
        time.sleep(dwell_s)
    for _ in range(steps):
        delta[:] = 0.0
        delta[axis] = step
        action = np.concatenate([np.zeros(3), delta, [0.0]])
        env.step(action)
        time.sleep(dwell_s)


def main() -> None:
    args = Args()
    env = RealRobotEndEffectorEnv(
        EnvConfig(
            rotation_frame="world",
            debug_ik=True
        )
    )
    try:
        # sweep_axis(env, axis=0, step=args.step_rpy, steps=args.steps, dwell_s=args.dwell_s)
        sweep_axis(env, axis=1, step=args.step_rpy, steps=args.steps, dwell_s=args.dwell_s)
        sweep_axis(env, axis=2, step=args.step_rpy, steps=args.steps, dwell_s=args.dwell_s)
    finally:
        env.close()


if __name__ == "__main__":
    main()

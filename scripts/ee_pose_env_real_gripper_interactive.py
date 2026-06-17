import time

import numpy as np

from i2rt.ee_pose_env_real import EnvConfig, RealRobotEndEffectorEnv


def parse_gripper_input(raw: str, last_value: float) -> float:
    text = raw.strip().lower()
    if text in {"q", "quit", "exit"}:
        raise SystemExit
    if text == "":
        return last_value
    return float(text)


def main() -> None:
    env = RealRobotEndEffectorEnv(EnvConfig(control_mode="delta"))
    last_gripper = 0.0
    print("Enter gripper position (float). Empty line repeats last. 'q' to quit.")
    try:
        while True:
            try:
                raw = input("gripper> ")
            except EOFError:
                break
            try:
                gripper_pos = parse_gripper_input(raw, last_gripper)
            except SystemExit:
                break
            last_gripper = gripper_pos

            action = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos], dtype=float)
            env.step(action)
            time.sleep(0.05)
    finally:
        env.close()


if __name__ == "__main__":
    main()

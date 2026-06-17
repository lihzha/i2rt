"""Replay trajectories collected by minimum_gello_collect.py on the robot.

Uses the RealRobotEndEffectorEnv for smooth trajectory execution with
configurable interpolation profiles (minimum_jerk, cubic, etc.).
"""

import glob
import json
import os
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import tyro

# Lazy imports for robot dependencies (allows list/info modes without hardware)
if TYPE_CHECKING:
    from i2rt.ee_pose_env_real import (
        TrajectoryConfig,
        TrajectoryInterpolator,
        TrajectoryProfile,
    )


def load_trajectory(filepath: str) -> dict:
    """Load a trajectory from an npz file."""
    data = np.load(filepath, allow_pickle=True)
    result = {key: data[key] for key in data.keys()}

    # Parse metadata
    if "metadata_json" in result:
        result["metadata"] = json.loads(result["metadata_json"][0])

    return result


def list_trajectories(directory: str, prefix: str = "follower_traj") -> list[str]:
    """List all trajectory files in a directory."""
    pattern = os.path.join(directory, f"{prefix}_*.npz")
    files = sorted(glob.glob(pattern))
    return files


@dataclass
class Args:
    trajectory_file: str | None = None
    """Path to a specific trajectory file to replay. If not provided, lists available trajectories."""
    trajectory_dir: str = "data/teleop"
    """Directory containing trajectory files."""
    trajectory_prefix: str = "follower_traj"
    """Prefix for trajectory files."""
    channel: str = "can0"
    """CAN channel for the robot."""
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "linear_4310"
    """Gripper type."""
    speed_factor: float = 1.0
    """Speed multiplier for replay (1.0 = original speed, 0.5 = half speed, 2.0 = double speed)."""
    loop: bool = False
    """Loop the trajectory continuously."""
    use_leader_pos: bool = True
    """Use leader joint positions (True) or follower joint positions (False) for replay."""
    move_duration: float = 0.0
    """Duration to move to start position (seconds). Set to 0 for auto-compute based on distance."""
    mode: Literal["replay", "list", "info"] = "replay"
    """Mode: replay a trajectory, list available trajectories, or show trajectory info."""
    trajectory_profile: str = "minimum_jerk"
    """Trajectory interpolation profile for smooth motion (linear, cubic, quintic, minimum_jerk, trapezoidal, s_curve)."""


def print_trajectory_info(filepath: str) -> None:
    """Print information about a trajectory file."""
    data = load_trajectory(filepath)
    print(f"\nTrajectory: {os.path.basename(filepath)}")
    print("-" * 50)

    # Metadata
    if "metadata" in data:
        print("Metadata:")
        for key, value in data["metadata"].items():
            print(f"  {key}: {value}")

    # Data shape info
    print("\nData arrays:")
    for key in data.keys():
        if key in ("metadata_json", "metadata"):
            continue
        arr = data[key]
        if isinstance(arr, np.ndarray):
            print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")

    # Duration estimation
    if "timestamps_s" in data:
        ts = data["timestamps_s"]
        if len(ts) > 1:
            # Handle object dtype timestamps
            if ts.dtype == object:
                ts = np.array([float(t) for t in ts])
            duration = ts[-1] - ts[0]
            print(f"\nDuration: {duration:.2f} seconds")
            print(f"Samples: {len(ts)}")
            print(f"Sample rate: {len(ts) / duration:.1f} Hz")


def compute_safe_duration(
    start_pos: np.ndarray,
    target_pos: np.ndarray,
    max_joint_velocity: float = 0.5,  # rad/s - conservative speed
    min_duration: float = 2.0,
    max_duration: float = 10.0,
) -> float:
    """Compute a safe duration based on joint displacement."""
    # Use the same length arrays
    min_len = min(len(start_pos), len(target_pos))
    delta = np.abs(target_pos[:min_len] - start_pos[:min_len])
    max_delta = np.max(delta)

    # Duration = max_displacement / velocity
    computed_duration = max_delta / max_joint_velocity

    return np.clip(computed_duration, min_duration, max_duration)


def slow_move_with_interpolator(
    env,
    target_pos: np.ndarray,
    duration: float | None = None,
    profile: "TrajectoryProfile | None" = None,
    max_joint_velocity: float = 0.5,  # rad/s - conservative for safety
) -> None:
    """Slowly move to target position using smooth trajectory interpolation.
    
    Args:
        env: Robot environment with robot.get_joint_pos() and robot.command_joint_pos()
        target_pos: Target joint positions
        duration: Duration in seconds. If None, computed automatically based on distance.
        profile: Trajectory interpolation profile (default: minimum_jerk)
        max_joint_velocity: Maximum joint velocity for auto-duration calculation (rad/s)
    """
    from i2rt.ee_pose_env_real import TrajectoryConfig, TrajectoryInterpolator, TrajectoryProfile

    if profile is None:
        profile = TrajectoryProfile.MINIMUM_JERK
    config = TrajectoryConfig(profile=profile)
    interpolator = TrajectoryInterpolator(config)

    start_pos = env.robot.get_joint_pos()
    dt = 0.01  # 100Hz control

    # Ensure same length
    if len(target_pos) != len(start_pos):
        # Pad target with current gripper if needed
        if len(target_pos) == len(start_pos) - 1:
            target_pos = np.concatenate([target_pos, [start_pos[-1]]])
        elif len(target_pos) > len(start_pos):
            target_pos = target_pos[:len(start_pos)]
        else:
            raise ValueError(f"Target position length mismatch: {len(target_pos)} vs {len(start_pos)}")

    # Auto-compute duration if not specified
    if duration is None:
        duration = compute_safe_duration(start_pos, target_pos, max_joint_velocity)

    # Print movement info
    delta = np.abs(target_pos - start_pos)
    max_delta_rad = np.max(delta)
    max_delta_deg = np.degrees(max_delta_rad)
    print(f"  Max joint displacement: {max_delta_rad:.3f} rad ({max_delta_deg:.1f} deg)")
    print(f"  Movement duration: {duration:.2f}s")

    steps = int(duration / dt)
    start_time = time.perf_counter()

    for i in range(steps + 1):
        loop_start = time.perf_counter()
        elapsed = loop_start - start_time
        t_normalized = min(elapsed / duration, 1.0)

        # Interpolate using smooth profile
        command_pos = interpolator.interpolate(start_pos, target_pos, t_normalized)
        env.robot.command_joint_pos(command_pos)

        # Progress feedback every 25%
        progress = int(t_normalized * 100)
        if i == 0 or (i > 0 and progress % 25 == 0 and progress != int((i-1) / steps * 100) // 25 * 25):
            print(f"  Moving... {progress}%", end="\r")

        # Maintain timing
        loop_elapsed = time.perf_counter() - loop_start
        sleep_time = dt - loop_elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

    # Ensure we reach target
    env.robot.command_joint_pos(target_pos)
    print("  Moving... 100% - Done!    ")


def replay_trajectory(
    env,
    data: dict,
    speed_factor: float = 1.0,
    use_leader_pos: bool = True,
    move_duration: float = 2.0,
    trajectory_profile: "TrajectoryProfile | None" = None,
) -> bool:
    """Replay a trajectory on the robot.

    Returns True if completed, False if interrupted.
    """
    from i2rt.ee_pose_env_real import TrajectoryProfile

    if trajectory_profile is None:
        trajectory_profile = TrajectoryProfile.MINIMUM_JERK

    # Select joint positions to replay
    if use_leader_pos and "leader_joint_pos" in data:
        joint_pos = data["leader_joint_pos"]
    elif "follower_joint_pos" in data:
        joint_pos = data["follower_joint_pos"]
        # Append gripper if available
        if "follower_gripper_pos" in data:
            gripper_pos = data["follower_gripper_pos"]
            joint_pos = np.concatenate([joint_pos, gripper_pos], axis=1)
    else:
        print("Error: No joint position data found in trajectory.")
        return False

    # Get timestamps
    timestamps = data.get("timestamps_s", None)
    if timestamps is not None and timestamps.dtype == object:
        timestamps = np.array([float(t) for t in timestamps])

    num_steps = len(joint_pos)
    print(f"Replaying {num_steps} steps...")

    # Show current vs target start position
    current_pos = env.robot.get_joint_pos()
    target_start = joint_pos[0]
    print(f"\nCurrent position:  {np.array2string(current_pos, precision=3, suppress_small=True)}")
    print(f"Start position:    {np.array2string(target_start[:len(current_pos)], precision=3, suppress_small=True)}")

    # Move to start position with smooth interpolation
    print(f"\n[Step 1] Moving slowly to start position using {trajectory_profile.value} profile...")
    slow_move_with_interpolator(
        env,
        joint_pos[0],
        duration=move_duration if move_duration > 0 else None,  # None = auto-compute
        profile=trajectory_profile,
    )

    # Brief pause at start position
    print("\nAt start position. Pausing for 1 second...")
    time.sleep(1.0)

    print("\n[Step 2] Starting trajectory replay...")

    # Determine timing
    if timestamps is not None and len(timestamps) > 1:
        # Use recorded timestamps for timing
        start_time = time.monotonic()
        base_ts = timestamps[0]

        for i, pos in enumerate(joint_pos):
            if i == 0:
                continue

            # Calculate target time based on original timestamps
            target_elapsed = (timestamps[i] - base_ts) / speed_factor
            current_elapsed = time.monotonic() - start_time

            # Wait until target time
            sleep_time = target_elapsed - current_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

            env.robot.command_joint_pos(pos)

            # Progress indicator
            if i % 50 == 0 or i == num_steps - 1:
                print(f"  Progress: {i + 1}/{num_steps} ({100 * (i + 1) / num_steps:.1f}%)")
    else:
        # Fallback: use fixed dt (10ms default from collect script)
        dt = 0.01 / speed_factor
        for i, pos in enumerate(joint_pos):
            env.robot.command_joint_pos(pos)
            time.sleep(dt)

            if i % 50 == 0 or i == num_steps - 1:
                print(f"  Progress: {i + 1}/{num_steps} ({100 * (i + 1) / num_steps:.1f}%)")

    print("Replay complete!")
    return True


def main(args: Args) -> None:
    if args.mode == "list":
        # List available trajectories
        files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
        if not files:
            print(f"No trajectories found in {args.trajectory_dir}")
            return
        print(f"\nAvailable trajectories in {args.trajectory_dir}:")
        print("-" * 50)
        for i, f in enumerate(files):
            print(f"  [{i}] {os.path.basename(f)}")
        print(f"\nTotal: {len(files)} trajectories")
        return

    if args.mode == "info":
        if args.trajectory_file:
            print_trajectory_info(args.trajectory_file)
        else:
            # Show info for all trajectories
            files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
            for f in files:
                print_trajectory_info(f)
        return

    # Replay mode
    if args.trajectory_file is None:
        # Interactive selection
        files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
        if not files:
            print(f"No trajectories found in {args.trajectory_dir}")
            return

        print(f"\nAvailable trajectories in {args.trajectory_dir}:")
        print("-" * 50)
        for i, f in enumerate(files):
            print(f"  [{i}] {os.path.basename(f)}")

        while True:
            try:
                selection = input(f"\nSelect trajectory to replay [0-{len(files)-1}] or 'q' to quit: ").strip()
                if selection.lower() == "q":
                    return
                idx = int(selection)
                if 0 <= idx < len(files):
                    args.trajectory_file = files[idx]
                    break
                print(f"Please enter a number between 0 and {len(files)-1}")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

    # Load trajectory
    print(f"\nLoading trajectory: {args.trajectory_file}")
    data = load_trajectory(args.trajectory_file)
    print_trajectory_info(args.trajectory_file)

    # Import robot dependencies (only needed for replay mode)
    from i2rt.ee_pose_env_real import TrajectoryProfile
    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.robots.utils import GripperType

    # Parse trajectory profile
    try:
        trajectory_profile = TrajectoryProfile(args.trajectory_profile)
    except ValueError:
        print(f"Unknown trajectory profile: {args.trajectory_profile}")
        print(f"Available: {[p.value for p in TrajectoryProfile]}")
        return

    # Initialize robot directly for joint-level control
    print(f"\nInitializing robot on {args.channel} with gripper {args.gripper}...")

    gripper_type = GripperType.from_string_name(args.gripper)
    robot = get_yam_robot(channel=args.channel, gripper_type=gripper_type)

    # Create a minimal wrapper to hold the robot
    class MinimalEnv:
        def __init__(self, robot):
            self.robot = robot

    env = MinimalEnv(robot)

    current_pos = env.robot.get_joint_pos()
    print(f"Current robot position: {current_pos}")

    # Confirm before replay
    response = input("\nReady to replay. Press Enter to start or 'q' to quit: ").strip()
    if response.lower() == "q":
        return

    print("\n" + "=" * 50)
    print("CONTROLS:")
    print("  Ctrl+C: Stop replay")
    print("=" * 50 + "\n")

    try:
        while True:
            replay_trajectory(
                env,
                data,
                speed_factor=args.speed_factor,
                use_leader_pos=args.use_leader_pos,
                move_duration=args.move_duration,
                trajectory_profile=trajectory_profile,
            )
            if not args.loop:
                break
            print("\nLooping... Press Ctrl+C to stop.\n")
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n\nReplay interrupted by user.")

    print("\nReplay session ended.")


if __name__ == "__main__":
    main(tyro.cli(Args))

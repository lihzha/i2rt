import glob
import json
import os
import shutil
import time
from dataclasses import dataclass
from typing import Dict, Literal

import numpy as np
import portal
import tyro

from i2rt.cameras.multi_camera_wrapper import MultiCameraWrapper
from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.robot import Robot

DEFAULT_ROBOT_PORT = 11333


class ServerRobot:
    """A simple server for a leader robot."""

    def __init__(self, robot: Robot, port: str):
        self._robot = robot
        self._server = portal.Server(port)
        print(f"Robot Sever Binding to {port}, Robot: {robot}")

        self._server.bind("num_dofs", self._robot.num_dofs)
        self._server.bind("get_joint_pos", self._robot.get_joint_pos)
        self._server.bind("command_joint_pos", self._robot.command_joint_pos)
        self._server.bind("command_joint_state", self._robot.command_joint_state)
        self._server.bind("get_observations", self._robot.get_observations)

    def serve(self) -> None:
        """Serve the leader robot."""
        self._server.start()


class ClientRobot(Robot):
    """A simple client for a leader robot."""

    def __init__(self, port: int = DEFAULT_ROBOT_PORT, host: str = "127.0.0.1"):
        self._client = portal.Client(f"{host}:{port}")

    def num_dofs(self) -> int:
        """Get the number of joints in the robot.

        Returns:
            int: The number of joints in the robot.
        """
        return self._client.num_dofs().result()

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot.

        Returns:
            T: The current state of the leader robot.
        """
        return self._client.get_joint_pos().result()

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_pos (T): The state to command the leader robot to.
        """
        self._client.command_joint_pos(joint_pos)

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to the given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        self._client.command_joint_state(joint_state)

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the leader robot.

        Returns:
            Dict[str, np.ndarray]: The current observations of the leader robot.
        """
        return self._client.get_observations().result()


class YAMLeaderRobot:
    def __init__(self, robot: MotorChainRobot):
        self._robot = robot
        self._motor_chain = robot.motor_chain

    def get_info(self) -> np.ndarray:
        qpos = self._robot.get_observations()["joint_pos"]
        encoder_obs = self._motor_chain.get_same_bus_device_states()
        time.sleep(0.01)
        gripper_cmd = 1 - encoder_obs[0].position
        qpos_with_gripper = np.concatenate([qpos, [gripper_cmd]])
        return qpos_with_gripper, encoder_obs[0].io_inputs

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        assert joint_pos.shape[0] == 6
        self._robot.command_joint_pos(joint_pos)

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        self._robot.update_kp_kd(kp, kd)


@dataclass
class Args:
    gripper: Literal[
        "crank_4310",
        "linear_3507",
        "linear_4310",
        "yam_teaching_handle",
        "no_gripper",
    ] = "linear_4310"
    mode: Literal["follower", "leader_collect"] = "follower"
    server_host: str = "localhost"
    server_port: int = DEFAULT_ROBOT_PORT
    can_channel: str = "can0"
    bilateral_kp: float = 0.2
    output_dir: str = "data/teleop"
    output_prefix: str = "follower_traj"
    num_trajectories: int = 20
    record_dt: float = 0.01
    record_video: bool = True
    record_images: bool = True


def _stack_or_object(values: list) -> np.ndarray:
    if not values:
        return np.array([], dtype=object)
    if all(isinstance(v, np.ndarray) for v in values):
        try:
            return np.stack(values, axis=0)
        except ValueError:
            return np.array(values, dtype=object)
    return np.array(values, dtype=object)


def save_trajectory(
    output_dir: str,
    output_prefix: str,
    data: Dict[str, list],
    image_data: Dict[str, list],
    camera_timestamps: Dict[str, list],
    metadata: Dict[str, str],
    timestamp: str | None = None,
) -> str:
    os.makedirs(output_dir, exist_ok=True)
    if timestamp is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"{output_prefix}_{timestamp}.npz")

    arrays: Dict[str, np.ndarray] = {}
    for key, values in data.items():
        if not values:
            continue
        arrays[key] = _stack_or_object(values)

    for key, values in image_data.items():
        if not values:
            continue
        arrays[f"image_{key}"] = _stack_or_object(values)

    for key, values in camera_timestamps.items():
        if not values:
            continue
        arrays[f"camera_ts_{key}"] = np.array(values)

    arrays["metadata_json"] = np.array([json.dumps(metadata, sort_keys=True)])
    np.savez(output_path, **arrays)
    return output_path


def main(args: Args) -> None:
    from i2rt.robots.utils import GripperType

    gripper_type = GripperType.from_string_name(args.gripper)

    if args.mode == "follower":
        robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)
        server_robot = ServerRobot(robot, args.server_port)
        server_robot.serve()
        return

    robot = get_yam_robot(channel=args.can_channel, gripper_type=gripper_type)
    robot = YAMLeaderRobot(robot)
    robot_current_kp = robot._robot._kp
    client_robot = ClientRobot(args.server_port, host=args.server_host)

    camera_reader = None
    if args.record_images or args.record_video:
        camera_reader = MultiCameraWrapper()
    time.sleep(10)  # wait for everything to be ready

    current_joint_pos, current_button = robot.get_info()
    current_follower_joint_pos = client_robot.get_joint_pos()
    print(f"Current leader joint pos: {current_joint_pos}")
    print(f"Current follower joint pos: {current_follower_joint_pos}")


    def slow_move(joint_pos: np.ndarray, duration: float = 1.0) -> None:
        for i in range(100):
            current_joint_pos = joint_pos
            follower_command_joint_pos = current_joint_pos * i / 100 + current_follower_joint_pos * (1 - i / 100)
            client_robot.command_joint_pos(follower_command_joint_pos)
            time.sleep(duration / 100)

    def init_buffers() -> tuple[Dict[str, list], Dict[str, list], Dict[str, list], float]:
        record_data: Dict[str, list] = {
            "timestamps_s": [],
            "leader_joint_pos": [],
            "follower_joint_pos": [],
            "follower_joint_vel": [],
            "follower_joint_eff": [],
        }
        image_data: Dict[str, list] = {}
        camera_timestamps: Dict[str, list] = {}
        next_sample_time = time.monotonic()
        return record_data, image_data, camera_timestamps, next_sample_time

    def next_traj_tag(output_dir: str, output_prefix: str) -> str:
        base = time.strftime("%Y%m%d_%H%M%S")
        tag = base
        counter = 1
        while os.path.exists(os.path.join(output_dir, f"{output_prefix}_{tag}.npz")) or os.path.exists(
            os.path.join(output_dir, f"{output_prefix}_{tag}")
        ):
            tag = f"{base}_{counter}"
            counter += 1
        return tag

    def count_existing_trajectories(output_dir: str, output_prefix: str) -> int:
        """Count existing trajectory files in the output directory."""
        if not os.path.exists(output_dir):
            return 0
        pattern = os.path.join(output_dir, f"{output_prefix}_*.npz")
        existing_files = glob.glob(pattern)
        return len(existing_files)

    def prompt_save(traj_index: int) -> bool:
        while True:
            response = input(f"Save trajectory {traj_index}? [y/N]: ").strip().lower()
            if response in ("y", "yes"):
                return True
            if response in ("n", "no", ""):
                return False
            print("Please enter 'y' or 'n'.")

    def wait_button_release(button_idx: int) -> None:
        """Wait until the specified button is released."""
        while True:
            _, current_button = robot.get_info()
            if current_button[button_idx] <= 0.5:
                break
            time.sleep(0.03)

    def handle_recording_end(
        record_data: Dict[str, list],
        image_data: Dict[str, list],
        camera_timestamps: Dict[str, list],
        traj_timestamp: str,
        traj_video_dir: str | None,
        saved_count: int,
    ) -> tuple[bool, int]:
        """Handle end of recording: prompt to save and return (should_continue, new_saved_count)."""
        if camera_reader is not None and args.record_video:
            camera_reader.stop_recording()

        if not record_data["timestamps_s"]:
            print("No samples recorded for this trajectory.")
            if traj_video_dir is not None:
                shutil.rmtree(traj_video_dir, ignore_errors=True)
            return True, saved_count

        should_save = prompt_save(saved_count + 1)
        if should_save:
            metadata = {
                "gripper": args.gripper,
                "server_host": args.server_host,
                "server_port": str(args.server_port),
                "bilateral_kp": str(args.bilateral_kp),
                "record_dt": str(args.record_dt),
                "trajectory_index": str(saved_count + 1),
            }
            output_path = save_trajectory(
                args.output_dir,
                args.output_prefix,
                record_data,
                image_data,
                camera_timestamps,
                metadata,
                timestamp=traj_timestamp,
            )
            print(f"Saved follower trajectory to {output_path}")
            return True, saved_count + 1
        else:
            print("Discarding trajectory. Ready to re-try.")
            if traj_video_dir is not None:
                shutil.rmtree(traj_video_dir, ignore_errors=True)
            return True, saved_count

    # Count existing trajectories for auto-resume
    existing_count = count_existing_trajectories(args.output_dir, args.output_prefix)

    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  Button 0: Toggle following (synchronized mode)")
    print("  Button 1: Toggle data recording (start/stop collection)")
    print("  Ctrl+C:   Exit the program")
    print("=" * 60)
    print(f"\nOutput directory: {args.output_dir}")
    print(f"Existing trajectories: {existing_count}")
    print(f"Target total: {args.num_trajectories}")
    if existing_count >= args.num_trajectories:
        print(f"\nAlready have {existing_count} trajectories (target: {args.num_trajectories}).")
        print("Use --num-trajectories to set a higher target, or delete existing files.")
        return
    print(f"Will collect {args.num_trajectories - existing_count} more trajectories.")
    print("=" * 60 + "\n")

    try:
        target_count = args.num_trajectories
        saved_count = existing_count  # Resume from existing count
        synchronized = False
        recording = False
        record_data, image_data, camera_timestamps, next_sample_time = init_buffers()
        record_gripper = None
        traj_timestamp = None
        traj_video_dir = None
        button0_was_pressed = False
        button1_was_pressed = False

        while saved_count < target_count:
            current_joint_pos, current_button = robot.get_info()

            # Button 0: Toggle synchronized (following) mode
            button0_pressed = current_button[0] > 0.5
            if button0_pressed and not button0_was_pressed:
                if not synchronized:
                    # Start following
                    robot.update_kp_kd(kp=robot_current_kp * args.bilateral_kp, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_joint_pos[:6])
                    slow_move(current_joint_pos)
                    synchronized = True
                    print("[Button 0] Following STARTED (synchronized)")
                else:
                    # Stop following
                    print("[Button 0] Following STOPPED")
                    robot.update_kp_kd(kp=np.ones(6) * 0.0, kd=np.ones(6) * 0.0)
                    robot.command_joint_pos(current_follower_joint_pos[:6])
                    synchronized = False
            button0_was_pressed = button0_pressed

            # Button 1: Toggle recording mode
            button1_pressed = current_button[1] > 0.5
            if button1_pressed and not button1_was_pressed:
                if not recording:
                    # Start recording
                    record_data, image_data, camera_timestamps, next_sample_time = init_buffers()
                    record_gripper = None
                    traj_timestamp = next_traj_tag(args.output_dir, args.output_prefix)
                    traj_video_dir = None
                    if args.record_video and camera_reader is not None:
                        traj_video_dir = os.path.join(args.output_dir, f"{args.output_prefix}_{traj_timestamp}")
                        camera_reader.start_recording(traj_video_dir)
                    recording = True
                    print(f"[Button 1] Recording STARTED (trajectory {saved_count + 1}/{target_count})")
                else:
                    # Stop recording and prompt to save
                    recording = False
                    print(f"[Button 1] Recording STOPPED ({len(record_data['timestamps_s'])} samples)")
                    wait_button_release(1)  # Wait for button release before prompting
                    _, saved_count = handle_recording_end(
                        record_data,
                        image_data,
                        camera_timestamps,
                        traj_timestamp,
                        traj_video_dir,
                        saved_count,
                    )
                    traj_timestamp = None
                    traj_video_dir = None
                    if saved_count >= target_count:
                        print(f"\nAll {target_count} trajectories collected!")
                        break
                    print(f"\nReady for next trajectory ({saved_count}/{target_count} saved)")
            button1_was_pressed = button1_pressed

            current_follower_joint_pos = client_robot.get_joint_pos()

            # Execute following if synchronized
            if synchronized:
                client_robot.command_joint_pos(current_joint_pos)
                # This will set the bilateral force in joint space proportional to the bilateral kp.
                robot.command_joint_pos(current_follower_joint_pos[:6])

            # Record data if recording and synchronized
            now = time.monotonic()
            if recording and synchronized and now >= next_sample_time:
                obs = client_robot.get_observations()
                images = {}
                camera_ts = {}
                if args.record_images and camera_reader is not None:
                    camera_obs, camera_ts = camera_reader.read_cameras()
                    images = camera_obs.get("image", {})
                step_index = len(record_data["timestamps_s"])
                for key in images:
                    if key not in image_data:
                        image_data[key] = [None] * step_index
                for key in image_data:
                    image_data[key].append(images.get(key))
                for key, value in camera_ts.items():
                    if key not in camera_timestamps:
                        camera_timestamps[key] = [None] * step_index
                    camera_timestamps[key].append(value)
                for key in camera_timestamps:
                    if key not in camera_ts:
                        camera_timestamps[key].append(None)

                record_data["timestamps_s"].append(time.time())
                record_data["leader_joint_pos"].append(current_joint_pos.copy())
                record_data["follower_joint_pos"].append(obs["joint_pos"].copy())
                record_data["follower_joint_vel"].append(obs["joint_vel"].copy())
                record_data["follower_joint_eff"].append(obs["joint_eff"].copy())
                if "gripper_pos" in obs:
                    if record_gripper is None:
                        record_gripper = True
                        record_data["follower_gripper_pos"] = []
                    record_data["follower_gripper_pos"].append(obs["gripper_pos"].copy())
                next_sample_time += args.record_dt

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\nKeyboard interrupt: exiting teleop.")
        # If we were recording, offer to save
        if recording and record_data["timestamps_s"]:
            print("Recording was in progress...")
            if camera_reader is not None and args.record_video:
                camera_reader.stop_recording()
            should_save = prompt_save(saved_count + 1)
            if should_save:
                metadata = {
                    "gripper": args.gripper,
                    "server_host": args.server_host,
                    "server_port": str(args.server_port),
                    "bilateral_kp": str(args.bilateral_kp),
                    "record_dt": str(args.record_dt),
                    "trajectory_index": str(saved_count + 1),
                }
                output_path = save_trajectory(
                    args.output_dir,
                    args.output_prefix,
                    record_data,
                    image_data,
                    camera_timestamps,
                    metadata,
                    timestamp=traj_timestamp,
                )
                print(f"Saved follower trajectory to {output_path}")
            else:
                print("Discarding trajectory.")
                if traj_video_dir is not None:
                    shutil.rmtree(traj_video_dir, ignore_errors=True)
    finally:
        if camera_reader is not None:
            camera_reader.disable_cameras()
        print("Teleop session ended.")


if __name__ == "__main__":
    main(tyro.cli(Args))

"""Add end effector poses to collected trajectory data using forward kinematics.

This script loads trajectory .npz files, computes end effector poses from joint
positions using forward kinematics, and saves the augmented data back.
"""

import glob
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
import tyro

# Lazy imports for robot dependencies (allows list/info modes without hardware deps)
if TYPE_CHECKING:
    from i2rt.robots.kinematics import Kinematics
    from i2rt.robots.utils import GripperType


def rot_to_extrinsic_xyz(rot: np.ndarray) -> np.ndarray:
    """Convert rotation matrix to extrinsic XYZ euler angles (roll, pitch, yaw)."""
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw])


# Coordinate frame transformation used by RealRobotEndEffectorEnv
# This flips X and Z axes to convert between MuJoCo frame and user frame
CANONICAL_AXIS_FLIP = np.diag([-1.0, 1.0, -1.0])


def raw_rot_to_user_rot(raw_rot: np.ndarray) -> np.ndarray:
    """Transform rotation from raw MuJoCo frame to user frame.
    
    This matches the transformation in RealRobotEndEffectorEnv.get_observation():
        ee_rot_user = axis_flip.T @ ee_rot @ axis_flip
    
    The user frame is what the policy sees during inference, so training data
    must use the same frame for consistency.
    """
    return CANONICAL_AXIS_FLIP.T @ raw_rot @ CANONICAL_AXIS_FLIP


def pose_matrix_to_pos_euler(
    pose: np.ndarray,
    apply_user_transform: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert 4x4 pose matrix to position and euler angles.
    
    Args:
        pose: 4x4 transformation matrix
        apply_user_transform: If True, apply the coordinate frame transformation
            used by RealRobotEndEffectorEnv to match observation/action frames.
            This should be True when preparing data for policy training.
    
    Returns:
        position: (3,) array [x, y, z]
        euler: (3,) array [roll, pitch, yaw] in radians
    """
    position = pose[:3, 3]
    rot = pose[:3, :3]
    
    if apply_user_transform:
        rot = raw_rot_to_user_rot(rot)
    
    euler = rot_to_extrinsic_xyz(rot)
    return position, euler


def compute_ee_poses(
    kinematics: "Kinematics",
    joint_positions: np.ndarray,
    site_name: str = "grasp_site",
    apply_user_transform: bool = True,
) -> dict[str, np.ndarray]:
    """Compute end effector poses for a sequence of joint positions.
    
    Args:
        kinematics: Kinematics object with FK capability
        joint_positions: (N, num_joints) array of joint positions
        site_name: Name of the end effector site
        apply_user_transform: If True, apply coordinate frame transformation to match
            RealRobotEndEffectorEnv.get_observation() output. This ensures consistency
            between training data and inference observations/actions.
        
    Returns:
        Dictionary with:
            - ee_pose_matrix: (N, 4, 4) pose matrices (raw MuJoCo frame, for reference)
            - ee_pos: (N, 3) positions [x, y, z]
            - ee_euler: (N, 3) euler angles [roll, pitch, yaw] (in user frame if transformed)
            - ee_pose_6d: (N, 6) combined [x, y, z, roll, pitch, yaw]
    """
    n_samples = len(joint_positions)
    num_arm_joints = min(joint_positions.shape[1], kinematics.nq)
    
    pose_matrices = np.zeros((n_samples, 4, 4))
    positions = np.zeros((n_samples, 3))
    eulers = np.zeros((n_samples, 3))
    
    for i in range(n_samples):
        # Use only arm joints (not gripper) for FK
        q = joint_positions[i, :num_arm_joints]
        pose = kinematics.fk(q, site_name)
        pose_matrices[i] = pose
        positions[i], eulers[i] = pose_matrix_to_pos_euler(pose, apply_user_transform)
    
    return {
        "ee_pose_matrix": pose_matrices,  # Raw MuJoCo frame (for debugging/reference)
        "ee_pos": positions,
        "ee_euler": eulers,
        "ee_pose_6d": np.concatenate([positions, eulers], axis=1),
    }


def process_trajectory_file(
    filepath: str,
    gripper_type: "GripperType",
    site_name: str = "grasp_site",
    apply_user_transform: bool = True,
    overwrite: bool = False,
    dry_run: bool = False,
) -> bool:
    """Process a single trajectory file to add end effector poses.
    
    Args:
        filepath: Path to the .npz file
        gripper_type: Type of gripper (determines robot model)
        site_name: Name of the end effector site
        apply_user_transform: If True, apply coordinate frame transformation to match
            RealRobotEndEffectorEnv. This should be True for policy training data.
        overwrite: Whether to overwrite existing EE pose data
        dry_run: If True, don't save changes
        
    Returns:
        True if file was modified, False otherwise
    """
    print(f"\nProcessing: {os.path.basename(filepath)}")
    
    # Load data
    data = dict(np.load(filepath, allow_pickle=True))
    
    # Check if already has EE poses
    has_leader_ee = "leader_ee_pose_6d" in data
    has_follower_ee = "follower_ee_pose_6d" in data
    
    if has_leader_ee and has_follower_ee and not overwrite:
        print("  Already has EE poses. Use --overwrite to recompute.")
        return False
    
    # Initialize kinematics (import here to allow list/info modes without deps)
    from i2rt.robots.kinematics import Kinematics
    
    xml_path = gripper_type.get_xml_path()
    kinematics = Kinematics(xml_path, site_name)
    print(f"  Using model: {os.path.basename(xml_path)}")
    print(f"  User frame transform: {'enabled' if apply_user_transform else 'disabled (raw MuJoCo frame)'}")
    
    modified = False
    
    # Process leader joint positions
    if "leader_joint_pos" in data:
        leader_joints = data["leader_joint_pos"]
        print(f"  Leader joint positions: {leader_joints.shape}")
        
        leader_ee = compute_ee_poses(kinematics, leader_joints, site_name, apply_user_transform)
        data["leader_ee_pose_matrix"] = leader_ee["ee_pose_matrix"]
        data["leader_ee_pos"] = leader_ee["ee_pos"]
        data["leader_ee_euler"] = leader_ee["ee_euler"]
        data["leader_ee_pose_6d"] = leader_ee["ee_pose_6d"]
        print(f"  Added leader EE poses: {leader_ee['ee_pose_6d'].shape}")
        modified = True
    
    # Process follower joint positions
    if "follower_joint_pos" in data:
        follower_joints = data["follower_joint_pos"]
        print(f"  Follower joint positions: {follower_joints.shape}")
        
        follower_ee = compute_ee_poses(kinematics, follower_joints, site_name, apply_user_transform)
        data["follower_ee_pose_matrix"] = follower_ee["ee_pose_matrix"]
        data["follower_ee_pos"] = follower_ee["ee_pos"]
        data["follower_ee_euler"] = follower_ee["ee_euler"]
        data["follower_ee_pose_6d"] = follower_ee["ee_pose_6d"]
        print(f"  Added follower EE poses: {follower_ee['ee_pose_6d'].shape}")
        modified = True
    
    if not modified:
        print("  No joint positions found in file.")
        return False
    
    # Save updated data
    if not dry_run:
        np.savez(filepath, **data)
        print(f"  Saved: {filepath}")
    else:
        print("  [DRY RUN] Would save changes.")
    
    return True


def list_trajectories(directory: str, prefix: str = "follower_traj") -> list[str]:
    """List all trajectory files in a directory."""
    pattern = os.path.join(directory, f"{prefix}_*.npz")
    files = sorted(glob.glob(pattern))
    return files


@dataclass
class Args:
    trajectory_file: str | None = None
    """Path to a specific trajectory file. If not provided, processes all in directory."""
    trajectory_dir: str = "data/teleop"
    """Directory containing trajectory files."""
    trajectory_prefix: str = "follower_traj"
    """Prefix for trajectory files."""
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "linear_4310"
    """Gripper type (determines robot model for FK)."""
    site_name: str = "grasp_site"
    """End effector site name in the robot model."""
    apply_user_transform: bool = True
    """Apply coordinate frame transform to match RealRobotEndEffectorEnv. 
    Enable this (default) for policy training data so observations/actions match inference."""
    overwrite: bool = False
    """Overwrite existing EE pose data if present."""
    dry_run: bool = False
    """Don't save changes, just show what would be done."""
    mode: Literal["process", "list", "info"] = "process"
    """Mode: process files, list available files, or show file info."""


def show_file_info(filepath: str) -> None:
    """Show information about a trajectory file."""
    print(f"\nFile: {os.path.basename(filepath)}")
    print("-" * 50)
    
    data = np.load(filepath, allow_pickle=True)
    
    # Show metadata
    if "metadata_json" in data:
        metadata = json.loads(data["metadata_json"][0])
        print("Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    print("\nArrays:")
    for key in sorted(data.keys()):
        if key == "metadata_json":
            continue
        arr = data[key]
        print(f"  {key}: shape={arr.shape}, dtype={arr.dtype}")
    
    # Check for EE poses
    has_leader_ee = "leader_ee_pose_6d" in data
    has_follower_ee = "follower_ee_pose_6d" in data
    print(f"\nEE poses computed: leader={has_leader_ee}, follower={has_follower_ee}")


def main(args: Args) -> None:
    if args.mode == "list":
        files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
        if not files:
            print(f"No trajectories found in {args.trajectory_dir}")
            return
        print(f"\nTrajectories in {args.trajectory_dir}:")
        print("-" * 50)
        for i, f in enumerate(files):
            # Quick check if has EE poses
            data = np.load(f, allow_pickle=True)
            has_ee = "leader_ee_pose_6d" in data or "follower_ee_pose_6d" in data
            status = "[EE ✓]" if has_ee else "[no EE]"
            print(f"  [{i}] {status} {os.path.basename(f)}")
        print(f"\nTotal: {len(files)} files")
        return
    
    if args.mode == "info":
        if args.trajectory_file:
            show_file_info(args.trajectory_file)
        else:
            files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
            for f in files[:5]:  # Show first 5
                show_file_info(f)
            if len(files) > 5:
                print(f"\n... and {len(files) - 5} more files")
        return
    
    # Process mode - import robot dependencies
    from i2rt.robots.utils import GripperType
    
    gripper_type = GripperType.from_string_name(args.gripper)
    
    if args.trajectory_file:
        files = [args.trajectory_file]
    else:
        files = list_trajectories(args.trajectory_dir, args.trajectory_prefix)
    
    if not files:
        print("No trajectory files found.")
        return
    
    print(f"Processing {len(files)} trajectory file(s)...")
    print(f"Gripper type: {args.gripper}")
    print(f"Site name: {args.site_name}")
    print(f"User frame transform: {'enabled (for policy training)' if args.apply_user_transform else 'disabled (raw MuJoCo frame)'}")
    if args.dry_run:
        print("[DRY RUN MODE - no files will be modified]")
    
    modified_count = 0
    for filepath in files:
        if process_trajectory_file(
            filepath,
            gripper_type,
            site_name=args.site_name,
            apply_user_transform=args.apply_user_transform,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        ):
            modified_count += 1
    
    print(f"\n{'=' * 50}")
    print(f"Done. Modified {modified_count}/{len(files)} files.")


if __name__ == "__main__":
    main(tyro.cli(Args))

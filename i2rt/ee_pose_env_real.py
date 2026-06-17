import time
from dataclasses import dataclass
from enum import Enum
from typing import Literal, Optional, Sequence

import mujoco
import numpy as np

from i2rt.robots.get_robot import get_yam_robot
from i2rt.robots.utils import GripperType
from i2rt.cameras.multi_camera_wrapper import MultiCameraWrapper


class TrajectoryProfile(Enum):
    """Available trajectory interpolation profiles."""
    LINEAR = "linear"
    CUBIC = "cubic"  # Cubic polynomial (smooth start/end velocity)
    QUINTIC = "quintic"  # Quintic polynomial (smooth start/end velocity & acceleration)
    MINIMUM_JERK = "minimum_jerk"  # Minimizes jerk for smoothest motion
    TRAPEZOIDAL = "trapezoidal"  # Trapezoidal velocity profile
    S_CURVE = "s_curve"  # S-curve (smoothed trapezoidal)


@dataclass
class TrajectoryConfig:
    """Configuration for trajectory generation."""
    profile: TrajectoryProfile = TrajectoryProfile.MINIMUM_JERK
    # Velocity and acceleration limits (rad/s and rad/s^2)
    max_velocity: float = 2.0  # Maximum joint velocity
    max_acceleration: float = 5.0  # Maximum joint acceleration
    # For trapezoidal/s-curve profiles
    accel_fraction: float = 0.25  # Fraction of time spent accelerating (0-0.5)
    # Feedback correction
    use_feedback: bool = True
    feedback_gain: float = 0.3  # Gain for feedback correction


class TrajectoryInterpolator:
    """
    Advanced trajectory interpolator with multiple smooth profiles.
    
    Supports:
    - Linear: Simple linear interpolation (jerky)
    - Cubic: 3rd order polynomial with zero velocity at endpoints
    - Quintic: 5th order polynomial with zero velocity and acceleration at endpoints
    - Minimum Jerk: Minimizes jerk for the smoothest possible motion
    - Trapezoidal: Constant acceleration/deceleration with cruise phase
    - S-Curve: Smoothed trapezoidal with limited jerk
    """
    
    def __init__(self, config: TrajectoryConfig = None):
        self.config = config or TrajectoryConfig()
    
    def _linear_profile(self, t_normalized: float) -> float:
        """Linear interpolation: s(t) = t"""
        return t_normalized
    
    def _cubic_profile(self, t_normalized: float) -> float:
        """
        Cubic polynomial: s(t) = 3t^2 - 2t^3
        - Zero velocity at start and end
        - Continuous acceleration (but non-zero at endpoints)
        """
        t = t_normalized
        return 3 * t**2 - 2 * t**3
    
    def _quintic_profile(self, t_normalized: float) -> float:
        """
        Quintic polynomial: s(t) = 6t^5 - 15t^4 + 10t^3
        - Zero velocity at start and end
        - Zero acceleration at start and end
        """
        t = t_normalized
        return 6 * t**5 - 15 * t**4 + 10 * t**3
    
    def _minimum_jerk_profile(self, t_normalized: float) -> float:
        """
        Minimum jerk trajectory: s(t) = 10t^3 - 15t^4 + 6t^5
        
        This is the smoothest possible trajectory that:
        - Starts and ends at rest (zero velocity)
        - Has zero acceleration at start and end
        - Minimizes the integral of jerk squared
        
        Widely used in robotics and models human arm movements well.
        """
        t = t_normalized
        return 10 * t**3 - 15 * t**4 + 6 * t**5
    
    def _trapezoidal_profile(self, t_normalized: float) -> float:
        """
        Trapezoidal velocity profile.
        
        Three phases:
        1. Acceleration (constant acceleration)
        2. Cruise (constant velocity)
        3. Deceleration (constant deceleration)
        
        Has discontinuous acceleration at phase transitions.
        """
        t = t_normalized
        ta = self.config.accel_fraction  # Acceleration phase duration
        
        if ta >= 0.5:
            # No cruise phase - triangular profile
            if t < 0.5:
                return 2 * t**2
            else:
                return 1 - 2 * (1 - t)**2
        
        td = 1.0 - ta  # Deceleration start time
        
        if t < ta:
            # Acceleration phase
            return (t**2) / (2 * ta * (1 - ta))
        elif t < td:
            # Cruise phase
            return (t - ta / 2) / (1 - ta)
        else:
            # Deceleration phase
            return 1 - ((1 - t)**2) / (2 * ta * (1 - ta))
    
    def _s_curve_profile(self, t_normalized: float) -> float:
        """
        S-curve (7-segment) profile with smooth jerk.
        
        Uses sinusoidal blending to smooth the trapezoidal profile,
        eliminating discontinuities in acceleration.
        
        Seven phases with limited jerk throughout.
        """
        t = t_normalized
        ta = self.config.accel_fraction
        
        if ta >= 0.5:
            # Simplified S-curve for short moves
            return 0.5 * (1 - np.cos(np.pi * t))
        
        td = 1.0 - ta
        
        # Use sinusoidal blending for smooth acceleration
        if t < ta:
            # Acceleration phase with sinusoidal jerk
            phase = np.pi * t / ta
            return (ta / (2 * (1 - ta))) * (t / ta - np.sin(phase) / np.pi)
        elif t < td:
            # Cruise phase
            accel_contrib = ta / (2 * (1 - ta))
            return accel_contrib + (t - ta) / (1 - ta)
        else:
            # Deceleration phase with sinusoidal jerk
            t_decel = (t - td) / ta
            phase = np.pi * t_decel
            decel_contrib = (ta / (2 * (1 - ta))) * (t_decel - np.sin(phase) / np.pi)
            cruise_end = 1 - ta / (2 * (1 - ta))
            return cruise_end + decel_contrib
    
    def get_interpolation_factor(self, t_normalized: float) -> float:
        """
        Get the interpolation factor s(t) for the configured profile.
        
        Args:
            t_normalized: Normalized time in [0, 1]
            
        Returns:
            Interpolation factor in [0, 1]
        """
        t = np.clip(t_normalized, 0.0, 1.0)
        
        profile_map = {
            TrajectoryProfile.LINEAR: self._linear_profile,
            TrajectoryProfile.CUBIC: self._cubic_profile,
            TrajectoryProfile.QUINTIC: self._quintic_profile,
            TrajectoryProfile.MINIMUM_JERK: self._minimum_jerk_profile,
            TrajectoryProfile.TRAPEZOIDAL: self._trapezoidal_profile,
            TrajectoryProfile.S_CURVE: self._s_curve_profile,
        }
        
        return profile_map[self.config.profile](t)
    
    def interpolate(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        t_normalized: float,
    ) -> np.ndarray:
        """
        Interpolate between start and end positions.
        
        Args:
            start_pos: Starting joint positions
            end_pos: Target joint positions
            t_normalized: Normalized time in [0, 1]
            
        Returns:
            Interpolated joint positions
        """
        s = self.get_interpolation_factor(t_normalized)
        return (1.0 - s) * start_pos + s * end_pos
    
    def compute_velocity(self, t_normalized: float, duration: float) -> float:
        """
        Compute the velocity scaling factor ds/dt at time t.
        
        Args:
            t_normalized: Normalized time in [0, 1]
            duration: Total trajectory duration
            
        Returns:
            Velocity scaling factor
        """
        # Numerical derivative
        eps = 1e-6
        if t_normalized < eps:
            s1 = self.get_interpolation_factor(0.0)
            s2 = self.get_interpolation_factor(eps)
        elif t_normalized > 1.0 - eps:
            s1 = self.get_interpolation_factor(1.0 - eps)
            s2 = self.get_interpolation_factor(1.0)
        else:
            s1 = self.get_interpolation_factor(t_normalized - eps)
            s2 = self.get_interpolation_factor(t_normalized + eps)
        
        return (s2 - s1) / (2 * eps * duration)
    
    def compute_duration(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        min_duration: float = 0.1,
        max_duration: float = 5.0,
    ) -> float:
        """
        Compute appropriate duration based on joint displacement and limits.
        
        Args:
            start_pos: Starting joint positions
            end_pos: Target joint positions
            min_duration: Minimum allowed duration
            max_duration: Maximum allowed duration
            
        Returns:
            Computed duration in seconds
        """
        delta = np.abs(end_pos - start_pos)
        max_delta = np.max(delta)
        
        if max_delta < 1e-6:
            return min_duration
        
        # Duration based on velocity limit
        # For minimum jerk, peak velocity is 1.875 * (delta / T)
        # So T >= 1.875 * delta / v_max
        velocity_factor = 1.875 if self.config.profile == TrajectoryProfile.MINIMUM_JERK else 1.5
        duration_vel = velocity_factor * max_delta / self.config.max_velocity
        
        # Duration based on acceleration limit
        # For minimum jerk, peak acceleration is 5.77 * delta / T^2
        # So T >= sqrt(5.77 * delta / a_max)
        accel_factor = 5.77 if self.config.profile == TrajectoryProfile.MINIMUM_JERK else 4.0
        duration_accel = np.sqrt(accel_factor * max_delta / self.config.max_acceleration)
        
        duration = max(duration_vel, duration_accel)
        return np.clip(duration, min_duration, max_duration)
    
    def generate_trajectory(
        self,
        start_pos: np.ndarray,
        end_pos: np.ndarray,
        duration: float,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a complete trajectory.
        
        Args:
            start_pos: Starting joint positions
            end_pos: Target joint positions
            duration: Trajectory duration in seconds
            dt: Time step in seconds
            
        Returns:
            Tuple of (positions, times) arrays
        """
        num_steps = max(int(duration / dt), 1)
        times = np.linspace(0, duration, num_steps + 1)
        positions = np.zeros((num_steps + 1, len(start_pos)))
        
        for i, t in enumerate(times):
            t_normalized = t / duration
            positions[i] = self.interpolate(start_pos, end_pos, t_normalized)
        
        return positions, times


def rot_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def extrinsic_xyz_to_rot(euler_xyz: np.ndarray) -> np.ndarray:
    rx, ry, rz = euler_xyz.tolist()
    return rot_z(rz) @ rot_y(ry) @ rot_x(rx)


def rot_to_extrinsic_xyz(rot: np.ndarray) -> np.ndarray:
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    return np.array([roll, pitch, yaw])


def rot_to_axis_angle(rot: np.ndarray) -> np.ndarray:
    trace = np.clip((np.trace(rot) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(trace))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            rot[2, 1] - rot[1, 2],
            rot[0, 2] - rot[2, 0],
            rot[1, 0] - rot[0, 1],
        ]
    )
    axis = axis / (2.0 * np.sin(angle))
    return axis * angle


@dataclass
class EnvConfig:
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "linear_4310"
    site_name: str = "grasp_site"
    qpos_init: Sequence[float] = (0.0, 1.0, 1.0, -1.5, 0.0, 0)
    pre_qpos: Sequence[float] = (0.0, 0.0, 0.3, 0.0, 0.0, 0.0)
    action_space: Literal["end_effector_pose", "joint_position"] = "end_effector_pose"
    rotation_frame: Literal["local", "world"] = "world"
    control_mode: Literal["delta", "absolute"] = "absolute"
    damping: float = 0.05
    max_iters: int = 200
    channel: str = "can0"
    debug_rotation: bool = False
    debug_ik: bool = False
    # Trajectory interpolation settings
    trajectory_profile: str = "minimum_jerk"  # linear, cubic, quintic, minimum_jerk, trapezoidal, s_curve
    trajectory_max_velocity: float = 2.0  # rad/s
    trajectory_max_acceleration: float = 5.0  # rad/s^2
    trajectory_use_feedback: bool = True
    trajectory_feedback_gain: float = 0.3
    # Initial qpos randomization
    qpos_init_randomize: bool = False
    qpos_init_noise_scale: float = 0.0  # Uniform noise range: [-scale, scale] per joint (radians)
    # IK regularization to avoid singularity (joints 0,1,2 collinear)
    ik_joint2_regularization: float = 0.0  # Weight to keep joint 2 near initial position in IK null space
    # Maximum joint position delta per step (radians) - caps joint movement for safety
    max_joint_delta: float = 100 # ~5.7 degrees default per joint


class RealRobotEndEffectorEnv:
    def __init__(self, 
            config: EnvConfig,
            default_resolution={"web": (640, 480), "rs": (640, 480)},
            resize_resolution={"web": (0, 0), "rs": (0, 0)},
        ) -> None:
        self.camera_reader = MultiCameraWrapper(
                default_resolution=default_resolution, resize_resolution=resize_resolution
        )
        gripper_type = GripperType.from_string_name(config.gripper)
        self.robot = get_yam_robot(channel=config.channel, gripper_type=gripper_type)

        self.model = mujoco.MjModel.from_xml_path(gripper_type.get_xml_path())
        self.data = mujoco.MjData(self.model)
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, config.site_name)
        if self.site_id < 0:
            raise ValueError(f"Unknown site name: {config.site_name}")

        self.arm_dofs = 6
        self.action_space = config.action_space
        self.rotation_frame = config.rotation_frame
        self.control_mode = config.control_mode
        self.debug_rotation = bool(config.debug_rotation)
        self.debug_ik = bool(config.debug_ik)
        self._canonical_base_rot = np.eye(3)
        self._canonical_axis_flip = np.diag([-1.0, 1.0, -1.0])
        qpos_init = np.array(config.qpos_init, dtype=float)
        if qpos_init.shape[0] != self.arm_dofs:
            raise ValueError(f"qpos_init must have {self.arm_dofs} values, got {qpos_init.shape[0]}")

        # Initialize trajectory interpolator
        trajectory_profile = TrajectoryProfile(config.trajectory_profile)
        self.trajectory_config = TrajectoryConfig(
            profile=trajectory_profile,
            max_velocity=config.trajectory_max_velocity,
            max_acceleration=config.trajectory_max_acceleration,
            use_feedback=config.trajectory_use_feedback,
            feedback_gain=config.trajectory_feedback_gain,
        )
        self.trajectory_interpolator = TrajectoryInterpolator(self.trajectory_config)

        # IK regularization to avoid singularity (joints 0,1,2 collinear)
        self.ik_joint2_regularization = config.ik_joint2_regularization
        self.qpos_reference = qpos_init.copy()  # Will be updated in reset()

        # Maximum joint position delta per step (radians) - caps joint movement for safety
        self.max_joint_delta = config.max_joint_delta

        self.reset(config)
        self.damping = float(config.damping)
        self.max_iters = int(config.max_iters)
        self.action_translation_scale = 1

    def _site_pose(self, arm_qpos: np.ndarray) -> np.ndarray:
        self.data.qpos[: self.arm_dofs] = arm_qpos
        mujoco.mj_forward(self.model, self.data)
        pose = np.eye(4)
        pose[:3, :3] = self.data.site_xmat[self.site_id].reshape(3, 3)
        pose[:3, 3] = self.data.site_xpos[self.site_id]
        return pose


    def _move_joints(
        self,
        target_joint_pos: np.ndarray,
        duration_s: Optional[float] = None,
        steps: Optional[int] = None,
        dt: float = 0.01,
    ) -> None:
        """
        Move joints smoothly using the configured trajectory interpolator.
        
        Args:
            target_joint_pos: Target joint positions
            duration_s: Optional duration override. If None, computed automatically
                        based on displacement and velocity/acceleration limits.
            steps: Optional number of steps override. If None, computed from duration and dt.
            dt: Time step for trajectory execution (default 10ms for 100Hz control)
        """
        start_pos = self.robot.get_joint_pos()
        if start_pos.shape[0] != target_joint_pos.shape[0]:
            raise ValueError(
                f"Expected target_joint_pos length {start_pos.shape[0]}, got {target_joint_pos.shape[0]}"
            )
        
        # Compute duration if not specified
        if duration_s is None:
            duration_s = self.trajectory_interpolator.compute_duration(
                start_pos, target_joint_pos, min_duration=0.1, max_duration=5.0
            )
        
        # Compute steps if not specified
        if steps is None:
            steps = max(int(duration_s / dt), 10)
        
        actual_dt = duration_s / steps
        
        # Pre-generate trajectory for reference
        planned_positions, planned_times = self.trajectory_interpolator.generate_trajectory(
            start_pos, target_joint_pos, duration_s, actual_dt
        )
        
        # Execute trajectory with optional feedback correction
        use_feedback = self.trajectory_config.use_feedback
        feedback_gain = self.trajectory_config.feedback_gain
        
        start_time = time.perf_counter()
        
        for i in range(steps + 1):
            loop_start = time.perf_counter()
            
            # Compute elapsed time and normalized time
            elapsed = loop_start - start_time
            t_normalized = min(elapsed / duration_s, 1.0)
            
            # Get planned position from trajectory
            planned_pos = self.trajectory_interpolator.interpolate(
                start_pos, target_joint_pos, t_normalized
            )
            
            if use_feedback and i > 0:
                # Apply feedback correction based on actual position
                actual_pos = self.robot.get_joint_pos()
                tracking_error = planned_pos - actual_pos
                
                # Blend planned trajectory with feedback correction
                # This helps compensate for execution delays and model errors
                command_pos = planned_pos + feedback_gain * tracking_error
            else:
                command_pos = planned_pos
            
            self.robot.command_joint_pos(command_pos)
            
            # Maintain consistent timing
            loop_elapsed = time.perf_counter() - loop_start
            sleep_time = actual_dt - loop_elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Final command to ensure we reach the target
        self.robot.command_joint_pos(target_joint_pos)
    
    def read_cameras(self):
        return self.camera_reader.read_cameras()

    def reset(self, config: EnvConfig=EnvConfig()) -> None:
        qpos_init = np.array(config.qpos_init, dtype=float)
        pre_qpos = np.array(config.pre_qpos, dtype=float)
        if qpos_init.shape[0] != self.arm_dofs:
            raise ValueError(f"qpos_init must have {self.arm_dofs} values, got {qpos_init.shape[0]}")
        if pre_qpos.shape[0] != self.arm_dofs:
            raise ValueError(f"pre_qpos must have {self.arm_dofs} values, got {pre_qpos.shape[0]}")

        # Apply randomization to initial qpos if enabled
        if config.qpos_init_randomize:
            noise = np.random.uniform(
                -config.qpos_init_noise_scale,
                config.qpos_init_noise_scale,
                size=qpos_init.shape
            )
            qpos_init = qpos_init + noise

        # Store reference qpos for IK regularization (after randomization)
        self.qpos_reference = qpos_init.copy()

        def build_target(arm_qpos: np.ndarray, grip_pos: Optional[float]) -> np.ndarray:
            if grip_pos is not None:
                return np.concatenate([arm_qpos, [grip_pos]])
            return arm_qpos

        self._move_joints(build_target(pre_qpos, 1))
        self._move_joints(build_target(qpos_init, 1))
        # self._move_joints(build_target(qpos_init, 1), duration_s=1.0, steps=40)
        obs = self.robot.get_observations()
        arm_qpos = obs["joint_pos"].copy()
        # self._canonical_base_rot = self._site_pose(arm_qpos)[:3, :3].copy()


    def close(self) -> None:
        # Move to zero position with a minimum duration for safety
        self._move_joints(np.zeros_like(self.robot.get_joint_pos()), duration_s=2.5)

    def _solve_ik_step(self, target_pose: np.ndarray) -> np.ndarray:
        current = self._site_pose(self.data.qpos[: self.arm_dofs])
        pos_err = target_pose[:3, 3] - current[:3, 3]
        rot_err = target_pose[:3, :3] @ current[:3, :3].T
        ori_err = rot_to_axis_angle(rot_err)
        err = np.concatenate([pos_err, ori_err])

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.site_id)
        jac = np.vstack([jacp, jacr])[:, : self.arm_dofs]

        jj_t = jac @ jac.T
        damped_inv = np.linalg.solve(jj_t + (self.damping**2) * np.eye(6), err)
        dq = jac.T @ damped_inv

        # Null-space regularization to keep joint 2 near reference (avoids singularity)
        if self.ik_joint2_regularization > 0:
            # Compute pseudo-inverse: J^+ = J^T (J J^T + λ²I)^{-1}
            jac_pinv = jac.T @ np.linalg.inv(jj_t + (self.damping**2) * np.eye(6))
            # Null-space projector: N = I - J^+ J
            null_space_proj = np.eye(self.arm_dofs) - jac_pinv @ jac
            # Secondary objective: pull joint 2 toward reference
            q_secondary = np.zeros(self.arm_dofs)
            q_secondary[2] = self.ik_joint2_regularization * (
                self.qpos_reference[2] - self.data.qpos[2]
            )
            # Add null-space contribution
            dq += null_space_proj @ q_secondary

        return dq

    def step(self, action: np.ndarray) -> None:
        action = np.asarray(action, dtype=float).reshape(7)
        target_gripper_pos = float(action[6])

        if self.action_space == "joint_position":
            # Direct joint position control: action is [joint0, ..., joint5, gripper]
            target_arm_qpos = action[:self.arm_dofs].copy()
        elif self.action_space == "end_effector_pose":
            # End-effector pose control with IK: action is [x, y, z, roll, pitch, yaw, gripper]
            target_arm_qpos = self._solve_ik_from_action(action)
        else:
            raise ValueError(f"Unknown action_space: {self.action_space}")

        # Cap joint position delta to max_joint_delta for safety
        current_arm_qpos = self.robot.get_joint_pos()[:self.arm_dofs]
        delta_qpos = target_arm_qpos - current_arm_qpos
        delta_qpos = np.clip(delta_qpos, -self.max_joint_delta, self.max_joint_delta)
        target_arm_qpos = current_arm_qpos + delta_qpos

        target_joint_pos = np.concatenate([target_arm_qpos, [target_gripper_pos]])
        current_joint_pos = self.robot.get_joint_pos().copy()
        current_gripper_pos = current_joint_pos[-1]
        current_gripper_pos = float(current_gripper_pos > 0.5)
        gripper_move = np.abs(target_gripper_pos - current_gripper_pos) > 0.5

        if not gripper_move:
            # Use automatic duration computation based on trajectory limits
            # This ensures smooth motion respecting velocity and acceleration constraints
            # self._move_joints(target_joint_pos)
            self.robot.command_joint_pos(target_joint_pos)
        else:
            # Gripper movements use longer duration for safety
            self._move_joints(target_joint_pos, duration_s=0.5)

    def _solve_ik_from_action(self, action: np.ndarray) -> np.ndarray:
        """Solve IK for end-effector pose action and return target arm joint positions."""
        target_xyz = action[:3]
        target_euler_xyz = action[3:6]

        obs = self.robot.get_observations()
        arm_qpos = obs["joint_pos"].copy()
        current_pose = self._site_pose(arm_qpos)

        target_pose = np.eye(4)
        delta_pose = np.eye(4)
        if self.control_mode == "delta":
            delta_rot_user = extrinsic_xyz_to_rot(target_euler_xyz)
            delta_rot_canon = (
                self._canonical_axis_flip @ delta_rot_user @ self._canonical_axis_flip.T
            )
            delta_rot = self._canonical_base_rot @ delta_rot_canon @ self._canonical_base_rot.T
            delta_xyz_world = target_xyz
        elif self.control_mode == "absolute":
            target_rot_user = extrinsic_xyz_to_rot(target_euler_xyz)
            target_rot_canon = (
                self._canonical_axis_flip @ target_rot_user @ self._canonical_axis_flip.T
            )
            target_rot = self._canonical_base_rot @ target_rot_canon
            target_pos = target_xyz
            delta_xyz_world = target_pos - current_pose[:3, 3]
            if self.rotation_frame == "world":
                delta_rot = target_rot @ current_pose[:3, :3].T
            elif self.rotation_frame == "local":
                delta_rot = current_pose[:3, :3].T @ target_rot
            else:
                raise ValueError(f"Unknown rotation_frame: {self.rotation_frame}")
        else:
            raise ValueError(f"Unknown control_mode: {self.control_mode}")

        delta_pose[:3, :3] = delta_rot
        delta_pose[:3, 3] = delta_xyz_world

        if self.rotation_frame == "world":
            target_rot = delta_pose[:3, :3] @ current_pose[:3, :3]
        elif self.rotation_frame == "local":
            target_rot = current_pose[:3, :3] @ delta_pose[:3, :3]
        else:
            raise ValueError(f"Unknown rotation_frame: {self.rotation_frame}")
        target_pos = current_pose[:3, 3] + delta_xyz_world * self.action_translation_scale
        target_pose[:3, :3] = target_rot
        target_pose[:3, 3] = target_pos

        if self.debug_rotation and np.linalg.norm(target_euler_xyz) > 0:
            axis_angle = rot_to_axis_angle(delta_rot)
            site_axes = current_pose[:3, :3]
            print(
                "rot_dbg",
                f"input_rpy={target_euler_xyz.tolist()}",
                f"axis_world={axis_angle.tolist()}",
                f"rot_frame={self.rotation_frame}",
                f"site_x={site_axes[:, 0].tolist()}",
                f"site_y={site_axes[:, 1].tolist()}",
                f"site_z={site_axes[:, 2].tolist()}",
            )

        self.data.qpos[: self.arm_dofs] = arm_qpos
        for _ in range(self.max_iters):
            dq = self._solve_ik_step(target_pose)
            self.data.qpos[: self.arm_dofs] += dq[: self.arm_dofs]
            mujoco.mj_forward(self.model, self.data)

            current = self._site_pose(self.data.qpos[: self.arm_dofs])
            pos_err = np.linalg.norm(target_pose[:3, 3] - current[:3, 3])
            rot_err = np.linalg.norm(rot_to_axis_angle(target_pose[:3, :3] @ current[:3, :3].T))
            if self.debug_ik:
                print("ik_step", f"pos_err={pos_err:.6f}", f"rot_err={rot_err:.6f}", f"dq_norm={np.linalg.norm(dq):.6f}")
            if pos_err < 1e-4 and rot_err < 1e-4:
                break
        if self.debug_ik:
            final_err = rot_to_axis_angle(target_pose[:3, :3] @ current[:3, :3].T)
            print(
                "ik_final",
                f"pos_err={pos_err:.6f}",
                f"rot_err={rot_err:.6f}",
                f"axis_err={final_err.tolist()}",
            )

        return self.data.qpos[: self.arm_dofs].copy()

    def get_observation(self) -> dict:
        obs = self.robot.get_observations()
        arm_qpos = obs["joint_pos"].copy()
        pose = self._site_pose(arm_qpos)
        ee_xyz = pose[:3, 3]
        ee_rot = pose[:3, :3]
        ee_rot_canon = self._canonical_base_rot.T @ ee_rot
        # ee_euler_xyz = rot_to_extrinsic_xyz(ee_rot_canon)
        ee_rot_user = self._canonical_axis_flip.T @ ee_rot_canon @ self._canonical_axis_flip
        ee_euler_xyz = rot_to_extrinsic_xyz(ee_rot_user)
        gripper_pos = float(obs["gripper_pos"][0]) if "gripper_pos" in obs else 0.0

        camera_obs, _ = self.read_cameras()
        return {
            "robot_state": {
                "cartesian_position": np.concatenate([ee_xyz, ee_euler_xyz]),
                "gripper_position": gripper_pos,
                "joint_positions": arm_qpos,
            },
            "image": camera_obs["image"],
        }


def main() -> None:
    env = RealRobotEndEffectorEnv(EnvConfig())
    print("Robot initialized to qpos_init. Waiting for commands.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Exiting.")
    finally:
        env.close()


if __name__ == "__main__":
    main()

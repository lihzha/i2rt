import queue
import threading
import time
from dataclasses import dataclass
from typing import List, Literal, Optional

import mujoco
import mujoco.viewer
import numpy as np
import tyro

from i2rt.robots.utils import GripperType


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


def site_pose(model: mujoco.MjModel, data: mujoco.MjData, site_id: int) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, :3] = data.site_xmat[site_id].reshape(3, 3)
    pose[:3, 3] = data.site_xpos[site_id]
    return pose


def solve_ik_step(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    site_id: int,
    target_pose: np.ndarray,
    damping: float,
) -> np.ndarray:
    current = site_pose(model, data, site_id)
    pos_err = target_pose[:3, 3] - current[:3, 3]
    rot_err = target_pose[:3, :3] @ current[:3, :3].T
    ori_err = rot_to_axis_angle(rot_err)
    err = np.concatenate([pos_err, ori_err])

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
    jac = np.vstack([jacp, jacr])

    jj_t = jac @ jac.T
    dq = jac.T @ np.linalg.solve(jj_t + (damping**2) * np.eye(6), err)
    return dq


def print_pose(pose: np.ndarray) -> None:
    pos = pose[:3, 3]
    rot = pose[:3, :3]
    roll = np.arctan2(rot[2, 1], rot[2, 2])
    pitch = np.arctan2(-rot[2, 0], np.sqrt(rot[2, 1] ** 2 + rot[2, 2] ** 2))
    yaw = np.arctan2(rot[1, 0], rot[0, 0])
    print(
        f"pos: [{pos[0]:+.4f}, {pos[1]:+.4f}, {pos[2]:+.4f}]  "
        f"rpy: [{roll:+.4f}, {pitch:+.4f}, {yaw:+.4f}]"
    )


def input_thread(queue_out: "queue.Queue[np.ndarray]") -> None:
    print("Enter target pose: x y z roll pitch yaw (radians). Empty line to repeat prompt.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        if len(parts) != 6:
            print("Expected 6 values: x y z roll pitch yaw")
            continue
        try:
            values = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            print("Invalid number format.")
            continue
        queue_out.put(values)


def qpos_thread(queue_out: "queue.Queue[np.ndarray]", expected_len: int) -> None:
    print(f"Enter target qpos ({expected_len} values). Empty line to repeat prompt.")
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        parts = line.split()
        if len(parts) != expected_len:
            print(f"Expected {expected_len} values.")
            continue
        try:
            values = np.array([float(p) for p in parts], dtype=float)
        except ValueError:
            print("Invalid number format.")
            continue
        queue_out.put(values)


@dataclass
class Args:
    gripper: Literal["crank_4310", "linear_3507", "linear_4310", "yam_teaching_handle", "no_gripper"] = "linear_4310"
    site_name: str = "grasp_site"
    mode: Literal["input", "drag", "qpos"] = "input"
    rate_hz: float = 60.0
    damping: float = 0.05
    max_iters: int = 200
    print_hz: float = 5.0
    qpos: Optional[List[float]] = None
    qpos_len: Optional[int] = None
    show_site_marker: bool = True
    marker_size: float = 0.01


def main(args: Args) -> None:
    gripper_type = GripperType.from_string_name(args.gripper)
    model = mujoco.MjModel.from_xml_path(gripper_type.get_xml_path())
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, args.site_name)
    if site_id < 0:
        raise ValueError(f"Unknown site name: {args.site_name}")

    if args.qpos is not None:
        qpos = np.array(args.qpos, dtype=float)
        if qpos.shape[0] != model.nq:
            raise ValueError(f"Expected qpos length {model.nq}, got {qpos.shape[0]}")
        data.qpos[: model.nq] = qpos

    mujoco.mj_forward(model, data)
    print(f"Initial qpos: {data.qpos[: model.nq].tolist()}")

    def update_site_marker(viewer) -> None:
        if not args.show_site_marker:
            return
        pos = data.site_xpos[site_id]
        if viewer.user_scn.ngeom < 1:
            viewer.user_scn.ngeom = 1
        geom = viewer.user_scn.geoms[0]
        mujoco.mjv_initGeom(
            geom,
            mujoco.mjtGeom.mjGEOM_SPHERE,
            (args.marker_size * np.ones(3)).astype(np.float64),
            pos.astype(np.float64),
            np.eye(3, dtype=np.float64).reshape(9),
            np.array([1.0, 0.1, 0.1, 0.8], dtype=np.float32),
        )
    target_pose: Optional[np.ndarray] = None
    dt = 1.0 / args.rate_hz
    last_print = 0.0
    pending_print = False

    if args.mode == "input":
        pose_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        thread = threading.Thread(target=input_thread, args=(pose_queue,), daemon=True)
        thread.start()

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()
                while not pose_queue.empty():
                    vals = pose_queue.get()
                    delta_pose = np.eye(4)
                    delta_pose[:3, :3] = extrinsic_xyz_to_rot(vals[3:])
                    delta_pose[:3, 3] = vals[:3]
                    target_pose = delta_pose
                    pending_print = True

                if target_pose is not None:
                    converged = False
                    for _ in range(args.max_iters):
                        dq = solve_ik_step(model, data, site_id, target_pose, args.damping)
                        data.qpos[: model.nq] += dq[: model.nq]
                        mujoco.mj_forward(model, data)
                        current = site_pose(model, data, site_id)
                        pos_err = np.linalg.norm(target_pose[:3, 3] - current[:3, 3])
                        rot_err = np.linalg.norm(
                            rot_to_axis_angle(target_pose[:3, :3] @ current[:3, :3].T)
                        )
                        if pos_err < 1e-4 and rot_err < 1e-4:
                            converged = True
                            break
                    if pending_print and converged:
                        print_pose(site_pose(model, data, site_id))
                        pending_print = False

                update_site_marker(viewer)
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    elif args.mode == "qpos":
        qpos_queue: "queue.Queue[np.ndarray]" = queue.Queue()
        expected_len = args.qpos_len if args.qpos_len is not None else model.nq
        thread = threading.Thread(target=qpos_thread, args=(qpos_queue, expected_len), daemon=True)
        thread.start()

        with mujoco.viewer.launch_passive(
            model=model,
            data=data,
            show_left_ui=False,
            show_right_ui=False,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE

            while viewer.is_running():
                step_start = time.time()
                while not qpos_queue.empty():
                    qpos = qpos_queue.get()
                    data.qpos[: model.nq] = qpos
                    mujoco.mj_forward(model, data)
                    print_pose(site_pose(model, data, site_id))

                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
    else:
        with mujoco.viewer.launch(
            model=model,
            data=data,
            show_left_ui=True,
            show_right_ui=True,
        ) as viewer:
            mujoco.mjv_defaultFreeCamera(model, viewer.cam)
            viewer.opt.frame = mujoco.mjtFrame.mjFRAME_SITE
            print("Use the viewer controls to drag joints/bodies. Pose will print periodically.")

            while viewer.is_running():
                step_start = time.time()
                mujoco.mj_step(model, data)

                if time.time() - last_print > 1.0 / args.print_hz:
                    print_pose(site_pose(model, data, site_id))
                    last_print = time.time()

                update_site_marker(viewer)
                viewer.sync()
                time_until_next_step = dt - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)


if __name__ == "__main__":
    main(tyro.cli(Args))

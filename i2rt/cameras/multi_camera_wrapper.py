import os
import random
from collections import defaultdict

from i2rt.cameras.web_camera import gather_cameras
# from i2rt.cameras.zed_camera import ZedCamera, gather_zed_cameras


# Camera ID's #
hand_camera_id = "1"
right_camera_id = ""
left_camera_id = ""

camera_type_dict = {
    hand_camera_id: 3,
    right_camera_id: 0,
    left_camera_id: 2,
}

camera_type_to_string_dict = {
    3: "hand_camera",
    0: "right_camera",
    2: "left_camera",
}

def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    type_str = camera_type_to_string_dict[type_int]
    return type_str


class MultiCameraWrapper:
    def __init__(
        self,
        default_resolution={"web": (640, 480), "rs": (640, 480)},
        resize_resolution={"web": (0, 0), "rs": (0, 0)},
        camera_kwargs={}
                ):
        
        # # Open Cameras #
        all_cameras = gather_cameras(default_resolution=default_resolution, resize_resolution=resize_resolution)
        self.camera_dict = {cam.serial_number: cam for cam in all_cameras}
        # self.camera_dict = {}
        
        # # For zed
        # zed_cameras = gather_zed_cameras()
        # self.zed_camera_dict = {cam.serial_number: cam for cam in zed_cameras}
        # # Set Correct Parameters #
        # for cam_id in self.zed_camera_dict.keys():
        #     cam_type = get_camera_type(cam_id)
        #     curr_cam_kwargs = camera_kwargs.get(cam_type, {})
        #     self.zed_camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)
        
        # # Launch Camera #
        # self.set_trajectory_mode()
        # self.camera_dict.update(self.zed_camera_dict)

    def set_trajectory_mode(self):
        # If High Res Calibration, Close All #
        close_all = any(
            [cam.high_res_calibration and cam.current_mode == "calibration" for cam in self.zed_camera_dict.values()]
        )

        if close_all:
            for cam in self.zed_camera_dict.values():
                cam.disable_camera()

        # Put All Cameras In Trajectory Mode #
        for cam in self.zed_camera_dict.values():
            cam.set_trajectory_mode()

    def get_camera(self, camera_id):
        return self.camera_dict[camera_id]

    # ### Data Storing Functions ###
    # def start_recording(self, recording_folderpath):
    #     for cam in self.camera_dict.values():
    #         if isinstance(cam, ZedCamera):
    #             subdir = os.path.join(recording_folderpath, "SVO")
    #             ext = ".svo"
    #         else:
    #             subdir = os.path.join(recording_folderpath, "MP4")
    #             ext = ".mp4"
    #         if not os.path.isdir(subdir):
    #             os.makedirs(subdir)
    #         filepath = os.path.join(subdir, cam.serial_number + ext)
    #         cam.start_recording(filepath)

    def stop_recording(self):
        for cam in self.camera_dict.values():
            cam.stop_recording()

    ### Basic Camera Functions ###
    def read_cameras(self):
        full_obs_dict = defaultdict(dict)
        full_timestamp_dict = {}

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if hasattr(self.camera_dict[cam_id], "is_running") and not self.camera_dict[cam_id].is_running():
                continue

            data_dict, timestamp_dict = self.camera_dict[cam_id].read_camera()
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])
            full_timestamp_dict.update(timestamp_dict)

        return full_obs_dict, full_timestamp_dict
    
    def disable_cameras(self):
        for camera in self.camera_dict.values():
            camera.disable_camera()

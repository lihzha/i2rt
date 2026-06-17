from copy import deepcopy

import cv2
import numpy as np

import time
import pyrealsense2 as rs

import cv2
from collections import deque
import threading

hand_camera_id = "1"

def time_ms():
    return time.time_ns() // 1_000_000

def get_available_camera_indices(max_cameras=1):
    available_cameras = {}
    camera_idx = 0

    for camera_id in range(max_cameras):
        cap = cv2.VideoCapture(camera_id)

        if cap.isOpened():
            print(f"Camera {camera_id} is available. Showing preview...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to capture frame from camera {camera_id}. Skipping...")
                    break
                cv2.imshow(f"Camera {camera_id}", frame)
                key = cv2.waitKey(1) & 0xFF

                # Press 'y' to use the camera, 'n' to ignore, 'q' to quit
                if key == ord("y"):
                    available_cameras[camera_idx] = camera_id
                    camera_idx += 1
                    print(f"Camera {camera_id} selected as {camera_idx}.")
                    break
                elif key == ord("n"):
                    print(f"Camera {camera_id} skipped.")
                    break
                elif key == ord("q"):
                    print("Exiting camera selection.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return available_cameras  # Return early if user wants to quit

            cap.release()
            cv2.destroyAllWindows()

    # available_cameras["0"] = 8

    return available_cameras


def gather_web_cameras(default_resolution, resize_resolution):
    all_web_cameras = []
    cameras = get_available_camera_indices(1)
    for serial_number, cam_idx in cameras.items():
        cam = Camera(
            camera_idx=cam_idx,
            serial_number=serial_number,
            default_resolution=default_resolution,
            resize_resolution=resize_resolution,
        )
        all_web_cameras.append(cam)
    return all_web_cameras


def gather_cameras(default_resolution, resize_resolution):
    web_cameras = gather_web_cameras(
        default_resolution=default_resolution["web"], resize_resolution=resize_resolution["web"]
    )
    # # # return web_cameras
    # hand_camera = RSCamera(
    #     camera_idx=hand_camera_id,
    #     default_resolution=default_resolution["rs"],
    #     resize_resolution=resize_resolution["rs"],
    # )
    return web_cameras

class Camera:
    def __init__(self, camera_idx, serial_number=None, default_resolution=(640, 480), resize_resolution=(0, 0)):
        # Save Parameters
        self.serial_number = str(serial_number) if serial_number is not None else str(camera_idx)
        self.camera_idx = camera_idx
        self.is_hand_camera = self.serial_number == hand_camera_id
        self._extrinsics = {}
        self.latency = 0  # Can be estimated or calculated if required
        self.resizer_resolution = resize_resolution
        self.default_resolution = default_resolution
        self.skip_reading = False  # Example flag
        self.image = True  # Enable reading image
        self._video_writer = None
        # Open Camera
        self.launch_camera(resolution=default_resolution)

        # Queue for storing frames
        self.frame_queue = deque(maxlen=5)  # Limit queue size to avoid memory issues

        # Thread for capturing frames
        if not self.is_hand_camera:
            self.running = threading.Event()
            self.running.set()
            self.capture_thread = threading.Thread(target=self._update_frame, daemon=True)
            self.capture_thread.start()

    ### Intrinsics Method ###
    def get_intrinsics(self):
        """Return basic camera intrinsics, width, height, and fps."""
        return deepcopy(self._intrinsics)

    ### Basic Camera Utilities ###
    def _process_frame(self, frame):
        """Resize the frame if needed."""
        if self.resizer_resolution == (0, 0):  # If no resizing needed
            return frame
        # Resize the frame to the set resolution
        return cv2.resize(frame, self.resizer_resolution)

    def _update_frame(self):
        """Continuously capture frames and store recent ones in a deque."""
        while self.running.is_set():
            ret, frame = self._cam.read()
            if not ret:
                print("Warning: Failed to capture frame.")
                continue

            frame = self._process_frame(frame)  # Resize if needed
            self.frame_queue.append(frame)  # Automatically removes oldest if full

    ### Read Camera Frame ###
    def read_camera(self):
        """Read the camera frame, capture timestamps, and return data."""
        # Skip if Read Unnecessary
        # if self.skip_reading:
        #     return {}, {}
        if self.skip_reading or not self.frame_queue:
            return {}, {}

        # Read Camera Frame
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        # ret, frame = self._cam.read()
        # Get latest frame from the queue
        frame = self.frame_queue[-1]
        ret = True
        if self._video_writer is not None:
            self.add_frame(ret, frame)

        if not ret:  # Failed to grab frame
            print("Error: Could not read the frame.")
            return None

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency (Using time since webcam doesn't provide timestamps)
        received_time = timestamp_dict[self.serial_number + "_read_end"]
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data
        data_dict = {}

        if self.image and frame is not None:
            # Process the frame if needed (resize)
            # processed_frame = self._process_frame(frame)
            processed_frame = frame
            data_dict["image"] = {self.serial_number: processed_frame}


        # print('stopping in read camera function from web_camera.py')
        #breakpoint()
        return data_dict, timestamp_dict

    ### Release Camera ###
    def release(self):
        """Stop the camera thread and release the camera."""
        self.running.clear()
        self.capture_thread.join()
        self._cam.release()

    def disable_camera(self):
        if hasattr(self, "_cam"):
            self.release()

    ### Recording Utilities ###
    def start_recording(self, filename):
        assert filename.endswith(".mp4") or filename.endswith(".avi"), "File must be .mp4 or .avi"

        # Choose codec based on file extension
        if filename.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Appropriate codec for mp4
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for avi

        frame_width = int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = 30.0  # Frame rate (you can adjust it if needed)

        # Initialize video writer
        self._video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        print("Starting Recording")

    def stop_recording(self):
        if hasattr(self, "_video_writer"):
            print("Stopping Recording")
            self._video_writer.release()
            self._video_writer = None

    def add_frame(self, ret, frame):
        """Capture a frame from the camera and add it to the video."""
        if self._video_writer is None:
            raise RuntimeError("Video recording has not started yet.")

        if ret:
            self._video_writer.write(frame)  # Add the captured frame to the video
        else:
            print("Failed to capture frame from camera.")

    def launch_camera(self, resolution=(640, 480)):

        # Close Existing Camera
        self.disable_camera()

        # Initialize Camera using OpenCV
        self._cam = cv2.VideoCapture(int(self.camera_idx))

        print(f"Cam Idx: {self.camera_idx}")

        if not self._cam.isOpened():
            raise RuntimeError("Camera Failed To Open")

        # Set Width and Height  # TODO: use argument
        self._cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        self._cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        self._cam.set(cv2.CAP_PROP_FPS, 60)

        # Save Intrinsics with updated width and height
        self.latency = int(2.5 * (1e3 / self._cam.get(cv2.CAP_PROP_FPS)))  # Estimate latency

        # Get and Save Intrinsics (width, height, fps)
        self._intrinsics = {
            "width": int(self._cam.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cam.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": int(self._cam.get(cv2.CAP_PROP_FPS)),
        }
        print("Opening Camera: ", self.serial_number)
        print("Camera Intrinsics:", self._intrinsics)


class RSCamera(Camera):
    def __init__(self, camera_idx, default_resolution=(640, 480), resize_resolution=(0, 0)):
        super().__init__(
            camera_idx=camera_idx, default_resolution=default_resolution, resize_resolution=resize_resolution
        )

    ### Read Camera Frame ###
    def read_camera(self):
        """Read the camera frame, capture timestamps, and return data."""
        # Skip if Read Unnecessary
        if self.skip_reading:
            return {}, {}

        # Read Camera Frame
        timestamp_dict = {self.serial_number + "_read_start": time_ms()}
        frame = self.pipeline.wait_for_frames()
        color_frame = frame.get_color_frame()
        if not color_frame:
            ret = False
        else:
            ret = True
            color_frame = np.asanyarray(color_frame.get_data())
        if self._video_writer is not None:
            self.add_frame(ret, color_frame)

        if not ret:  # Failed to grab frame
            print("Error: Could not read the frame.")
            return None

        timestamp_dict[self.serial_number + "_read_end"] = time_ms()

        # Benchmark Latency (Using time since webcam doesn't provide timestamps)
        received_time = timestamp_dict[self.serial_number + "_read_end"]
        timestamp_dict[self.serial_number + "_frame_received"] = received_time
        timestamp_dict[self.serial_number + "_estimated_capture"] = received_time - self.latency

        # Return Data
        data_dict = {}

        if self.image and color_frame is not None:
            # Process the frame if needed (resize)
            processed_frame = self._process_frame(color_frame)
            data_dict["image"] = {self.serial_number: processed_frame}

        return data_dict, timestamp_dict


    ### Release Camera ###
    def release(self):
        """Release the camera resource."""
        self.pipeline.stop()

    def disable_camera(self):
        if hasattr(self, "pipeline"):
            self.release()

    ### Recording Utilities ###
    def start_recording(self, filename):
        assert filename.endswith(".mp4") or filename.endswith(".avi"), "File must be .mp4 or .avi"

        # Choose codec based on file extension
        if filename.endswith(".mp4"):
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Appropriate codec for mp4
        else:
            fourcc = cv2.VideoWriter_fourcc(*"XVID")  # Codec for avi

        frames = self.pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()

        frame_width = color_frame.get_width()
        frame_height = color_frame.get_height()
        fps = 30.0  # Frame rate (you can adjust it if needed)

        # Initialize video writer
        self._video_writer = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

        print("Starting Recording")

    def launch_camera(self, resolution=(640, 480)):
        # Close Existing Camera
        self.disable_camera()

        # Initialize Camera using OpenCV
        self.pipeline = rs.pipeline()
        config = rs.config()

        # TODO: use argument
        # supported: 1280 x 720, 424 x 240, 480 x 270, 640 x 360, 640 x 480, 848 x 480
        config.enable_stream(rs.stream.color, resolution[0], resolution[1], rs.format.bgr8, 30)

        profile = self.pipeline.start(config)
        time.sleep(2)  # Allow the camera to stabilize

        # Check if the camera is opened (RealSense doesn't use .isOpened() like OpenCV)
        device = profile.get_device()
        if not device:
            raise RuntimeError("Camera Failed to Open")

        # Save latency (estimate based on RealSense FPS)
        # self.color_sensor = profile.get_device().first_color_sensor()
        # self.fps = self.color_sensor.get_option(rs.option.framerate)
        self.fps = 30
        self.latency = int(2.5 * (1e3 / self.fps))  # Estimate latency

        # Get and save intrinsics (width, height, fps)
        color_stream = profile.get_stream(rs.stream.color)
        intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
        self._intrinsics = {
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "ppx": intrinsics.ppx,
            "ppy": intrinsics.ppy,
            "coeffs": intrinsics.coeffs,
            "fps": int(self.fps),
        }

        print("Opening Camera: ", self.serial_number)
        print("Color Camera Intrinsics:", self._intrinsics)
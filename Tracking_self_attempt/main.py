import cv2
import numpy as np
from cv2 import aruco
import pickle
import time


class ArUcoTracker:
    """
    Track camera pose using ArUco markers.
    """
    def estimate_pose_from_corners(self, marker_corners):
        """
        Estimate pose using solvePnP instead of estimatePoseSingleMarkers.
        marker_corners: (4, 2) array in image pixels.
        Returns rvec (3x1), tvec (3x1).
        """
        # Define marker 3D points in marker coordinate system (square centered at origin)
        half = self.marker_size / 2.0
        obj_points = np.array([
            [-half,  half, 0],   # top-left
            [ half,  half, 0],   # top-right
            [ half, -half, 0],   # bottom-right
            [-half, -half, 0],   # bottom-left
        ], dtype=np.float32)

        # marker_corners is (4, 1, 2) or (4, 2); reshape to (4, 2)
        img_points = marker_corners.reshape(4, 2).astype(np.float32)

        success, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_IPPE_SQUARE
        )
        if not success:
            return None, None

        return rvec, tvec

    def __init__(self, camera_source, calibration_file='camera_calibration.pkl', marker_size=0.15):
        """
        Args:
            camera_source: Camera index (int) or URL (string).
            calibration_file: Path to calibration file (pickle with 'camera_matrix' and 'dist_coeffs').
            marker_size: Physical size of marker in meters.
        """
        self.camera_source = camera_source
        self.marker_size = float(marker_size)
        self.cap = None

        # Tracking data
        self.trajectory = []
        self.pose_history = []

        # Load calibration data
        self.load_calibration(calibration_file)

        # ArUco dictionary and detector
        self.aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)
        params = aruco.DetectorParameters()
        self.detector = aruco.ArucoDetector(self.aruco_dict, params)

    def load_calibration(self, filename):
        """
        Load camera calibration data.
        Expected keys: 'camera_matrix', 'dist_coeffs'.
        """
        try:
            with open(filename, 'rb') as f:
                calib_data = pickle.load(f)

            self.camera_matrix = np.array(calib_data['camera_matrix'], dtype=float)
            self.dist_coeffs = np.array(calib_data['dist_coeffs'], dtype=float)
            print(f"✓ Loaded calibration from: {filename}")

        except (FileNotFoundError, KeyError, EOFError, OSError) as e:
            print(f"Warning: Could not load calibration file '{filename}': {e}")
            print("Using default camera parameters (less accurate).")
            # Rough default parameters
            self.camera_matrix = np.array([[800, 0, 320],
                                           [0, 800, 240],
                                           [0,   0,   1]], dtype=float)
            self.dist_coeffs = np.zeros((5, 1), dtype=float)

    def rotation_matrix_to_euler_angles(self, R):
        """
        Convert rotation matrix to Euler angles (roll, pitch, yaw) in degrees.
        """
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

        singular = sy < 1e-6

        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])      # roll
            y = np.arctan2(-R[2, 0], sy)          # pitch
            z = np.arctan2(R[1, 0], R[0, 0])      # yaw
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0.0

        return np.array([x, y, z]) * 180.0 / np.pi

    def draw_axis(self, frame, rvec, tvec, length=0.1):
        """
        Draw 3D coordinate axes on the frame.
        rvec, tvec must be 3x1 ndarray.
        """
        axis_points = np.float32([
            [0, 0, 0],          # Origin
            [length, 0, 0],     # X-axis (red)
            [0, length, 0],     # Y-axis (green)
            [0, 0, -length]     # Z-axis (blue)
        ])

        imgpts, _ = cv2.projectPoints(axis_points, rvec, tvec,
                                      self.camera_matrix, self.dist_coeffs)
        imgpts = imgpts.reshape(-1, 2).astype(int)

        corner = tuple(imgpts[0])
        frame = cv2.line(frame, corner, tuple(imgpts[1]), (0, 0, 255), 3)   # X: Red
        frame = cv2.line(frame, corner, tuple(imgpts[2]), (0, 255, 0), 3)   # Y: Green
        frame = cv2.line(frame, corner, tuple(imgpts[3]), (255, 0, 0), 3)   # Z: Blue

        return frame

    def draw_pose_info(self, frame, tvec, euler_angles, marker_id):
        """
        Draw pose information on frame.
        tvec: 3x1 translation vector (meters).
        euler_angles: [roll, pitch, yaw] in degrees.
        marker_id: integer ID.
        """
        y_offset = 30

        cv2.putText(frame, f"Marker ID: {marker_id}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 30

        # Translation
        cv2.putText(frame, "Position (m):", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"  X: {tvec[0, 0]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"  Y: {tvec[1, 0]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"  Z: {tvec[2, 0]:.3f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 30

        # Rotation
        cv2.putText(frame, "Rotation (deg):", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        y_offset += 25
        cv2.putText(frame, f"  Roll:  {euler_angles[0]:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"  Pitch: {euler_angles[1]:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        cv2.putText(frame, f"  Yaw:   {euler_angles[2]:.1f}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        distance = float(np.linalg.norm(tvec))
        cv2.putText(frame, f"Distance: {distance:.3f} m",
                    (10, frame.shape[0] - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        return frame

    def track(self, show_video=True, save_trajectory=False):
        """
        Start tracking camera pose.
        """
        self.cap = cv2.VideoCapture(self.camera_source)

        if not self.cap.isOpened():
            print("Error: Cannot open camera.")
            return

        print("\n" + "=" * 50)
        print("ArUco Marker Tracking Started")
        print("=" * 50)
        print(f"Marker size: {self.marker_size} m")
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save current frame")
        print("  'r' - Reset trajectory")
        print("  'p' - Print current pose")
        print("-" * 50)

        frame_count = 0
        fps_start_time = time.time()
        fps = 0.0

        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Can't receive frame.")
                    break

                frame_count += 1

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # Detect ArUco markers
                corners, ids, rejected = self.detector.detectMarkers(gray)

                if ids is not None and len(ids) > 0:
                    aruco.drawDetectedMarkers(frame, corners, ids)

                    # estimatePoseSingleMarkers returns (rvecs, tvecs, _objPoints)
                    for i in range(len(ids)):
                        marker_corners = corners[i]   # (1, 4, 2)
                        rvec, tvec = self.estimate_pose_from_corners(marker_corners)
                        if rvec is None:
                            continue

                        rvec = rvec.reshape(3, 1)
                        tvec = tvec.reshape(3, 1)
                        marker_id = int(ids[i][0])


                        frame = self.draw_axis(frame, rvec, tvec, self.marker_size / 2.0)

                        R, _ = cv2.Rodrigues(rvec)
                        euler_angles = self.rotation_matrix_to_euler_angles(R)

                        frame = self.draw_pose_info(frame, tvec, euler_angles, marker_id)

                        pose_data = {
                            'frame': frame_count,
                            'marker_id': marker_id,
                            'tvec': tvec.copy(),
                            'rvec': rvec.copy(),
                            'euler': euler_angles.copy(),
                            'timestamp': time.time()
                        }
                        self.pose_history.append(pose_data)

                        if save_trajectory:
                            self.trajectory.append(tvec.flatten().copy())

                    status_color = (0, 255, 0)
                    status_text = f"Tracking {len(ids)} marker(s)"
                else:
                    status_color = (0, 0, 255)
                    status_text = "No markers detected"

                # FPS
                if frame_count % 30 == 0:
                    now = time.time()
                    fps = 30.0 / (now - fps_start_time)
                    fps_start_time = now

                cv2.putText(frame, status_text,
                            (10, frame.shape[0] - 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            status_color, 2)

                cv2.putText(frame, f"FPS: {fps:.1f}",
                            (10, frame.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (255, 255, 0), 2)

                if show_video:
                    cv2.imshow('ArUco Tracking', frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    filename = f'tracking_frame_{frame_count}.jpg'
                    cv2.imwrite(filename, frame)
                    print(f"Frame saved as {filename}")
                elif key == ord('r'):
                    self.trajectory = []
                    self.pose_history = []
                    print("Trajectory reset.")
                elif key == ord('p'):
                    if self.pose_history:
                        latest_pose = self.pose_history[-1]
                        print(f"\nCurrent Pose (Frame {latest_pose['frame']}):")
                        print(f"  Marker ID: {latest_pose['marker_id']}")
                        print(f"  Position: {latest_pose['tvec'].flatten()}")
                        print(f"  Rotation (roll, pitch, yaw): {latest_pose['euler']}")

        finally:
            if self.cap is not None:
                self.cap.release()
            cv2.destroyAllWindows()

            print("\n✓ Tracking stopped")
            print(f"Total frames: {frame_count}")
            print(f"Poses recorded: {len(self.pose_history)}")

            if save_trajectory and len(self.trajectory) > 0:
                self.save_trajectory_data()

    def save_trajectory_data(self, filename='trajectory.npy'):
        """
        Save trajectory data and pose history.
        """
        trajectory_array = np.array(self.trajectory)
        np.save(filename, trajectory_array)
        print(f"✓ Trajectory saved to: {filename}")

        pose_filename = 'pose_history.pkl'
        with open(pose_filename, 'wb') as f:
            pickle.dump(self.pose_history, f)
        print(f"✓ Pose history saved to: {pose_filename}")


def main():
    """
    Main function.
    """
    print("=" * 50)
    print("ArUco Marker Tracking")
    print("=" * 50)

    print("\nCamera source:")
    print("1. Camera index (e.g., 0, 1, 2)")
    print("2. IP camera URL")
    choice = input("Enter choice (1 or 2): ").strip()

    if choice == "1":
        cam_str = input("Enter camera index [0]: ").strip()
        camera_source = int(cam_str) if cam_str else 0
    else:
        camera_source = input("Enter IP camera URL: ").strip()

    marker_str = input("\nMarker size in meters [0.15]: ").strip()
    marker_size = float(marker_str) if marker_str else 0.15

    calib_file = input("Calibration file [camera_calibration.pkl]: ").strip()
    if not calib_file:
        calib_file = "camera_calibration.pkl"

    save_traj_input = input("Save trajectory data? (y/n) [n]: ").strip().lower()
    save_traj = (save_traj_input == 'y')

    tracker = ArUcoTracker(camera_source, calib_file, marker_size)
    tracker.track(show_video=True, save_trajectory=save_traj)


if __name__ == "__main__":
    main()

"""
Module 7: Stereo Calibration, Pose Estimation, and Hand Tracking
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
import mediapipe as mp
import csv
from datetime import datetime

class StereoCalibration:
    """Stereo camera calibration and depth estimation"""
    
    def __init__(self):
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        self.baseline = None
    
    def calibrate_stereo(self, left_images, right_images, pattern_size=(9, 6), square_size=1.0):
        """
        Calibrate stereo camera pair
        
        Args:
            left_images: List of calibration images from left camera
            right_images: List of calibration images from right camera
            pattern_size: Chessboard pattern size (columns, rows)
            square_size: Size of chessboard square in real-world units
        
        Returns:
            Calibration success status
        """
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        objp *= square_size
        
        objpoints = []  # 3D points in real world
        imgpoints_left = []  # 2D points in left image
        imgpoints_right = []  # 2D points in right image
        
        # Find chessboard corners
        for left_img, right_img in zip(left_images, right_images):
            gray_left = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
            
            ret_left, corners_left = cv2.findChessboardCorners(gray_left, pattern_size)
            ret_right, corners_right = cv2.findChessboardCorners(gray_right, pattern_size)
            
            if ret_left and ret_right:
                objpoints.append(objp)
                
                # Refine corners
                corners_left = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1),
                                                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                corners_right = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1),
                                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                imgpoints_left.append(corners_left)
                imgpoints_right.append(corners_right)
        
        if len(objpoints) == 0:
            return False
        
        # Calibrate individual cameras
        img_shape = left_images[0].shape[:2][::-1]
        
        ret_left, self.camera_matrix_left, self.dist_coeffs_left, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_left, img_shape, None, None
        )
        
        ret_right, self.camera_matrix_right, self.dist_coeffs_right, _, _ = cv2.calibrateCamera(
            objpoints, imgpoints_right, img_shape, None, None
        )
        
        # Stereo calibration
        flags = cv2.CALIB_FIX_INTRINSIC
        
        ret, _, _, _, _, self.R, self.T, self.E, self.F = cv2.stereoCalibrate(
            objpoints, imgpoints_left, imgpoints_right,
            self.camera_matrix_left, self.dist_coeffs_left,
            self.camera_matrix_right, self.dist_coeffs_right,
            img_shape, flags=flags
        )
        
        # Calculate baseline (distance between cameras)
        self.baseline = np.linalg.norm(self.T)
        
        return ret
    
    def compute_disparity(self, left_image, right_image):
        """Compute disparity map from stereo pair"""
        # Convert to grayscale
        gray_left = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)
        
        # Create stereo matcher
        stereo = cv2.StereoBM_create(numDisparities=16*5, blockSize=15)
        
        # Compute disparity
        disparity = stereo.compute(gray_left, gray_right)
        
        # Normalize for visualization
        disparity_vis = cv2.normalize(disparity, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        
        return disparity, disparity_vis
    
    def estimate_depth(self, disparity, focal_length=None):
        """
        Estimate depth from disparity
        
        Depth = (Focal Length * Baseline) / Disparity
        """
        if focal_length is None and self.camera_matrix_left is not None:
            focal_length = self.camera_matrix_left[0, 0]
        
        if focal_length is None or self.baseline is None:
            return None
        
        # Avoid division by zero
        depth = np.zeros_like(disparity, dtype=np.float32)
        valid_disparity = disparity > 0
        depth[valid_disparity] = (focal_length * self.baseline) / disparity[valid_disparity]
        
        return depth
    
    def measure_object_size(self, left_image, right_image, bbox):
        """
        Measure object size using stereo reconstruction
        
        Args:
            left_image: Left stereo image
            right_image: Right stereo image
            bbox: Bounding box (x, y, w, h) of object in left image
        
        Returns:
            Object dimensions (width, height, depth)
        """
        # Compute disparity
        disparity, _ = self.compute_disparity(left_image, right_image)
        
        # Get depth map
        depth_map = self.estimate_depth(disparity)
        
        if depth_map is None:
            return None
        
        x, y, w, h = bbox
        
        # Get average depth in bbox
        roi_depth = depth_map[y:y+h, x:x+w]
        avg_depth = np.median(roi_depth[roi_depth > 0])
        
        if np.isnan(avg_depth) or avg_depth <= 0:
            return None
        
        # Get focal length
        focal_length = self.camera_matrix_left[0, 0] if self.camera_matrix_left is not None else 700
        
        # Calculate real-world dimensions
        real_width = (w * avg_depth) / focal_length
        real_height = (h * avg_depth) / focal_length
        
        return {
            'width': real_width,
            'height': real_height,
            'depth': avg_depth,
            'bbox': bbox
        }


class PoseEstimation:
    """Real-time pose estimation using Mediapipe"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.csv_data = []
        self.frame_count = 0
    
    def process_frame(self, frame):
        """
        Process frame for pose estimation
        
        Returns:
            (annotated_frame, pose_landmarks, pose_data)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.pose.process(rgb_frame)
        
        # Draw landmarks
        annotated_frame = frame.copy()
        
        pose_data = None
        if results.pose_landmarks:
            # Draw pose landmarks
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Extract pose data
            pose_data = self.extract_pose_data(results.pose_landmarks, self.frame_count)
            self.csv_data.append(pose_data)
            self.frame_count += 1
        
        return annotated_frame, results.pose_landmarks, pose_data
    
    def extract_pose_data(self, landmarks, frame_num):
        """Extract pose data for logging"""
        data = {'frame': frame_num}
        
        # Extract all landmarks
        for idx, landmark in enumerate(landmarks.landmark):
            landmark_name = self.mp_pose.PoseLandmark(idx).name
            data[f'{landmark_name}_x'] = landmark.x
            data[f'{landmark_name}_y'] = landmark.y
            data[f'{landmark_name}_z'] = landmark.z
            data[f'{landmark_name}_visibility'] = landmark.visibility
        
        return data
    
    def save_to_csv(self, filename='pose_data.csv'):
        """Save collected pose data to CSV"""
        if not self.csv_data:
            return False
        
        keys = self.csv_data[0].keys()
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.csv_data)
        
        return True
    
    def reset_data(self):
        """Reset collected data"""
        self.csv_data = []
        self.frame_count = 0


class HandTracking:
    """Real-time hand tracking using Mediapipe"""
    
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            max_num_hands=2
        )
        self.csv_data = []
        self.frame_count = 0
    
    def process_frame(self, frame):
        """
        Process frame for hand tracking
        
        Returns:
            (annotated_frame, hand_landmarks_list, hand_data)
        """
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process
        results = self.hands.process(rgb_frame)
        
        # Draw landmarks
        annotated_frame = frame.copy()
        
        hand_data = None
        if results.multi_hand_landmarks:
            hand_data = {'frame': self.frame_count, 'num_hands': len(results.multi_hand_landmarks)}
            
            for hand_idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                # Draw hand landmarks
                self.mp_drawing.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
                # Add handedness info
                if results.multi_handedness:
                    handedness = results.multi_handedness[hand_idx].classification[0].label
                    hand_data[f'hand_{hand_idx}_type'] = handedness
                
                # Extract landmarks
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    landmark_name = self.mp_hands.HandLandmark(idx).name
                    hand_data[f'hand_{hand_idx}_{landmark_name}_x'] = landmark.x
                    hand_data[f'hand_{hand_idx}_{landmark_name}_y'] = landmark.y
                    hand_data[f'hand_{hand_idx}_{landmark_name}_z'] = landmark.z
            
            self.csv_data.append(hand_data)
            self.frame_count += 1
        
        return annotated_frame, results.multi_hand_landmarks, hand_data
    
    def detect_gestures(self, hand_landmarks):
        """Detect simple hand gestures"""
        if not hand_landmarks:
            return "No hand detected"
        
        # Get landmarks
        landmarks = hand_landmarks.landmark
        
        # Thumb tip and thumb ip
        thumb_tip = landmarks[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = landmarks[self.mp_hands.HandLandmark.THUMB_IP]
        
        # Index finger tip and pip
        index_tip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = landmarks[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        
        # Middle finger tip and pip
        middle_tip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = landmarks[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        
        # Ring finger tip and pip
        ring_tip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = landmarks[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        
        # Pinky tip and pip
        pinky_tip = landmarks[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = landmarks[self.mp_hands.HandLandmark.PINKY_PIP]
        
        # Detect if fingers are extended
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        ring_extended = ring_tip.y < ring_pip.y
        pinky_extended = pinky_tip.y < pinky_pip.y
        
        # Count extended fingers
        extended_count = sum([index_extended, middle_extended, ring_extended, pinky_extended])
        
        if extended_count == 0:
            return "Fist"
        elif extended_count == 1 and index_extended:
            return "Pointing"
        elif extended_count == 2 and index_extended and middle_extended:
            return "Peace Sign"
        elif extended_count == 4:
            return "Open Hand"
        else:
            return f"{extended_count} fingers"
    
    def save_to_csv(self, filename='hand_data.csv'):
        """Save collected hand data to CSV"""
        if not self.csv_data:
            return False
        
        # Get all possible keys
        all_keys = set()
        for data in self.csv_data:
            all_keys.update(data.keys())
        
        keys = sorted(all_keys)
        
        with open(filename, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(self.csv_data)
        
        return True
    
    def reset_data(self):
        """Reset collected data"""
        self.csv_data = []
        self.frame_count = 0





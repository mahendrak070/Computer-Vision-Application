"""
Module 5-6: Real-Time Object Tracking
Marker-based, Marker-less, and SAM2-based tracking
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np

class MarkerBasedTracker:
    """Tracking using ArUco/QR/AprilTag markers"""
    
    def __init__(self, marker_type='aruco'):
        self.marker_type = marker_type
        
        if marker_type == 'aruco':
            self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
            self.aruco_params = cv2.aruco.DetectorParameters()
            self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        elif marker_type == 'qr':
            self.qr_detector = cv2.QRCodeDetector()
    
    def track(self, frame):
        """
        Track markers in frame
        
        Returns:
            (annotated_frame, tracking_data)
        """
        result_frame = frame.copy()
        tracking_data = []
        
        if self.marker_type == 'aruco':
            corners, ids, rejected = self.detector.detectMarkers(frame)
            
            if ids is not None:
                cv2.aruco.drawDetectedMarkers(result_frame, corners, ids)
                
                for i, corner in enumerate(corners):
                    # Calculate center
                    center = corner[0].mean(axis=0).astype(int)
                    marker_id = ids[i][0]
                    
                    # Draw center
                    cv2.circle(result_frame, tuple(center), 5, (0, 255, 0), -1)
                    
                    # Add text
                    cv2.putText(result_frame, f"ID: {marker_id}", 
                               (center[0] - 30, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    tracking_data.append({
                        'id': int(marker_id),
                        'center': center.tolist(),
                        'corners': corner[0].tolist(),
                        'type': 'aruco'
                    })
        
        elif self.marker_type == 'qr':
            data, points, _ = self.qr_detector.detectAndDecode(frame)
            
            if points is not None:
                points = points[0].astype(int)
                
                # Draw bounding box
                for i in range(4):
                    cv2.line(result_frame, tuple(points[i]), 
                            tuple(points[(i+1)%4]), (0, 255, 0), 3)
                
                # Calculate center
                center = points.mean(axis=0).astype(int)
                cv2.circle(result_frame, tuple(center), 5, (0, 255, 0), -1)
                
                # Add text
                if data:
                    cv2.putText(result_frame, f"QR: {data[:20]}", 
                               (center[0] - 50, center[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                tracking_data.append({
                    'data': data,
                    'center': center.tolist(),
                    'corners': points.tolist(),
                    'type': 'qr'
                })
        
        return result_frame, tracking_data


class MarkerlessTracker:
    """Classical CV-based marker-less tracking"""
    
    def __init__(self):
        self.tracker = None
        self.tracking_initialized = False
        self.bbox = None
        
        # Feature detector for initialization
        self.sift = cv2.SIFT_create()
        self.bf_matcher = cv2.BFMatcher()
        
        # Template for tracking
        self.template = None
        self.template_keypoints = None
        self.template_descriptors = None
    
    def initialize_tracking(self, frame, bbox):
        """
        Initialize tracking with bounding box
        
        Args:
            frame: Initial frame
            bbox: (x, y, w, h) bounding box
        """
        self.bbox = bbox
        x, y, w, h = bbox
        
        # Extract template
        self.template = frame[y:y+h, x:x+w].copy()
        
        # Detect features in template
        self.template_keypoints, self.template_descriptors = self.sift.detectAndCompute(
            self.template, None
        )
        
        # Initialize OpenCV tracker (for comparison)
        self.tracker = cv2.TrackerCSRT_create()
        self.tracker.init(frame, tuple(bbox))
        
        self.tracking_initialized = True
    
    def track(self, frame):
        """Track object in frame"""
        if not self.tracking_initialized:
            return frame, None
        
        result_frame = frame.copy()
        
        # Method 1: Template matching
        template_bbox = self.track_template_matching(frame)
        
        # Method 2: Feature matching
        feature_bbox = self.track_feature_matching(frame)
        
        # Method 3: OpenCV tracker
        success, opencv_bbox = self.tracker.update(frame)
        
        tracking_data = {
            'template_matching': template_bbox,
            'feature_matching': feature_bbox,
            'opencv_tracker': list(opencv_bbox) if success else None
        }
        
        # Draw results
        if template_bbox:
            x, y, w, h = template_bbox
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(result_frame, "Template", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        if feature_bbox:
            x, y, w, h = feature_bbox
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(result_frame, "Features", (x, y-30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        if success:
            x, y, w, h = [int(v) for v in opencv_bbox]
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.putText(result_frame, "CSRT", (x, y-50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        return result_frame, tracking_data
    
    def track_template_matching(self, frame):
        """Track using template matching"""
        if self.template is None:
            return None
        
        # Convert to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        template_gray = cv2.cvtColor(self.template, cv2.COLOR_BGR2GRAY) if len(self.template.shape) == 3 else self.template
        
        # Match template
        result = cv2.matchTemplate(frame_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.6:  # Confidence threshold
            h, w = self.template.shape[:2]
            return [max_loc[0], max_loc[1], w, h]
        
        return None
    
    def track_feature_matching(self, frame):
        """Track using feature matching"""
        if self.template_descriptors is None or len(self.template_descriptors) == 0:
            return None
        
        # Detect features in current frame
        frame_keypoints, frame_descriptors = self.sift.detectAndCompute(frame, None)
        
        if frame_descriptors is None or len(frame_descriptors) < 4:
            return None
        
        # Match features
        matches = self.bf_matcher.knnMatch(self.template_descriptors, frame_descriptors, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 4:
            return None
        
        # Get matched points
        src_pts = np.float32([self.template_keypoints[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([frame_keypoints[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Find homography
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        
        if H is None:
            return None
        
        # Transform template corners
        h, w = self.template.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        transformed_corners = cv2.perspectiveTransform(corners, H)
        
        # Get bounding box
        x_min = int(transformed_corners[:, 0, 0].min())
        y_min = int(transformed_corners[:, 0, 1].min())
        x_max = int(transformed_corners[:, 0, 0].max())
        y_max = int(transformed_corners[:, 0, 1].max())
        
        return [x_min, y_min, x_max - x_min, y_max - y_min]


class ColorBasedTracker:
    """Simple color-based tracking"""
    
    def __init__(self):
        self.target_color_range = None
    
    def initialize_tracking(self, frame, bbox):
        """Initialize by extracting color range from bbox"""
        x, y, w, h = bbox
        roi = frame[y:y+h, x:x+w]
        
        # Convert to HSV
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Calculate color range
        mean_color = hsv_roi.mean(axis=(0, 1))
        std_color = hsv_roi.std(axis=(0, 1))
        
        lower = np.array([max(0, mean_color[0] - 2*std_color[0]), 50, 50])
        upper = np.array([min(180, mean_color[0] + 2*std_color[0]), 255, 255])
        
        self.target_color_range = (lower, upper)
    
    def track(self, frame):
        """Track by color"""
        if self.target_color_range is None:
            return frame, None
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask
        lower, upper = self.target_color_range
        mask = cv2.inRange(hsv, lower, upper)
        
        # Morphological operations
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        result_frame = frame.copy()
        
        if contours:
            # Get largest contour
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Draw rectangle
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Calculate center
            center = (x + w//2, y + h//2)
            cv2.circle(result_frame, center, 5, (0, 0, 255), -1)
            
            return result_frame, {'bbox': [x, y, w, h], 'center': center}
        
        return result_frame, None


class SAM2Tracker:
    """
    SAM2-based tracking using precomputed NPZ segmentation
    Note: This is a placeholder for SAM2 integration
    In practice, you would load SAM2 masks from NPZ files
    """
    
    def __init__(self):
        self.masks = []
        self.current_frame = 0
    
    def load_segmentation(self, npz_path):
        """Load precomputed SAM2 segmentation masks"""
        try:
            data = np.load(npz_path)
            self.masks = data['masks']
            return True
        except:
            return False
    
    def track(self, frame):
        """Track using precomputed masks"""
        if self.current_frame >= len(self.masks):
            return frame, None
        
        result_frame = frame.copy()
        mask = self.masks[self.current_frame]
        
        # Apply mask
        colored_mask = np.zeros_like(frame)
        colored_mask[mask > 0] = [0, 255, 0]
        
        # Blend
        result_frame = cv2.addWeighted(result_frame, 0.7, colored_mask, 0.3, 0)
        
        # Find bounding box
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), 
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tracking_data = None
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            tracking_data = {
                'bbox': [x, y, w, h],
                'mask_area': cv2.contourArea(largest_contour),
                'frame': self.current_frame
            }
        
        self.current_frame += 1
        return result_frame, tracking_data
    
    def reset(self):
        """Reset to first frame"""
        self.current_frame = 0





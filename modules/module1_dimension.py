"""
Module 1: Real-World Dimension Estimation
Uses camera calibration and perspective projection for accurate measurements

Perspective Projection Formula:
    W_real = (W_pixel × Distance) / Focal_Length

Where:
    - W_real: Real-world dimension of the object
    - W_pixel: Pixel dimension measured in the image
    - Distance: Known distance from camera to object
    - Focal_Length: Focal length from camera calibration matrix

Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
import math

class DimensionEstimator:
    def __init__(self):
        # Camera intrinsic matrix from calibration
        # Format: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        self.MTX = np.array([
            [4.05312518e+03, 0.00000000e+00, 2.14541386e+03],
            [0.00000000e+00, 4.05154909e+03, 2.86292228e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        
        # Distortion coefficients
        self.DIST = np.array(
            [[0.11640574, -0.58255541, 0.007427, 0.01609734, 0.87916501]]
        )
        
        # Extract focal lengths from camera matrix
        self.fx = self.MTX[0, 0]  # Focal length in x (pixels)
        self.fy = self.MTX[1, 1]  # Focal length in y (pixels)
        
        # Use average focal length for calculations
        self.FOCAL_LENGTH_PIXELS = (self.fx + self.fy) / 2
        
        # Known distance from camera to object (in inches)
        self.CAMERA_DISTANCE_INCHES = 5.0
    
    def undistort_image(self, image):
        """
        Undistort image using camera calibration data.
        Removes lens distortion for more accurate measurements.
        """
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.MTX, self.DIST, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.MTX, self.DIST, None, newcameramtx)
        return undistorted
    
    def pixel_to_real_world(self, pixel_distance, distance_inches=None):
        """
        Convert pixel distance to real-world dimension using perspective projection.
        
        Formula: W_real = (W_pixel × D) / f
        
        Args:
            pixel_distance: Distance in pixels
            distance_inches: Distance from camera to object (default: 5 inches)
        
        Returns:
            Real-world dimension in inches
        """
        if distance_inches is None:
            distance_inches = self.CAMERA_DISTANCE_INCHES
        
        if self.FOCAL_LENGTH_PIXELS == 0:
            return 0
        
        # Perspective projection formula
        real_distance_inches = (pixel_distance * distance_inches) / self.FOCAL_LENGTH_PIXELS
        
        return real_distance_inches
    
    def estimate_from_points(self, points, image_shape, distance_inches=None, scale_factor=1.0):
        """
        Estimate real-world dimensions from clicked points using perspective projection.
        
        The perspective projection equation relates pixel measurements to real-world 
        dimensions when the camera parameters and object distance are known:
        
            W_real = (W_pixel × D) / f
        
        Args:
            points: List of (x, y) tuples representing clicked points (up to 4)
            image_shape: (height, width) of image for scaling
            distance_inches: Distance from camera to object in inches
            scale_factor: Scale factor if image was resized for display
        
        Returns:
            dict with measurements and methodology explanation
        """
        if len(points) < 2:
            return {"error": "Need at least 2 points"}
        
        if distance_inches is None:
            distance_inches = self.CAMERA_DISTANCE_INCHES
        
        results = {
            "method": "Perspective Projection",
            "formula": "W_real = (W_pixel × Distance) / Focal_Length",
            "camera_params": {
                "focal_length_px": round(self.FOCAL_LENGTH_PIXELS, 2),
                "fx": round(self.fx, 2),
                "fy": round(self.fy, 2),
                "cx": round(self.MTX[0, 2], 2),
                "cy": round(self.MTX[1, 2], 2)
            },
            "distance_inches": distance_inches,
            "distance_mm": round(distance_inches * 25.4, 2),
            "segments": [],
            "total_pixels": 0,
            "total_inches": 0,
            "total_mm": 0
        }
        
        total_pixels = 0
        total_inches = 0
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            
            # Calculate pixel distance between points
            # Apply scale factor to convert canvas coordinates to actual image coordinates
            dx = (p2[0] - p1[0]) * scale_factor
            dy = (p2[1] - p1[1]) * scale_factor
            pixel_distance = math.sqrt(dx * dx + dy * dy)
            
            # Apply perspective projection formula
            real_distance_inches = self.pixel_to_real_world(pixel_distance, distance_inches)
            real_distance_mm = real_distance_inches * 25.4
            
            total_pixels += pixel_distance
            total_inches += real_distance_inches
            
            results["segments"].append({
                "segment": f"Point {i + 1} → Point {i + 2}",
                "start_point": {"x": round(p1[0], 2), "y": round(p1[1], 2)},
                "end_point": {"x": round(p2[0], 2), "y": round(p2[1], 2)},
                "pixel_distance": round(pixel_distance, 2),
                "real_distance_inches": round(real_distance_inches, 4),
                "real_distance_mm": round(real_distance_mm, 2)
            })
        
        results["total_pixels"] = round(total_pixels, 2)
        results["total_inches"] = round(total_inches, 4)
        results["total_mm"] = round(total_inches * 25.4, 2)
        
        return results
    
    def get_calibration_info(self):
        """Return camera calibration information for display"""
        return {
            "camera_matrix": self.MTX.tolist(),
            "focal_length_x": self.fx,
            "focal_length_y": self.fy,
            "principal_point": (self.MTX[0, 2], self.MTX[1, 2]),
            "average_focal_length": self.FOCAL_LENGTH_PIXELS,
            "default_distance_inches": self.CAMERA_DISTANCE_INCHES
        }

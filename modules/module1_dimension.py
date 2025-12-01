"""
Module 1: Real-World Dimension Estimation
Uses camera calibration and perspective projection for accurate measurements
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
import math

class DimensionEstimator:
    def __init__(self):
        self.MTX = np.array([
            [5.05796030e+03, 0.00000000e+00, 2.36402921e+03],
            [0.00000000e+00, 5.03423975e+03, 2.73557174e+03],
            [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]
        ])
        
        self.DIST = np.array(
            [[0.11640574, -0.58255541, 0.007427, 0.01609734, 0.87916501]]
        )
        
        self.FOCAL_LENGTH_PIXELS = self.MTX[0, 0]
    
    def undistort_image(self, image):
        """Undistort image using camera calibration data"""
        h, w = image.shape[:2]
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            self.MTX, self.DIST, (w, h), 1, (w, h)
        )
        undistorted = cv2.undistort(image, self.MTX, self.DIST, None, newcameramtx)
        return undistorted
    
    def get_real_world_dimension(self, pixel_size, distance_mm):
        """Calculate real-world dimension using perspective projection formula: W_real = (W_pix * D) / f"""
        if self.FOCAL_LENGTH_PIXELS == 0:
            return 0
        return (pixel_size * distance_mm) / self.FOCAL_LENGTH_PIXELS
    
    def estimate_from_points(self, points, image_shape, distance_inches):
        """
        Estimate dimensions from clicked points (up to 4 points)
        
        Args:
            points: List of (x, y) tuples (up to 4 points)
            image_shape: (height, width) of image
            distance_inches: Distance to object in inches
        
        Returns:
            dict with measurements between consecutive points
        """
        if len(points) < 2:
            return {"error": "Need at least 2 points"}
        
        distance_mm = distance_inches * 25.4
        
        results = {
            "focal_length_px": self.FOCAL_LENGTH_PIXELS,
            "distance_inches": distance_inches,
            "distance_mm": distance_mm,
            "segments": []
        }
        
        for i in range(len(points) - 1):
            p1, p2 = points[i], points[i + 1]
            
            pixel_distance = math.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
            real_distance_mm = self.get_real_world_dimension(pixel_distance, distance_mm)
            real_distance_inches = real_distance_mm / 25.4
            
            results["segments"].append({
                "start_point": i + 1,
                "end_point": i + 2,
                "pixel_distance": round(pixel_distance, 2),
                "real_distance_mm": round(real_distance_mm, 2),
                "real_distance_inches": round(real_distance_inches, 2)
            })
        
        return results





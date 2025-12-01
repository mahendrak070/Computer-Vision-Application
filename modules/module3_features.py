"""
Module 3: Gradients, LoG, Edge/Corner Detection, Segmentation
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
from scipy import ndimage

class GradientComputation:
    """Compute image gradients and related features"""
    
    @staticmethod
    def compute_gradients(image):
        """
        Compute gradient magnitude and angle
        
        Returns:
            (magnitude, angle, grad_x, grad_y)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute gradients using Sobel
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Compute magnitude and angle
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x) * 180 / np.pi
        
        # Normalize magnitude for visualization
        magnitude_vis = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Normalize angle to 0-255
        angle_vis = ((angle + 180) / 360 * 255).astype(np.uint8)
        
        return magnitude_vis, angle_vis, grad_x, grad_y
    
    @staticmethod
    def compute_log(image, sigma=2.0):
        """
        Compute Laplacian of Gaussian
        
        Args:
            image: Input image
            sigma: Standard deviation for Gaussian
        
        Returns:
            LoG filtered image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur first
        blurred = cv2.GaussianBlur(gray, (0, 0), sigma)
        
        # Apply Laplacian
        log = cv2.Laplacian(blurred, cv2.CV_64F)
        
        # Normalize for visualization
        log_vis = cv2.normalize(np.abs(log), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return log_vis, log


class EdgeDetection:
    """Manual edge detection algorithms"""
    
    @staticmethod
    def simple_edge_detection(image, threshold_low=50, threshold_high=150):
        """
        Simple edge detection using gradient magnitude
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Apply hysteresis thresholding
        edges = np.zeros_like(gray)
        edges[magnitude > threshold_high] = 255
        
        # Add weak edges connected to strong edges
        weak_edges = (magnitude > threshold_low) & (magnitude <= threshold_high)
        strong_edges = (magnitude > threshold_high)
        
        # Simple dilation to connect edges
        kernel = np.ones((3, 3), np.uint8)
        strong_dilated = cv2.dilate(strong_edges.astype(np.uint8), kernel, iterations=1)
        
        edges[weak_edges & (strong_dilated > 0)] = 255
        
        return edges
    
    @staticmethod
    def canny_edge_detection(image, threshold1=50, threshold2=150):
        """Canny edge detection"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        edges = cv2.Canny(gray, threshold1, threshold2)
        return edges


class CornerDetection:
    """Manual corner detection algorithms"""
    
    @staticmethod
    def harris_corner_detection(image, block_size=2, ksize=3, k=0.04, threshold=0.01):
        """
        Harris corner detection
        
        Args:
            image: Input image
            block_size: Size of neighborhood
            ksize: Aperture parameter for Sobel
            k: Harris detector free parameter
            threshold: Threshold for corner detection
        
        Returns:
            Image with corners marked
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray_float = np.float32(gray)
        
        # Detect corners
        dst = cv2.cornerHarris(gray_float, block_size, ksize, k)
        
        # Dilate to mark corners
        dst = cv2.dilate(dst, None)
        
        # Create result image
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # Mark corners
        result[dst > threshold * dst.max()] = [0, 0, 255]
        
        # Get corner coordinates
        corners = np.argwhere(dst > threshold * dst.max())
        
        return result, corners, dst
    
    @staticmethod
    def shi_tomasi_corners(image, max_corners=100, quality=0.01, min_distance=10):
        """
        Shi-Tomasi corner detection (Good Features to Track)
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Detect corners
        corners = cv2.goodFeaturesToTrack(gray, max_corners, quality, min_distance)
        
        # Create result image
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        if corners is not None:
            corners = np.int0(corners)
            for corner in corners:
                x, y = corner.ravel()
                cv2.circle(result, (x, y), 5, (0, 255, 0), -1)
        
        return result, corners


class Segmentation:
    """Classical segmentation methods"""
    
    @staticmethod
    def threshold_segmentation(image, method='otsu'):
        """
        Threshold-based segmentation
        
        Args:
            image: Input image
            method: 'otsu', 'adaptive', or 'binary'
        
        Returns:
            Segmented binary image
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        if method == 'otsu':
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        elif method == 'adaptive':
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                          cv2.THRESH_BINARY, 11, 2)
        else:  # binary
            _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        return binary
    
    @staticmethod
    def watershed_segmentation(image):
        """Watershed segmentation"""
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Noise removal
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
        
        # Sure background
        sure_bg = cv2.dilate(opening, kernel, iterations=3)
        
        # Sure foreground
        dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        
        # Unknown region
        unknown = cv2.subtract(sure_bg, sure_fg)
        
        # Marker labelling
        _, markers = cv2.connectedComponents(sure_fg)
        markers = markers + 1
        markers[unknown == 255] = 0
        
        # Apply watershed
        markers = cv2.watershed(image, markers)
        
        # Create result
        result = image.copy()
        result[markers == -1] = [0, 0, 255]
        
        return result, markers
    
    @staticmethod
    def aruco_based_segmentation(image, aruco_dict_type=cv2.aruco.DICT_4X4_50):
        """
        Segmentation using ArUco markers as boundary markers
        
        Detects ArUco markers and segments the region they enclose
        """
        # Get ArUco dictionary
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        aruco_params = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(image)
        
        result = image.copy()
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        if ids is not None and len(corners) > 0:
            # Draw detected markers
            cv2.aruco.drawDetectedMarkers(result, corners, ids)
            
            # Get centers of markers
            centers = []
            for corner in corners:
                center = corner[0].mean(axis=0).astype(int)
                centers.append(center)
            
            # If we have at least 3 markers, create a polygon
            if len(centers) >= 3:
                centers_array = np.array(centers, dtype=np.int32)
                cv2.fillPoly(mask, [centers_array], 255)
                
                # Apply mask to image
                segmented = cv2.bitwise_and(image, image, mask=mask)
                
                # Draw polygon
                cv2.polylines(result, [centers_array], True, (0, 255, 0), 2)
                
                return result, segmented, mask, corners, ids
        
        return result, image, mask, corners, ids
    
    @staticmethod
    def contour_based_segmentation(image):
        """
        Extract exact object boundaries using contours
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create result image
        result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        
        # Draw all contours
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        # Find largest contour
        if len(contours) > 0:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Draw largest contour
            cv2.drawContours(result, [largest_contour], -1, (0, 0, 255), 3)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(largest_contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Create mask
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            return result, mask, largest_contour
        
        return result, np.zeros_like(gray), None


def process_dataset_gradients(images, output_dir='outputs'):
    """
    Process a dataset of images to compute gradients and LoG
    
    Args:
        images: List of images
        output_dir: Directory to save outputs
    
    Returns:
        List of processed results
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    gradient_comp = GradientComputation()
    
    for i, image in enumerate(images):
        # Compute gradients
        mag, angle, grad_x, grad_y = gradient_comp.compute_gradients(image)
        
        # Compute LoG
        log_vis, log_raw = gradient_comp.compute_log(image)
        
        # Save images
        cv2.imwrite(f'{output_dir}/image_{i}_magnitude.png', mag)
        cv2.imwrite(f'{output_dir}/image_{i}_angle.png', angle)
        cv2.imwrite(f'{output_dir}/image_{i}_log.png', log_vis)
        
        results.append({
            'index': i,
            'magnitude': mag,
            'angle': angle,
            'log': log_vis,
            'original': image
        })
    
    return results





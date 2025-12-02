"""
Module 3: Comprehensive Feature Detection
Gradient, LoG, Edges (NMS + Hysteresis), Corners (Harris), Boundary Detection, ArUco Segmentation
Author: Mahendra Krishna Koneru
"""

import cv2
import numpy as np
import math

class GradientComputation:
    """Compute gradient magnitude and angle"""
    
    def __init__(self):
        self.sobel_ksize = 3
    
    def compute_gradient_magnitude(self, image):
        """
        Compute gradient magnitude using Sobel operator
        Returns: 8-bit magnitude image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur for noise reduction
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Sobel gradients
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
        
        # Magnitude
        magnitude = cv2.magnitude(dx, dy)
        
        # Normalize to 0-255
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_8u = mag_normalized.astype(np.uint8)
        
        # Optional: enhance visibility
        kernel = np.ones((3, 3), np.uint8)
        mag_8u = cv2.dilate(mag_8u, kernel)
        
        # Convert to RGB for display
        result = cv2.cvtColor(mag_8u, cv2.COLOR_GRAY2BGR)
        
        return result
    
    def compute_gradient_angle(self, image):
        """
        Compute gradient angle visualized in HSV color space
        Returns: RGB image with angle as hue, magnitude as value
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Sobel gradients
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=self.sobel_ksize)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=self.sobel_ksize)
        
        # Magnitude and angle
        magnitude, angle = cv2.cartToPolar(dx, dy, angleInDegrees=True)
        
        # Normalize magnitude
        mag_normalized = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
        mag_8u = mag_normalized.astype(np.uint8)
        
        # Create mask for significant edges
        _, mask = cv2.threshold(mag_8u, 20, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel)
        
        # Convert angle to hue (0-179 for OpenCV)
        hue = (angle * 0.5).astype(np.uint8)
        hue = np.clip(hue, 0, 179)
        
        # Create HSV image
        saturation = np.full_like(hue, 255)
        value = cv2.bitwise_and(mag_8u, mask)
        
        hsv = cv2.merge([hue, saturation, value])
        
        # Convert to RGB
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return rgb


class LaplacianOfGaussian:
    """Laplacian of Gaussian (LoG) edge detection"""
    
    def __init__(self, gauss_ksize=5, gauss_sigma=1.0, laplace_ksize=3):
        self.gauss_ksize = gauss_ksize
        self.gauss_sigma = gauss_sigma
        self.laplace_ksize = laplace_ksize
    
    def detect(self, image):
        """
        Apply LoG: Gaussian blur + Laplacian
        Returns: Edge image
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.gauss_ksize, self.gauss_ksize), 
                                   self.gauss_sigma, self.gauss_sigma)
        
        # Laplacian
        laplacian = cv2.Laplacian(blurred, cv2.CV_32F, ksize=self.laplace_ksize)
        
        # Convert to absolute values and scale
        log_abs = cv2.convertScaleAbs(laplacian)
        
        # Enhance visibility
        kernel = np.ones((3, 3), np.uint8)
        log_abs = cv2.dilate(log_abs, kernel)
        
        # Equalize for better visualization
        log_abs = cv2.equalizeHist(log_abs)
        
        # Convert to RGB
        result = cv2.cvtColor(log_abs, cv2.COLOR_GRAY2BGR)
        
        return result


class EdgeDetection:
    """Custom edge detection with NMS and hysteresis"""
    
    def __init__(self, sigma=1.0, low_threshold=15, high_threshold=40):
        self.sigma = sigma
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
    
    def detect(self, image):
        """
        Edge detection: Sobel + NMS + Hysteresis
        Returns: (edge_image, edge_count)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur
        ksize = int(2 * round(2 * self.sigma) + 1)
        blurred = cv2.GaussianBlur(gray, (ksize, ksize), self.sigma)
        
        # Sobel gradients
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        
        # Magnitude and angle
        magnitude = cv2.magnitude(dx, dy)
        angle = cv2.phase(dx, dy, angleInDegrees=True)
        
        # Non-maximum suppression
        nms = self._non_maximum_suppression(magnitude, angle)
        
        # Normalize
        nms_normalized = cv2.normalize(nms, None, 0, 255, cv2.NORM_MINMAX)
        nms_8u = nms_normalized.astype(np.uint8)
        
        # Hysteresis thresholding
        edges = self._hysteresis_threshold(nms_8u, self.low_threshold, self.high_threshold)
        
        # Thicken edges for visibility
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel)
        
        # Count edge pixels
        edge_count = cv2.countNonZero(edges)
        
        # Create overlay on original image
        if len(image.shape) == 3:
            result = image.copy()
        else:
            result = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        
        # Create red overlay for edges
        red_mask = np.zeros_like(result)
        red_mask[:,:,2] = edges  # Red channel (BGR format)
        result = cv2.addWeighted(result, 0.7, red_mask, 0.8, 0)
        
        return result, edge_count
    
    def _non_maximum_suppression(self, magnitude, angle):
        """Apply non-maximum suppression"""
        rows, cols = magnitude.shape
        nms = np.zeros_like(magnitude)
        
        # Normalize angle to 0-180
        angle = angle % 180
        
        for y in range(1, rows - 1):
            for x in range(1, cols - 1):
                a = angle[y, x]
                m = magnitude[y, x]
                
                # Determine neighbors based on gradient direction
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    # Horizontal
                    n1 = magnitude[y, x-1]
                    n2 = magnitude[y, x+1]
                elif 22.5 <= a < 67.5:
                    # Diagonal /
                    n1 = magnitude[y-1, x+1]
                    n2 = magnitude[y+1, x-1]
                elif 67.5 <= a < 112.5:
                    # Vertical
                    n1 = magnitude[y-1, x]
                    n2 = magnitude[y+1, x]
                else:  # 112.5 <= a < 157.5
                    # Diagonal \
                    n1 = magnitude[y-1, x-1]
                    n2 = magnitude[y+1, x+1]
                
                # Keep only local maxima
                if m >= n1 and m >= n2:
                    nms[y, x] = m
        
        return nms
    
    def _hysteresis_threshold(self, nms, low, high):
        """
        Apply hysteresis thresholding using morphological propagation
        More efficient than iterative pixel-by-pixel checking
        """
        # Create strong and weak edge maps
        strong = np.zeros_like(nms, dtype=np.uint8)
        weak = np.zeros_like(nms, dtype=np.uint8)
        
        strong[nms >= high] = 255
        weak[nms >= low] = 255
        
        # Remove strong from weak to get only weak edges
        weak_only = cv2.subtract(weak, strong)
        
        # Propagate strong edges into connected weak edges
        connected = strong.copy()
        kernel = np.ones((3, 3), np.uint8)
        
        # Iteratively dilate and connect weak edges (max 12 iterations)
        for _ in range(12):
            # Dilate connected edges
            dilated = cv2.dilate(connected, kernel, iterations=1)
            
            # Find weak edges that touch dilated strong edges
            add = cv2.bitwise_and(dilated, weak_only)
            
            # If no new edges added, we're done
            if cv2.countNonZero(add) == 0:
                break
            
            # Add these edges to connected
            connected = cv2.bitwise_or(connected, add)
            
            # Remove them from weak_only
            weak_only = cv2.subtract(weak_only, add)
        
        return connected


class CornerDetection:
    """Harris corner detection"""
    
    def __init__(self, k=0.04, threshold=70, window_size=5):
        self.k = k
        self.threshold = threshold
        self.window_size = window_size
    
    def detect(self, image):
        """
        Harris corner detection
        Returns: (result_image, corner_count)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0.8)
        
        # Compute gradients
        Ix = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        Iy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        
        # Products of derivatives
        Ixx = Ix * Ix
        Iyy = Iy * Iy
        Ixy = Ix * Iy
        
        # Gaussian window
        Sxx = cv2.GaussianBlur(Ixx, (self.window_size, self.window_size), 1.0)
        Syy = cv2.GaussianBlur(Iyy, (self.window_size, self.window_size), 1.0)
        Sxy = cv2.GaussianBlur(Ixy, (self.window_size, self.window_size), 1.0)
        
        # Harris response
        det = Sxx * Syy - Sxy * Sxy
        trace = Sxx + Syy
        R = det - self.k * (trace ** 2)
        
        # Normalize
        R_normalized = cv2.normalize(R, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_32F)
        R_8u = R_normalized.astype(np.uint8)
        
        # Threshold
        _, threshold_mask = cv2.threshold(R_8u, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Non-maximum suppression (local maxima)
        dilated = cv2.dilate(R_8u, np.ones((3, 3), np.uint8))
        local_max = (R_8u == dilated) & (threshold_mask == 255)
        
        # Find corner coordinates
        corners = np.argwhere(local_max)
        corner_count = len(corners)
        
        # Create result with heatmap and corner markers
        result = image.copy()
        
        # Add heatmap overlay
        heatmap = cv2.equalizeHist(R_8u)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        result = cv2.addWeighted(result, 0.6, heatmap_color, 0.4, 0)
        
        # Draw corner points (green circles)
        for corner in corners:
            y, x = corner
            cv2.circle(result, (x, y), 5, (0, 255, 0), 2, cv2.LINE_AA)
            # Add small center dot for visibility
            cv2.circle(result, (x, y), 2, (0, 255, 0), -1, cv2.LINE_AA)
        
        return result, corner_count


class BoundaryDetection:
    """Boundary detection using Canny + morphological operations + contours"""
    
    def __init__(self):
        pass
    
    def detect(self, image):
        """
        Detect object boundaries using Canny + morphological operations + contour analysis
        Returns: (result_image, contour_count)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        h, w = gray.shape
        img_area = h * w
        
        # Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0.8)
        
        # Compute gradients for Otsu auto-threshold
        dx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(dx, dy)
        mag_8u = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Auto-threshold using Otsu on gradient magnitude
        otsu_val, _ = cv2.threshold(mag_8u, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        high_threshold = max(30, int(otsu_val))
        low_threshold = max(5, int(0.5 * high_threshold))
        
        # Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold, apertureSize=3, L2gradient=True)
        
        # Morphological closing to connect nearby edges
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area (min 2% of image area)
        min_area = 0.02 * img_area
        significant_contours = []
        
        # Find best contour (largest area with preference for center)
        cx0, cy0 = w / 2, h / 2
        diag = np.hypot(w, h)
        best_score = -1
        best_contour = None
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_area:
                continue
            
            # Compute centroid
            M = cv2.moments(cnt)
            if M['m00'] == 0:
                continue
            
            cx = M['m10'] / M['m00']
            cy = M['m01'] / M['m00']
            
            # Distance from image center (normalized)
            dist = np.hypot(cx - cx0, cy - cy0) / diag
            
            # Score: larger area and closer to center is better
            center_bonus = 1.2 if dist <= 0.4 else 1.0
            score = area * center_bonus * (1.0 - 0.6 * dist)
            
            if score > best_score:
                best_score = score
                best_contour = cnt
            
            significant_contours.append(cnt)
        
        # Create result
        result = image.copy()
        
        if best_contour is not None:
            # Approximate polygon for cleaner boundary
            perimeter = cv2.arcLength(best_contour, True)
            epsilon = max(0.5, 0.015 * perimeter)
            approx = cv2.approxPolyDP(best_contour, epsilon, True)
            
            # Draw best contour with green semi-transparent fill
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(mask, [approx], 255)
            
            # Green overlay
            overlay = result.copy()
            overlay[mask == 255] = overlay[mask == 255] * 0.65 + np.array([0, 255, 0]) * 0.35
            result = overlay.astype(np.uint8)
            
            # Draw boundary (thick green line)
            cv2.polylines(result, [approx], True, (0, 255, 0), 3, cv2.LINE_AA)
            
            # Draw all other contours (thinner)
            other_contours = [cnt for cnt in significant_contours if cnt is not best_contour]
            cv2.drawContours(result, other_contours, -1, (0, 200, 200), 2, cv2.LINE_AA)
            
            # Draw centroid and bounding box for best contour
            M = cv2.moments(best_contour)
            if M['m00'] != 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                cv2.circle(result, (cx, cy), 6, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(result, (cx, cy), 8, (0, 0, 255), 2, cv2.LINE_AA)
            
            x, y, w, h = cv2.boundingRect(best_contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 0, 0), 2, cv2.LINE_AA)
        
        return result, len(significant_contours)


class ArucoSegmentation:
    """ArUco marker-based object segmentation"""
    
    def __init__(self):
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
    
    def segment(self, image):
        """
        Segment object using ArUco markers
        Returns: (result_image, marker_count)
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)
        
        result = image.copy()
        marker_count = 0
        
        if ids is not None and len(ids) > 0:
            marker_count = len(ids)
            
            # Draw detected markers with IDs
            cv2.aruco.drawDetectedMarkers(result, corners, ids, borderColor=(0, 255, 0))
            
            # If multiple markers, create convex hull
            if len(corners) >= 3:
                all_corners = np.vstack([c.reshape(-1, 2) for c in corners])
                hull = cv2.convexHull(all_corners.astype(np.float32))
                hull_int = hull.astype(np.int32)
                
                # Draw segmentation boundary (thick green line)
                cv2.polylines(result, [hull_int], True, (0, 255, 0), 4, cv2.LINE_AA)
                
                # Fill with semi-transparent overlay
                overlay = result.copy()
                cv2.fillPoly(overlay, [hull_int], (0, 255, 100))
                result = cv2.addWeighted(result, 0.7, overlay, 0.3, 0)
            
            # Draw marker centers (blue dots)
            for corner in corners:
                center = corner[0].mean(axis=0).astype(int)
                cv2.circle(result, tuple(center), 6, (255, 0, 0), -1, cv2.LINE_AA)
                cv2.circle(result, tuple(center), 8, (255, 0, 0), 2, cv2.LINE_AA)
        
        return result, marker_count


# Legacy classes for backward compatibility
class Segmentation:
    """Simple segmentation using thresholding"""
    
    def segment(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        result = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        return result

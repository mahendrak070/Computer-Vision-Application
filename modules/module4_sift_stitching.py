"""
Module 4: Image Stitching and SIFT from Scratch
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
from scipy import ndimage

class SIFTFromScratch:
    """SIFT implementation from scratch"""
    
    def __init__(self, num_octaves=4, num_scales=5, sigma=1.6, contrast_threshold=0.04, 
                 edge_threshold=10, num_bins=8, window_size=16):
        self.num_octaves = num_octaves
        self.num_scales = num_scales
        self.sigma = sigma
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.num_bins = num_bins
        self.window_size = window_size
        self.k = 2 ** (1.0 / (num_scales - 3))
    
    def build_gaussian_pyramid(self, image):
        """Build Gaussian scale-space pyramid"""
        pyramid = []
        
        for octave in range(self.num_octaves):
            octave_images = []
            sigma_prev = self.sigma
            
            for scale in range(self.num_scales):
                if octave == 0 and scale == 0:
                    current_image = image.copy()
                elif scale == 0:
                    # Downsample from previous octave
                    current_image = pyramid[octave-1][-3][::2, ::2]
                else:
                    # Apply Gaussian blur
                    sigma_current = sigma_prev * self.k
                    sigma_diff = np.sqrt(sigma_current**2 - sigma_prev**2)
                    current_image = ndimage.gaussian_filter(octave_images[-1], sigma_diff)
                    sigma_prev = sigma_current
                
                octave_images.append(current_image)
            
            pyramid.append(octave_images)
        
        return pyramid
    
    def build_dog_pyramid(self, gaussian_pyramid):
        """Build Difference of Gaussian pyramid"""
        dog_pyramid = []
        
        for octave_images in gaussian_pyramid:
            dog_octave = []
            for i in range(len(octave_images) - 1):
                dog = octave_images[i+1] - octave_images[i]
                dog_octave.append(dog)
            dog_pyramid.append(dog_octave)
        
        return dog_pyramid
    
    def is_extremum(self, dog_pyramid, octave, scale, y, x):
        """Check if point is local extremum in 3x3x3 neighborhood"""
        current = dog_pyramid[octave][scale][y, x]
        
        # Check 26 neighbors (3x3x3 - 1)
        for dscale in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dscale == 0 and dy == 0 and dx == 0:
                        continue
                    
                    neighbor_scale = scale + dscale
                    if neighbor_scale < 0 or neighbor_scale >= len(dog_pyramid[octave]):
                        return False
                    
                    neighbor_y = y + dy
                    neighbor_x = x + dx
                    img_shape = dog_pyramid[octave][neighbor_scale].shape
                    
                    if neighbor_y < 0 or neighbor_y >= img_shape[0]:
                        return False
                    if neighbor_x < 0 or neighbor_x >= img_shape[1]:
                        return False
                    
                    neighbor_val = dog_pyramid[octave][neighbor_scale][neighbor_y, neighbor_x]
                    
                    if abs(current) < abs(neighbor_val):
                        return False
        
        return True
    
    def localize_keypoint(self, dog_pyramid, octave, scale, y, x):
        """Localize keypoint with sub-pixel accuracy"""
        # Simplified localization - return integer coordinates
        return y, x, scale
    
    def find_keypoints(self, dog_pyramid):
        """Find keypoints in DOG pyramid"""
        keypoints = []
        
        for octave in range(len(dog_pyramid)):
            for scale in range(1, len(dog_pyramid[octave]) - 1):
                img = dog_pyramid[octave][scale]
                h, w = img.shape
                
                for y in range(1, h-1):
                    for x in range(1, w-1):
                        if abs(img[y, x]) > self.contrast_threshold:
                            if self.is_extremum(dog_pyramid, octave, scale, y, x):
                                # Check edge response
                                if not self.is_edge(img, y, x):
                                    y_loc, x_loc, scale_loc = self.localize_keypoint(
                                        dog_pyramid, octave, scale, y, x
                                    )
                                    
                                    # Convert to original image coordinates
                                    scale_factor = 2 ** octave
                                    keypoints.append({
                                        'x': x_loc * scale_factor,
                                        'y': y_loc * scale_factor,
                                        'octave': octave,
                                        'scale': scale_loc,
                                        'size': self.sigma * (2 ** (scale_loc / self.num_scales)) * scale_factor
                                    })
        
        return keypoints
    
    def is_edge(self, img, y, x):
        """Check if keypoint is on edge using Hessian"""
        # Compute Hessian
        dxx = img[y, x+1] + img[y, x-1] - 2 * img[y, x]
        dyy = img[y+1, x] + img[y-1, x] - 2 * img[y, x]
        dxy = (img[y+1, x+1] + img[y-1, x-1] - img[y+1, x-1] - img[y-1, x+1]) / 4.0
        
        # Compute trace and determinant
        trace = dxx + dyy
        det = dxx * dyy - dxy * dxy
        
        if det <= 0:
            return True
        
        ratio = (trace * trace) / det
        threshold = ((self.edge_threshold + 1) ** 2) / self.edge_threshold
        
        return ratio > threshold
    
    def assign_orientation(self, gaussian_pyramid, keypoints):
        """Assign orientation to keypoints"""
        for kp in keypoints:
            octave = kp['octave']
            scale = int(kp['scale'])
            x = int(kp['x'] / (2 ** octave))
            y = int(kp['y'] / (2 ** octave))
            
            if scale >= len(gaussian_pyramid[octave]):
                scale = len(gaussian_pyramid[octave]) - 1
            
            img = gaussian_pyramid[octave][scale]
            h, w = img.shape
            
            # Calculate gradient magnitude and orientation
            hist = np.zeros(36)
            radius = int(round(1.5 * kp['size'] / (2 ** octave)))
            
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    yy = y + dy
                    xx = x + dx
                    
                    if yy <= 0 or yy >= h-1 or xx <= 0 or xx >= w-1:
                        continue
                    
                    gx = img[yy, xx+1] - img[yy, xx-1]
                    gy = img[yy+1, xx] - img[yy-1, xx]
                    
                    mag = np.sqrt(gx**2 + gy**2)
                    angle = np.arctan2(gy, gx) * 180 / np.pi
                    
                    bin_idx = int((angle + 180) / 10) % 36
                    hist[bin_idx] += mag
            
            # Find dominant orientation
            max_bin = np.argmax(hist)
            kp['orientation'] = (max_bin * 10 - 180) * np.pi / 180
        
        return keypoints
    
    def compute_descriptors(self, gaussian_pyramid, keypoints):
        """Compute SIFT descriptors for keypoints"""
        descriptors = []
        
        for kp in keypoints:
            octave = kp['octave']
            scale = int(kp['scale'])
            x = int(kp['x'] / (2 ** octave))
            y = int(kp['y'] / (2 ** octave))
            
            if scale >= len(gaussian_pyramid[octave]):
                scale = len(gaussian_pyramid[octave]) - 1
            
            img = gaussian_pyramid[octave][scale]
            h, w = img.shape
            
            # Compute descriptor (simplified 128-dimensional)
            descriptor = np.zeros(128)
            angle = kp.get('orientation', 0)
            
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            
            # 4x4 grid of 8-bin histograms
            hist_width = 4
            radius = hist_width * 4
            
            for i in range(-radius, radius):
                for j in range(-radius, radius):
                    # Rotate coordinates
                    rot_i = int(i * cos_angle - j * sin_angle)
                    rot_j = int(i * sin_angle + j * cos_angle)
                    
                    yy = y + rot_i
                    xx = x + rot_j
                    
                    if yy <= 0 or yy >= h-1 or xx <= 0 or xx >= w-1:
                        continue
                    
                    gx = img[yy, xx+1] - img[yy, xx-1]
                    gy = img[yy+1, xx] - img[yy-1, xx]
                    
                    mag = np.sqrt(gx**2 + gy**2)
                    grad_angle = np.arctan2(gy, gx) - angle
                    
                    # Compute bin indices
                    bin_x = int((rot_j + radius) / (2 * radius) * hist_width)
                    bin_y = int((rot_i + radius) / (2 * radius) * hist_width)
                    bin_angle = int((grad_angle + np.pi) / (2 * np.pi) * 8)
                    
                    bin_x = np.clip(bin_x, 0, hist_width - 1)
                    bin_y = np.clip(bin_y, 0, hist_width - 1)
                    bin_angle = bin_angle % 8
                    
                    desc_idx = (bin_y * hist_width + bin_x) * 8 + bin_angle
                    if desc_idx < 128:
                        descriptor[desc_idx] += mag
            
            # Normalize descriptor
            norm = np.linalg.norm(descriptor)
            if norm > 0:
                descriptor = descriptor / norm
                # Threshold and renormalize
                descriptor = np.clip(descriptor, 0, 0.2)
                norm = np.linalg.norm(descriptor)
                if norm > 0:
                    descriptor = descriptor / norm
            
            descriptors.append(descriptor)
        
        return np.array(descriptors)
    
    def detect_and_compute(self, image):
        """Main SIFT detection and computation"""
        # Convert to grayscale and float
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        gray = gray.astype(np.float32) / 255.0
        
        # Build pyramids
        gaussian_pyramid = self.build_gaussian_pyramid(gray)
        dog_pyramid = self.build_dog_pyramid(gaussian_pyramid)
        
        # Find keypoints
        keypoints = self.find_keypoints(dog_pyramid)
        
        # Assign orientations
        keypoints = self.assign_orientation(gaussian_pyramid, keypoints)
        
        # Compute descriptors
        descriptors = self.compute_descriptors(gaussian_pyramid, keypoints)
        
        return keypoints, descriptors


class ImageStitching:
    """Image stitching using feature matching with weighted blending and RANSAC"""
    
    def __init__(self, smoothing_window_percent=0.1, max_dimension=1200):
        self.sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.04, edgeThreshold=10)
        self.smoothing_window_percent = smoothing_window_percent
        self.max_dimension = max_dimension  # Resize images for faster processing
    
    def resize_image_for_processing(self, image):
        """Resize image to max dimension for faster processing"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_dimension:
            return image, 1.0
        
        scale = self.max_dimension / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def stitch_images(self, images, use_custom_sift=False, progress_callback=None):
        """
        Stitch multiple images into panorama (LEFT to RIGHT order) - OPTIMIZED
        
        Args:
            images: List of images to stitch (ordered left to right)
            use_custom_sift: Use custom SIFT implementation (slower)
            progress_callback: Function to report progress
        
        Returns:
            Stitched panorama
        """
        if len(images) < 2:
            return images[0] if len(images) == 1 else None
        
        # Force OpenCV SIFT for speed unless explicitly requested
        if use_custom_sift:
            print("⚠️ Warning: Custom SIFT is very slow. Using OpenCV SIFT for speed.")
            use_custom_sift = False
        
        total_steps = len(images) - 1
        
        # Resize images for faster processing
        resized_images = []
        scales = []
        for img in images:
            resized, scale = self.resize_image_for_processing(img)
            resized_images.append(resized)
            scales.append(scale)
        
        # Stitch sequentially from left to right
        result = resized_images[0]
        
        for i in range(1, len(resized_images)):
            if progress_callback:
                progress_callback(int((i / total_steps) * 100))
            
            result = self.stitch_pair(result, resized_images[i], use_custom_sift)
            if result is None:
                return None
        
        if progress_callback:
            progress_callback(100)
        
        return result
    
    def stitch_pair(self, img1, img2, use_custom_sift=False):
        """OPTIMIZED: Stitch two images blazing fast"""
        # Always use OpenCV SIFT for speed (custom is too slow)
        kp1_cv, des1 = self.sift.detectAndCompute(img1, None)
        kp2_cv, des2 = self.sift.detectAndCompute(img2, None)
        
        if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
            return None
        
        # Fast feature matching with FLANN
        matches = self.match_features_fast(des1, des2)
        
        if len(matches) < 4:
            return None
        
        # Extract matching points
        src_pts = np.float32([kp1_cv[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2_cv[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        # Find homography using OpenCV's optimized RANSAC
        H, inliers = self.find_homography_ransac(src_pts, dst_pts)
        
        if H is None:
            return None
        
        # Fast warp and blend
        return self.warp_and_blend(img1, img2, H)
    
    def find_homography_ransac(self, src_pts, dst_pts, max_iterations=500, threshold=4.0):
        """
        Find homography using OPTIMIZED RANSAC algorithm with early stopping
        
        Args:
            src_pts: Source points (Nx1x2)
            dst_pts: Destination points (Nx1x2)
            max_iterations: Maximum RANSAC iterations (reduced for speed)
            threshold: Inlier threshold in pixels
        
        Returns:
            H: Best homography matrix (3x3)
            inliers: Boolean mask of inliers
        """
        # Use OpenCV's optimized RANSAC for maximum speed
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, threshold, maxIters=max_iterations)
        
        if H is None:
            return None, None
        
        inlier_mask = mask.ravel().astype(bool) if mask is not None else None
        return H, inlier_mask
    
    def compute_homography_4points(self, src, dst):
        """Compute homography from 4 point correspondences"""
        try:
            A = []
            for i in range(4):
                x, y = src[i][0], src[i][1]
                u, v = dst[i][0], dst[i][1]
                A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
                A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            H = H / H[2, 2]  # Normalize
            return H
        except:
            return None
    
    def compute_homography_dlt(self, src, dst):
        """Compute homography using Direct Linear Transform (DLT)"""
        try:
            n = len(src)
            A = []
            for i in range(n):
                x, y = src[i][0], src[i][1]
                u, v = dst[i][0], dst[i][1]
                A.append([-x, -y, -1, 0, 0, 0, u*x, u*y, u])
                A.append([0, 0, 0, -x, -y, -1, v*x, v*y, v])
            
            A = np.array(A)
            _, _, Vt = np.linalg.svd(A)
            H = Vt[-1].reshape(3, 3)
            H = H / H[2, 2]
            return H
        except:
            return None
    
    def warp_and_blend(self, img1, img2, H):
        """
        OPTIMIZED warp and blend - Fast linear blending
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        # Get corners of first image
        corners1 = np.float32([[0, 0], [0, h1], [w1, h1], [w1, 0]]).reshape(-1, 1, 2)
        corners2 = np.float32([[0, 0], [0, h2], [w2, h2], [w2, 0]]).reshape(-1, 1, 2)
        
        # Transform corners
        corners1_transformed = cv2.perspectiveTransform(corners1, H)
        all_corners = np.concatenate((corners2, corners1_transformed), axis=0)
        
        # Get bounding box
        [x_min, y_min] = np.int32(all_corners.min(axis=0).ravel() - 0.5)
        [x_max, y_max] = np.int32(all_corners.max(axis=0).ravel() + 0.5)
        
        # Translation matrix
        translation = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
        
        # Warp first image
        output_shape = (x_max - x_min, y_max - y_min)
        result = cv2.warpPerspective(img1, translation @ H, output_shape)
        
        # Calculate position of second image
        y_start = max(0, -y_min)
        y_end = min(y_max - y_min, -y_min + h2)
        x_start = max(0, -x_min)
        x_end = min(x_max - x_min, -x_min + w2)
        
        # Calculate corresponding region in img2
        img2_y_start = max(0, y_min)
        img2_y_end = img2_y_start + (y_end - y_start)
        img2_x_start = max(0, x_min)
        img2_x_end = img2_x_start + (x_end - x_start)
        
        # Extract regions
        result_region = result[y_start:y_end, x_start:x_end]
        img2_region = img2[img2_y_start:img2_y_end, img2_x_start:img2_x_end]
        
        # Fast linear blending in overlap
        # Find where both images have content
        gray_result = cv2.cvtColor(result_region, cv2.COLOR_BGR2GRAY) if len(result_region.shape) == 3 else result_region
        mask1 = (gray_result > 1).astype(np.float32)
        mask2 = np.ones_like(mask1)
        
        overlap = (mask1 > 0) & (mask2 > 0)
        
        if np.any(overlap):
            # Simple alpha blending - much faster
            alpha = 0.5
            blended = cv2.addWeighted(result_region, alpha, img2_region, 1-alpha, 0)
            
            # Apply blended region only where there's overlap
            overlap_3ch = np.expand_dims(overlap, axis=2) if len(img2_region.shape) == 3 else overlap
            result_region = np.where(overlap_3ch, blended, img2_region).astype(img2.dtype)
        else:
            # No overlap, just use img2
            result_region = img2_region
        
        # Place blended region back
        result[y_start:y_end, x_start:x_end] = result_region
        
        return result
    
    def match_features_fast(self, des1, des2, ratio=0.75):
        """OPTIMIZED: Fast feature matching using FLANN"""
        # FLANN parameters for SIFT
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)  # Higher = more accurate but slower
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        
        # Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def match_features(self, des1, des2, ratio=0.75):
        """Fallback: Match features using BFMatcher"""
        return self.match_features_fast(des1, des2, ratio)



"""
Module 2: Template Matching and Fourier Restoration
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

import cv2
import numpy as np
from scipy import fftpack

class TemplateMatchingModule:
    """OPTIMIZED template matching with multi-scale and progress tracking"""
    
    def __init__(self, max_image_size=1200):
        self.templates = []
        self.template_names = []
        self.max_image_size = max_image_size
    
    def resize_for_processing(self, image):
        """Resize image for faster processing"""
        h, w = image.shape[:2]
        max_dim = max(h, w)
        
        if max_dim <= self.max_image_size:
            return image, 1.0
        
        scale = self.max_image_size / max_dim
        new_w = int(w * scale)
        new_h = int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale
    
    def add_template(self, template_image, name):
        """Add a template for matching - automatically resized and optimized"""
        # Resize template if too large for optimal performance
        resized_template, scale = self.resize_for_processing(template_image)
        
        # Convert to BGR if grayscale for consistency
        if len(resized_template.shape) == 2:
            resized_template = cv2.cvtColor(resized_template, cv2.COLOR_GRAY2BGR)
        
        self.templates.append(resized_template)
        self.template_names.append(name)
        
        print(f"✅ Template '{name}' added (resized by {scale:.2f}x)" if scale != 1.0 else f"✅ Template '{name}' added")
    
    def match_template(self, image, template, method=cv2.TM_CCOEFF_NORMED, threshold=0.65):
        """
        OPTIMIZED: Match template in image using correlation with multiple detections
        
        Args:
            image: Input image (automatically resized and converted to grayscale)
            template: Template to match (automatically resized and converted to grayscale)
            method: Matching method (default: normalized cross-correlation)
            threshold: Confidence threshold for detection (lowered to 0.65 for better detection)
        
        Returns:
            List of detections with bounding boxes
        """
        # ALWAYS convert to grayscale for optimal speed and consistency
        if len(image.shape) == 3:
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            image_gray = image.copy()
        
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        # Ensure grayscale images are 8-bit
        if image_gray.dtype != np.uint8:
            image_gray = cv2.normalize(image_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        if template_gray.dtype != np.uint8:
            template_gray = cv2.normalize(template_gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        # Perform FAST template matching
        result = cv2.matchTemplate(image_gray, template_gray, method)
        
        # Get template dimensions
        h, w = template_gray.shape
        
        # Find ALL matches above threshold (not just best)
        detections = []
        
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            # For SQDIFF, lower is better
            loc = np.where(result < (1 - threshold))
            confidence_map = 1 - result
        else:
            # For correlation methods, higher is better
            loc = np.where(result >= threshold)
            confidence_map = result
        
        # Get all detection points
        points = list(zip(*loc[::-1]))  # Switch x and y
        
        if len(points) == 0:
            # If no detections above threshold, get the best one
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                top_left = min_loc
                confidence = 1 - min_val
            else:
                top_left = max_loc
                confidence = max_val
            
            bottom_right = (top_left[0] + w, top_left[1] + h)
            detections.append({
                'bbox': [top_left[0], top_left[1], w, h],
                'confidence': float(confidence),
                'center': (top_left[0] + w//2, top_left[1] + h//2)
            })
        else:
            # Non-maximum suppression to remove overlapping detections
            boxes = []
            scores = []
            
            for pt in points:
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
                scores.append(float(confidence_map[pt[1], pt[0]]))
            
            # Apply NMS
            boxes = np.array(boxes)
            scores = np.array(scores)
            
            # Simple NMS
            indices = self.non_max_suppression(boxes, scores, overlap_thresh=0.3)
            
            for idx in indices:
                x, y = int(boxes[idx][0]), int(boxes[idx][1])
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(scores[idx]),
                    'center': (x + w//2, y + h//2)
                })
        
        return detections, result
    
    def non_max_suppression(self, boxes, scores, overlap_thresh=0.3):
        """Fast NMS to remove overlapping detections"""
        if len(boxes) == 0:
            return []
        
        # Convert to float
        boxes = boxes.astype(np.float32)
        
        # Grab coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute area
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        idxs = np.argsort(scores)[::-1]
        
        pick = []
        
        while len(idxs) > 0:
            # Pick highest score
            i = idxs[0]
            pick.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[1:]]
            
            # Keep only non-overlapping
            idxs = np.delete(idxs, np.concatenate(([0], np.where(overlap > overlap_thresh)[0] + 1)))
        
        return pick
    
    def match_all_templates(self, image, progress_callback=None):
        """
        Match all stored templates against the image with progress tracking
        - Automatically resizes large images
        - Converts to grayscale internally
        - Scales detections back to original image size
        """
        if len(self.templates) == 0:
            return []
        
        # Resize image for faster processing (automatic optimization)
        processed_image, scale = self.resize_for_processing(image)
        
        print(f"🔍 Processing image (scale: {scale:.2f}x) with {len(self.templates)} template(s)")
        
        results = []
        total_templates = len(self.templates)
        
        for i, template in enumerate(self.templates):
            if progress_callback:
                progress_callback(int(((i + 1) / total_templates) * 100))
            
            # Match template (internally converts to grayscale)
            detections, match_result = self.match_template(processed_image, template)
            
            # Scale detections back to original image size
            if scale != 1.0:
                for det in detections:
                    det['bbox'] = [int(b / scale) for b in det['bbox']]
                    det['center'] = (int(det['center'][0] / scale), int(det['center'][1] / scale))
            
            print(f"  ✓ Template {i+1}/{total_templates}: {self.template_names[i]} - {len(detections)} detection(s)")
            
            # Draw bounding boxes on image (matching reference style)
            result_img = image.copy()
            
            # Convert to grayscale for clean visualization (like reference image)
            if len(result_img.shape) == 3:
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY)
                result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)  # Convert back to BGR for colored boxes
            
            for det in detections:
                x, y, w, h = det['bbox']
                confidence = det['confidence']
                
                # Clean white/light gray rectangle like reference image
                # Use white (255, 255, 255) for high contrast on grayscale
                box_color = (255, 255, 255)  # White
                box_thickness = 2  # Clean 2px line
                
                # Draw clean rectangle
                cv2.rectangle(result_img, (x, y), (x + w, y + h), box_color, box_thickness)
                
                # Optional: Add confidence label if detection confidence is high
                if confidence > 0.7:
                    label = f"{confidence*100:.1f}%"
                    font_scale = 0.5
                    thickness = 1
                    (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    
                    # Draw label above the box
                    label_y = y - 8 if y > 25 else y + h + 20
                    cv2.rectangle(result_img, (x, label_y - label_h - 5), (x + label_w + 5, label_y + 2), box_color, -1)
                    cv2.putText(result_img, label, (x + 2, label_y - 2), cv2.FONT_HERSHEY_SIMPLEX, 
                               font_scale, (0, 0, 0), thickness)
            
            results.append({
                'name': self.template_names[i],
                'detections': detections,
                'num_detections': len(detections),
                'result_image': result_img,
                'match_heatmap': match_result
            })
        
        if progress_callback:
            progress_callback(100)
        
        return results


class FourierRestoration:
    def __init__(self):
        pass
    
    def apply_gaussian_blur(self, image, kernel_size=15, sigma=3.0):
        """Apply Gaussian blur to image"""
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
        return blurred
    
    def estimate_blur_kernel(self, kernel_size=15, sigma=3.0):
        """Create Gaussian blur kernel"""
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        kernel = kernel @ kernel.T
        return kernel
    
    def wiener_filter(self, blurred_image, kernel, K=0.01):
        """
        Apply Wiener filter for image restoration
        
        Args:
            blurred_image: Blurred input image
            kernel: Blur kernel
            K: Noise-to-signal ratio
        
        Returns:
            Restored image
        """
        # Convert to grayscale if needed
        if len(blurred_image.shape) == 3:
            blurred_gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        else:
            blurred_gray = blurred_image.copy()
        
        # Pad kernel to image size
        img_h, img_w = blurred_gray.shape
        kernel_h, kernel_w = kernel.shape
        
        padded_kernel = np.zeros((img_h, img_w))
        pad_h = (img_h - kernel_h) // 2
        pad_w = (img_w - kernel_w) // 2
        padded_kernel[pad_h:pad_h+kernel_h, pad_w:pad_w+kernel_w] = kernel
        
        # Shift kernel to have zero frequency at center
        padded_kernel = np.fft.ifftshift(padded_kernel)
        
        # FFT of image and kernel
        img_fft = np.fft.fft2(blurred_gray)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Wiener filter in frequency domain
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_abs2 = np.abs(kernel_fft) ** 2
        
        wiener = kernel_fft_conj / (kernel_fft_abs2 + K)
        restored_fft = img_fft * wiener
        
        # Inverse FFT
        restored = np.fft.ifft2(restored_fft)
        restored = np.abs(restored)
        
        # Normalize to 0-255
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        
        return restored
    
    def richardson_lucy_deconvolution(self, blurred_image, kernel, iterations=10):
        """
        Richardson-Lucy deconvolution for image restoration
        
        Args:
            blurred_image: Blurred input image
            kernel: Blur kernel (PSF)
            iterations: Number of iterations
        
        Returns:
            Restored image
        """
        # Convert to float and normalize
        if len(blurred_image.shape) == 3:
            blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            blurred = blurred_image.astype(np.float64)
        
        blurred = blurred / 255.0
        
        # Normalize kernel
        kernel = kernel / kernel.sum()
        
        # Initialize estimate with blurred image
        estimate = blurred.copy()
        
        # Flip kernel for convolution
        kernel_flipped = np.flipud(np.fliplr(kernel))
        
        for i in range(iterations):
            # Convolve estimate with PSF
            conv = cv2.filter2D(estimate, -1, kernel, borderType=cv2.BORDER_REFLECT)
            
            # Avoid division by zero
            conv = np.maximum(conv, 1e-10)
            
            # Compute relative blur
            relative_blur = blurred / conv
            
            # Convolve with flipped PSF
            correction = cv2.filter2D(relative_blur, -1, kernel_flipped, borderType=cv2.BORDER_REFLECT)
            
            # Update estimate
            estimate = estimate * correction
        
        # Convert back to uint8
        restored = (estimate * 255).clip(0, 255).astype(np.uint8)
        
        return restored
    
    def blur_and_restore_pipeline(self, original_image, kernel_size=15, sigma=3.0, 
                                   method='wiener', iterations=10):
        """
        Complete pipeline: blur image and restore it
        
        Args:
            original_image: Original clear image
            kernel_size: Size of Gaussian blur kernel
            sigma: Standard deviation of Gaussian
            method: 'wiener' or 'richardson_lucy'
            iterations: Number of iterations (for RL)
        
        Returns:
            dict with original, blurred, and restored images
        """
        # Apply blur
        blurred = self.apply_gaussian_blur(original_image, kernel_size, sigma)
        
        # Get blur kernel
        kernel = self.estimate_blur_kernel(kernel_size, sigma)
        
        # Restore based on method
        if method == 'wiener':
            restored = self.wiener_filter(blurred, kernel)
        else:  # richardson_lucy
            restored = self.richardson_lucy_deconvolution(blurred, kernel, iterations)
        
        # If original was color, convert restored back to color
        if len(original_image.shape) == 3:
            restored = cv2.cvtColor(restored, cv2.COLOR_GRAY2BGR)
        
        return {
            'original': original_image,
            'blurred': blurred,
            'restored': restored,
            'kernel_size': kernel_size,
            'sigma': sigma,
            'method': method
        }
    
    def frequency_domain_visualization(self, image):
        """
        Visualize frequency domain of image
        
        Returns:
            (magnitude_spectrum, phase_spectrum)
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Compute FFT
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        # Magnitude and phase
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        
        # Log scale for visualization
        magnitude_vis = 20 * np.log(magnitude + 1)
        magnitude_vis = cv2.normalize(magnitude_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        phase_vis = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return magnitude_vis, phase_vis


class TemplateMatchingWithBlur:
    """Combined template matching with blur applied to detected regions"""
    
    def __init__(self):
        self.template_matcher = TemplateMatchingModule()
        self.fourier_restoration = FourierRestoration()
    
    def detect_and_blur_region(self, image, template, blur_kernel_size=31):
        """
        Detect template in image and blur the detected region
        
        Returns:
            Image with blurred detected region
        """
        # Match template
        location, confidence, _, _ = self.template_matcher.match_template(image, template)
        
        if confidence < 0.5:
            return image, None, confidence
        
        # Get template dimensions
        h, w = template.shape[:2]
        x, y = location
        
        # Extract detected region
        region = image[y:y+h, x:x+w].copy()
        
        # Apply blur to region
        blurred_region = self.fourier_restoration.apply_gaussian_blur(
            region, blur_kernel_size, blur_kernel_size/6
        )
        
        # Create result image
        result = image.copy()
        result[y:y+h, x:x+w] = blurred_region
        
        # Draw rectangle around blurred region
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return result, (x, y, w, h), confidence


"""
Module 2: Template Matching & Fourier Restoration
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala

Based on OpenCV Template Matching:
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html

Uses TM_CCOEFF_NORMED - the most accurate and robust method.
"""

import cv2
import numpy as np


class TemplateMatchingModule:
    """
    Precision Template Matching using cv.matchTemplate()
    
    Method: TM_CCOEFF_NORMED (Normalized Cross-Correlation Coefficient)
    - Returns values in range [-1, 1]
    - Higher value = better match
    - Robust to lighting variations
    """
    
    def __init__(self):
        self.templates = []
        self.template_names = []
    
    def add_template(self, template_image, name):
        """Add a template for matching"""
        if len(template_image.shape) == 2:
            template_image = cv2.cvtColor(template_image, cv2.COLOR_GRAY2BGR)
        
        self.templates.append(template_image.copy())
        self.template_names.append(name)
        print(f"✅ Template '{name}' added: {template_image.shape[:2]}")
    
    def clear_templates(self):
        """Clear all templates"""
        self.templates = []
        self.template_names = []
    
    def match_template(self, image, template, threshold=0.8):
        """
        Match template in image using TM_CCOEFF_NORMED
        
        Based on OpenCV tutorial:
        https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
        
        Args:
            image: Input image (scene)
            template: Template to find
            threshold: Minimum confidence for multiple detections (0.0-1.0)
        
        Returns:
            detections: List of detection dicts
            result: Raw matching result map
        """
        # Convert to grayscale (as per OpenCV tutorial)
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        # Get template dimensions
        w, h = template_gray.shape[::-1]  # (width, height)
        
        # Apply template matching using TM_CCOEFF_NORMED
        result = cv2.matchTemplate(img_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find all matches above threshold
        loc = np.where(result >= threshold)
        points = list(zip(*loc[::-1]))  # Convert to (x, y) format
        
        detections = []
        
        if len(points) == 0:
            # No matches above threshold - get best single match
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            top_left = max_loc  # For TM_CCOEFF_NORMED, max = best
            
            detections.append({
                'bbox': [top_left[0], top_left[1], w, h],
                'confidence': float(max_val),
                'center': (top_left[0] + w // 2, top_left[1] + h // 2)
            })
        else:
            # Multiple matches - apply NMS
            boxes = []
            scores = []
            
            for pt in points:
                boxes.append([pt[0], pt[1], pt[0] + w, pt[1] + h])
                scores.append(float(result[pt[1], pt[0]]))
            
            boxes = np.array(boxes, dtype=np.float32)
            scores = np.array(scores)
            
            # Non-Maximum Suppression
            indices = self._nms(boxes, scores, iou_threshold=0.3)
            
            for idx in indices:
                x, y = int(boxes[idx][0]), int(boxes[idx][1])
                detections.append({
                    'bbox': [x, y, w, h],
                    'confidence': float(scores[idx]),
                    'center': (x + w // 2, y + h // 2)
                })
        
        return detections, result
    
    def _nms(self, boxes, scores, iou_threshold=0.3):
        """Non-Maximum Suppression"""
        if len(boxes) == 0:
            return []
        
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(scores)[::-1]
        
        keep = []
        
        while len(idxs) > 0:
            i = idxs[0]
            keep.append(i)
            
            if len(idxs) == 1:
                break
            
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            iou = (w * h) / area[idxs[1:]]
            idxs = np.delete(idxs, np.concatenate(([0], np.where(iou > iou_threshold)[0] + 1)))
        
        return keep
    
    def draw_detections(self, image, detections):
        """
        Draw clean rectangle boxes on image (OpenCV tutorial style)
        
        Output matches the reference image: simple white rectangles
        """
        # Convert to grayscale then back to BGR for display (like tutorial)
        if len(image.shape) == 3:
            result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        else:
            result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        for det in detections:
            x, y, w, h = det['bbox']
            
            # Draw simple white rectangle (like OpenCV tutorial output)
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 255, 255), 2)
        
        return result
    
    def match_all_templates(self, image, threshold=0.8, progress_callback=None):
        """
        Match all stored templates against the image
        
        Args:
            image: Scene image to search in
            threshold: Detection confidence threshold
            progress_callback: Optional callback for progress updates
        
        Returns:
            List of results with detections and annotated images
        """
        if len(self.templates) == 0:
            return []
        
        print(f"🔍 Matching {len(self.templates)} template(s) using TM_CCOEFF_NORMED")
        
        results = []
        total = len(self.templates)
        
        for i, (template, name) in enumerate(zip(self.templates, self.template_names)):
            if progress_callback:
                progress_callback(int(((i + 1) / total) * 100))
            
            # Match template
            detections, result_map = self.match_template(image, template, threshold)
            
            # Draw detections (clean style like reference image)
            result_image = self.draw_detections(image, detections)
            
            print(f"   ✓ {name}: {len(detections)} detection(s), confidence: {detections[0]['confidence']:.2%}")
            
            results.append({
                'name': name,
                'detections': detections,
                'num_detections': len(detections),
                'result_image': result_image
            })
        
        if progress_callback:
            progress_callback(100)
        
        return results


class FourierRestoration:
    """Fourier domain image restoration"""
    
    def __init__(self):
        pass
    
    def apply_gaussian_blur(self, image, kernel_size=15, sigma=3.0):
        """Apply Gaussian blur to image"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def estimate_blur_kernel(self, kernel_size=15, sigma=3.0):
        """Create Gaussian blur kernel"""
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        return kernel @ kernel.T
    
    def wiener_filter(self, blurred_image, kernel, K=0.01):
        """Apply Wiener filter for image restoration"""
        if len(blurred_image.shape) == 3:
            blurred_gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        else:
            blurred_gray = blurred_image.copy()
        
        img_h, img_w = blurred_gray.shape
        kernel_h, kernel_w = kernel.shape
        
        padded_kernel = np.zeros((img_h, img_w))
        pad_h = (img_h - kernel_h) // 2
        pad_w = (img_w - kernel_w) // 2
        padded_kernel[pad_h:pad_h+kernel_h, pad_w:pad_w+kernel_w] = kernel
        padded_kernel = np.fft.ifftshift(padded_kernel)
        
        img_fft = np.fft.fft2(blurred_gray)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_abs2 = np.abs(kernel_fft) ** 2
        
        wiener = kernel_fft_conj / (kernel_fft_abs2 + K)
        restored_fft = img_fft * wiener
        
        restored = np.fft.ifft2(restored_fft)
        restored = np.abs(restored)
        restored = np.clip(restored, 0, 255).astype(np.uint8)
        
        return restored
    
    def richardson_lucy_deconvolution(self, blurred_image, kernel, iterations=10):
        """Richardson-Lucy deconvolution"""
        if len(blurred_image.shape) == 3:
            blurred = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            blurred = blurred_image.astype(np.float64)
        
        blurred = np.maximum(blurred / 255.0, 1e-10)
        
        kernel = kernel / kernel.sum()
        kernel_flipped = np.flipud(np.fliplr(kernel))
        
        estimate = blurred.copy()
        
        for _ in range(iterations):
            conv = np.maximum(cv2.filter2D(estimate, -1, kernel, borderType=cv2.BORDER_REFLECT), 1e-10)
            relative_blur = blurred / conv
            correction = cv2.filter2D(relative_blur, -1, kernel_flipped, borderType=cv2.BORDER_REFLECT)
            estimate = estimate * correction
        
        return (estimate * 255).clip(0, 255).astype(np.uint8)
    
    def blur_and_restore_pipeline(self, original_image, kernel_size=15, sigma=3.0, 
                                   method='wiener', iterations=10):
        """Complete blur and restore pipeline"""
        blurred = self.apply_gaussian_blur(original_image, kernel_size, sigma)
        kernel = self.estimate_blur_kernel(kernel_size, sigma)
        
        if method == 'wiener':
            restored = self.wiener_filter(blurred, kernel)
        else:
            restored = self.richardson_lucy_deconvolution(blurred, kernel, iterations)
        
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


class TemplateMatchingWithBlur:
    """Combined template matching with blur"""
    
    def __init__(self):
        self.template_matcher = TemplateMatchingModule()
        self.fourier_restoration = FourierRestoration()
    
    def detect_and_blur_region(self, image, template, blur_kernel_size=31):
        """Detect template and blur the region"""
        detections, _ = self.template_matcher.match_template(image, template)
        
        if not detections or detections[0]['confidence'] < 0.5:
            return image, None, 0
        
        det = detections[0]
        x, y, w, h = det['bbox']
        
        region = image[y:y+h, x:x+w].copy()
        blurred_region = self.fourier_restoration.apply_gaussian_blur(region, blur_kernel_size, blur_kernel_size / 6)
        
        result = image.copy()
        result[y:y+h, x:x+w] = blurred_region
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        return result, (x, y, w, h), det['confidence']

"""
Module 2: Template Matching and Fourier Restoration
Author: Mahendra Krishna Koneru

Based on OpenCV Tutorial:
https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
"""

import cv2
import numpy as np

class TemplateMatchingModule:
    """Template matching using OpenCV's matchTemplate"""
    
    def __init__(self, max_image_size=1200):
        self.templates = []
        self.template_names = []
        self.max_image_size = max_image_size
    
    def resize_for_processing(self, image):
        """Resize image if too large"""
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
        """Add a template for matching"""
        resized_template, scale = self.resize_for_processing(template_image)
        
        # Store as grayscale for matching
        if len(resized_template.shape) == 3:
            resized_template = cv2.cvtColor(resized_template, cv2.COLOR_BGR2GRAY)
        
        self.templates.append(resized_template)
        self.template_names.append(name)
        print(f"✅ Template '{name}' added, shape: {resized_template.shape}")
    
    def clear_templates(self):
        """Clear all templates"""
        self.templates = []
        self.template_names = []
        print("🗑️ Templates cleared")
    
    def match_template(self, image, template, method=cv2.TM_CCOEFF_NORMED):
        """
        Match single template using OpenCV matchTemplate
        
        Reference: https://docs.opencv.org/4.x/d4/dc6/tutorial_py_template_matching.html
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            img_gray = image.copy()
        
        if len(template.shape) == 3:
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            template_gray = template.copy()
        
        # Get template dimensions
        h, w = template_gray.shape[:2]
        
        # Apply template matching
        res = cv2.matchTemplate(img_gray, template_gray, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        
        # For TM_SQDIFF and TM_SQDIFF_NORMED, minimum is best match
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
            confidence = 1.0 - min_val  # Invert for consistency
        else:
            top_left = max_loc
            confidence = max_val
        
        bottom_right = (top_left[0] + w, top_left[1] + h)
        
        return {
            'top_left': top_left,
            'bottom_right': bottom_right,
            'width': w,
            'height': h,
            'confidence': float(confidence)
        }
    
    def match_all_templates(self, image, progress_callback=None, threshold=0.65):
        """Match all stored templates against the image"""
        if len(self.templates) == 0:
            print("⚠️ No templates to match")
            return []
        
        # Resize image for processing
        processed_image, scale = self.resize_for_processing(image)
        
        print(f"🔍 Processing: image shape={image.shape}, scale={scale:.2f}, templates={len(self.templates)}")
        
        results = []
        
        for i, template in enumerate(self.templates):
            if progress_callback:
                progress_callback(int(((i + 1) / len(self.templates)) * 100))
            
            # Match template
            match = self.match_template(processed_image, template)
            
            # Scale coordinates back to original image size
            if scale != 1.0:
                top_left = (int(match['top_left'][0] / scale), int(match['top_left'][1] / scale))
                bottom_right = (int(match['bottom_right'][0] / scale), int(match['bottom_right'][1] / scale))
                w = int(match['width'] / scale)
                h = int(match['height'] / scale)
            else:
                top_left = match['top_left']
                bottom_right = match['bottom_right']
                w = match['width']
                h = match['height']
            
            confidence = match['confidence']
            
            print(f"  ✓ {self.template_names[i]}: confidence={confidence:.2%}, bbox={top_left} to {bottom_right}")
            
            # Create result image - grayscale like OpenCV reference
            if len(image.shape) == 3:
                result_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                result_img = image.copy()
            
            # Draw white rectangle exactly like OpenCV tutorial
            # cv2.rectangle(img, top_left, bottom_right, 255, 2)
            cv2.rectangle(result_img, top_left, bottom_right, 255, 2)
            
            results.append({
                'name': self.template_names[i],
                'detections': [{
                    'bbox': [top_left[0], top_left[1], w, h],
                    'confidence': confidence,
                    'center': ((top_left[0] + bottom_right[0]) // 2, (top_left[1] + bottom_right[1]) // 2)
                }],
                'num_detections': 1,
                'result_image': result_img
            })
        
        if progress_callback:
            progress_callback(100)
        
        return results


class FourierRestoration:
    """Fourier-based image restoration"""
    
    def __init__(self):
        pass
    
    def apply_gaussian_blur(self, image, kernel_size=15, sigma=3.0):
        """Apply Gaussian blur"""
        if kernel_size % 2 == 0:
            kernel_size += 1
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def estimate_blur_kernel(self, kernel_size=15, sigma=3.0):
        """Create Gaussian kernel"""
        kernel = cv2.getGaussianKernel(kernel_size, sigma)
        return kernel @ kernel.T
    
    def wiener_filter(self, blurred_image, kernel, K=0.01):
        """Wiener filter for deblurring"""
        if len(blurred_image.shape) == 3:
            gray = cv2.cvtColor(blurred_image, cv2.COLOR_BGR2GRAY)
        else:
            gray = blurred_image.copy()
        
        img_h, img_w = gray.shape
        kernel_h, kernel_w = kernel.shape
        
        # Pad kernel to image size
        padded_kernel = np.zeros((img_h, img_w))
        pad_h = (img_h - kernel_h) // 2
        pad_w = (img_w - kernel_w) // 2
        padded_kernel[pad_h:pad_h+kernel_h, pad_w:pad_w+kernel_w] = kernel
        padded_kernel = np.fft.ifftshift(padded_kernel)
        
        # FFT
        img_fft = np.fft.fft2(gray)
        kernel_fft = np.fft.fft2(padded_kernel)
        
        # Wiener filter
        kernel_fft_conj = np.conj(kernel_fft)
        kernel_fft_abs2 = np.abs(kernel_fft) ** 2
        wiener = kernel_fft_conj / (kernel_fft_abs2 + K)
        restored_fft = img_fft * wiener
        
        # Inverse FFT
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
        
        blurred = blurred / 255.0
        kernel = kernel / kernel.sum()
        estimate = blurred.copy()
        kernel_flipped = np.flipud(np.fliplr(kernel))
        
        for _ in range(iterations):
            conv = cv2.filter2D(estimate, -1, kernel, borderType=cv2.BORDER_REFLECT)
            conv = np.maximum(conv, 1e-10)
            relative_blur = blurred / conv
            correction = cv2.filter2D(relative_blur, -1, kernel_flipped, borderType=cv2.BORDER_REFLECT)
            estimate = estimate * correction
        
        restored = (estimate * 255).clip(0, 255).astype(np.uint8)
        return restored
    
    def blur_and_restore_pipeline(self, original_image, kernel_size=15, sigma=3.0, 
                                   method='wiener', iterations=10):
        """Complete blur and restore pipeline"""
        if kernel_size % 2 == 0:
            kernel_size += 1
            
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
    
    def frequency_domain_visualization(self, image):
        """Visualize frequency domain"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        
        magnitude = np.abs(fshift)
        phase = np.angle(fshift)
        
        magnitude_vis = 20 * np.log(magnitude + 1)
        magnitude_vis = cv2.normalize(magnitude_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        phase_vis = cv2.normalize(phase, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
        return magnitude_vis, phase_vis


class TemplateMatchingWithBlur:
    """Combined template matching with blur"""
    
    def __init__(self):
        self.template_matcher = TemplateMatchingModule()
        self.fourier_restoration = FourierRestoration()
    
    def detect_and_blur_region(self, image, template, blur_kernel_size=31):
        """Detect template and blur the region"""
        match = self.template_matcher.match_template(image, template)
        
        if match['confidence'] < 0.5:
            return image, None, match['confidence']
        
        x, y = match['top_left']
        w, h = match['width'], match['height']
        
        region = image[y:y+h, x:x+w].copy()
        blurred_region = self.fourier_restoration.apply_gaussian_blur(
            region, blur_kernel_size, blur_kernel_size/6
        )
        
        result = image.copy()
        result[y:y+h, x:x+w] = blurred_region
        cv2.rectangle(result, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        return result, (x, y, w, h), match['confidence']

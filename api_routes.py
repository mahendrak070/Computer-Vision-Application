"""
API Routes for CV Modules
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

from flask import request, jsonify, send_file, session
import cv2
import numpy as np
import base64
import io
import os
import gc
from datetime import datetime
from modules.module1_dimension import DimensionEstimator
from modules.module2_template import TemplateMatchingModule, FourierRestoration, TemplateMatchingWithBlur
from modules.module3_features import (GradientComputation, EdgeDetection, CornerDetection, Segmentation,
                                       LaplacianOfGaussian, BoundaryDetection, ArucoSegmentation)
from modules.module4_sift_stitching import SIFTFromScratch, ImageStitching
from modules.module5_tracking import MarkerBasedTracker, MarkerlessTracker, ColorBasedTracker
# Lazy import heavy Mediapipe modules
pose_estimator = None
hand_tracker = None
stereo_calib = None

def get_pose_estimator():
    global pose_estimator
    if pose_estimator is None:
        from modules.module7_pose_hand import PoseEstimation
        pose_estimator = PoseEstimation()
    return pose_estimator

def get_hand_tracker():
    global hand_tracker
    if hand_tracker is None:
        from modules.module7_pose_hand import HandTracking
        hand_tracker = HandTracking()
    return hand_tracker

def get_stereo_calib():
    global stereo_calib
    if stereo_calib is None:
        from modules.module7_pose_hand import StereoCalibration
        stereo_calib = StereoCalibration()
    return stereo_calib

# Initialize lightweight module instances
dimension_estimator = DimensionEstimator()
template_matcher = TemplateMatchingModule()
fourier_restoration = FourierRestoration()
template_blur = TemplateMatchingWithBlur()
gradient_comp = GradientComputation()
log_detector = LaplacianOfGaussian()
edge_detector_simple = EdgeDetection()
corner_detector_harris = CornerDetection()
boundary_detector = BoundaryDetection()
aruco_segmentation = ArucoSegmentation()
segmentation = Segmentation()
image_stitcher = ImageStitching()

# Tracking instances (stored per session)
trackers = {}

def decode_image(image_data):
    """Decode base64 image"""
    if ',' in image_data:
        image_data = image_data.split(',')[1]
    image_bytes = base64.b64decode(image_data)
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return image

def encode_image(image, quality=85):
    """Encode image to base64 with compression"""
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    _, buffer = cv2.imencode('.jpg', image, encode_param)
    img_base64 = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{img_base64}"

def resize_for_processing(image, max_width=1280):
    """Resize image for faster processing while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if w > max_width:
        scale = max_width / w
        new_w = max_width
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def register_api_routes(app):
    """Register all API routes to Flask app"""
    
    # ==================== MODULE 1 APIs ====================
    
    @app.route('/api/module1/undistort', methods=['POST'])
    def api_module1_undistort():
        """Undistort image using camera calibration"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            
            image = decode_image(image_data)
            undistorted_image = dimension_estimator.undistort_image(image)
            
            return jsonify({
                'success': True,
                'undistorted_image': encode_image(undistorted_image, quality=90)
            })
        except Exception as e:
            print(f"Module 1 undistort error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module1/estimate', methods=['POST'])
    def api_module1_estimate():
        """Estimate dimensions from points using camera calibration"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            points = data.get('points')
            distance_inches = float(data.get('distance', 5))
            image_width = int(data.get('image_width', 800))
            image_height = int(data.get('image_height', 600))
            
            image_shape = (image_height, image_width)
            
            results = dimension_estimator.estimate_from_points(
                points, image_shape, distance_inches
            )
            
            return jsonify({
                'success': True,
                'results': results
            })
        except Exception as e:
            print(f"Module 1 error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    # ==================== MODULE 2 APIs ====================
    
    @app.route('/api/module2/add_template', methods=['POST'])
    def api_module2_add_template():
        """Add template for matching"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            name = data.get('name', f'template_{len(template_matcher.templates)}')
            
            template = decode_image(image_data)
            template_matcher.add_template(template, name)
            
            return jsonify({
                'success': True,
                'message': f'Template "{name}" added',
                'total_templates': len(template_matcher.templates)
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module2/match_templates', methods=['POST'])
    def api_module2_match_templates():
        """Match all templates in image - OPTIMIZED with bounding boxes"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            
            image = decode_image(image_data)
            results = template_matcher.match_all_templates(image)
            
            # Encode result images
            for result in results:
                result['result_image'] = encode_image(result['result_image'], quality=90)
                # Remove match_heatmap from response (too large)
                if 'match_heatmap' in result:
                    del result['match_heatmap']
            
            return jsonify({
                'success': True,
                'matches': results
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module2/blur_restore', methods=['POST'])
    def api_module2_blur_restore():
        """Blur and restore image pipeline"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            kernel_size = int(data.get('kernel_size', 15))
            sigma = float(data.get('sigma', 3.0))
            method = data.get('method', 'wiener')
            
            image = decode_image(image_data)
            results = fourier_restoration.blur_and_restore_pipeline(
                image, kernel_size, sigma, method
            )
            
            return jsonify({
                'success': True,
                'original': encode_image(results['original']),
                'blurred': encode_image(results['blurred']),
                'restored': encode_image(results['restored']),
                'kernel_size': kernel_size,
                'sigma': sigma,
                'method': results['method']
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    # ==================== MODULE 3 APIs ====================
    
    @app.route('/api/module3/process', methods=['POST'])
    def api_module3_process():
        """Comprehensive feature detection processing"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            mode = data.get('mode', 'gradient_mag')
            params = data.get('params', {})
            
            app.logger.info(f"Module 3 processing: mode={mode}, params={params}")
            
            image = decode_image(image_data)
            if image is None:
                return jsonify({'success': False, 'message': 'Failed to decode image'}), 400
            
            app.logger.info(f"Image decoded: shape={image.shape}, dtype={image.dtype}")
            
            result_image = None
            count = None
            message = "Processing complete"
            
            # Process based on mode
            if mode == 'gradient_mag':
                app.logger.info("Computing gradient magnitude...")
                result_image = gradient_comp.compute_gradient_magnitude(image)
                message = "Gradient magnitude computed"
                
            elif mode == 'gradient_angle':
                app.logger.info("Computing gradient angle...")
                result_image = gradient_comp.compute_gradient_angle(image)
                message = "Gradient angle visualized in HSV"
                
            elif mode == 'log':
                app.logger.info("Applying Laplacian of Gaussian...")
                sigma = params.get('sigma', 1.0)
                log_detector.gauss_sigma = sigma
                result_image = log_detector.detect(image)
                message = "Laplacian of Gaussian applied"
                
            elif mode == 'edges':
                app.logger.info(f"Detecting edges with params: {params}")
                sigma = params.get('sigma', 1.0)
                low = params.get('low_threshold', 15)
                high = params.get('high_threshold', 40)
                edge_det = EdgeDetection(sigma=sigma, low_threshold=low, high_threshold=high)
                result_image, count = edge_det.detect(image)
                message = f"Edge detection complete ({count} edge pixels)"
                app.logger.info(f"Edges detected: {count} pixels")
                
            elif mode == 'corners':
                app.logger.info(f"Detecting corners with params: {params}")
                k = params.get('k', 0.04)
                threshold = params.get('threshold', 70)
                window_size = params.get('window_size', 5)
                corner_det = CornerDetection(k=k, threshold=threshold, window_size=window_size)
                result_image, count = corner_det.detect(image)
                message = f"Harris corner detection ({count} corners found)"
                app.logger.info(f"Corners detected: {count}")
                
            elif mode == 'boundary':
                app.logger.info("Detecting boundaries...")
                result_image, count = boundary_detector.detect(image)
                message = f"Boundary detection complete ({count} objects found)"
                app.logger.info(f"Boundaries detected: {count} objects")
                
            elif mode == 'aruco':
                app.logger.info("Detecting ArUco markers...")
                result_image, count = aruco_segmentation.segment(image)
                if count > 0:
                    message = f"ArUco segmentation ({count} markers detected)"
                else:
                    message = "No ArUco markers detected - Print markers from https://chev.me/arucogen/"
                app.logger.info(f"ArUco markers detected: {count}")
            
            else:
                return jsonify({'success': False, 'message': 'Invalid mode'}), 400
            
            if result_image is None:
                return jsonify({'success': False, 'message': 'Processing returned no result'}), 500
            
            app.logger.info(f"Result image: shape={result_image.shape}, dtype={result_image.dtype}")
            
            encoded = encode_image(result_image, quality=90)
            
            return jsonify({
                'success': True,
                'result_image': encoded,
                'count': count,
                'message': message
            })
            
        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            app.logger.error(f"Module 3 processing error: {str(e)}\n{error_details}")
            return jsonify({'success': False, 'message': f"Processing error: {str(e)}"}), 500
    
    # ==================== MODULE 4 APIs ====================
    
    @app.route('/api/module4/stitch', methods=['POST'])
    def api_module4_stitch():
        """Stitch multiple images - OPTIMIZED with progress tracking"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            images_data = data.get('images')  # List of base64 images
            use_custom_sift = data.get('use_custom_sift', False)
            
            if not images_data or len(images_data) < 2:
                return jsonify({'success': False, 'message': 'Need at least 2 images'}), 400
            
            print(f"[Module 4] Starting stitch with {len(images_data)} images")
            
            # Decode images and resize if too large (memory optimization)
            images = []
            for i, img_data in enumerate(images_data):
                try:
                    img = decode_image(img_data)
                    if img is None:
                        return jsonify({'success': False, 'message': f'Failed to decode image {i+1}'}), 400
                    
                    # Resize if too large to prevent OOM (Railway has limited RAM)
                    h, w = img.shape[:2]
                    max_dim = 1920
                    if max(h, w) > max_dim:
                        scale = max_dim / max(h, w)
                        new_w, new_h = int(w * scale), int(h * scale)
                        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                        print(f"[Module 4] Resized image {i+1}: {(h,w)} -> {img.shape}")
                    else:
                        print(f"[Module 4] Image {i+1}: {img.shape}")
                    
                    images.append(img)
                except Exception as e:
                    print(f"[Module 4] Error decoding image {i+1}: {str(e)}")
                    return jsonify({'success': False, 'message': f'Invalid image data at position {i+1}'}), 400
            
            # Stitch images
            print(f"[Module 4] Starting stitching process...")
            try:
                stitched = image_stitcher.stitch_images(images, use_custom_sift)
            except Exception as stitch_err:
                print(f"[Module 4] Stitching exception: {str(stitch_err)}")
                import traceback
                traceback.print_exc()
                del images
                gc.collect()
                return jsonify({'success': False, 'message': f'Stitching failed: {str(stitch_err)}'}), 500
            
            if stitched is None:
                print(f"[Module 4] Stitching returned None")
                del images
                gc.collect()
                return jsonify({'success': False, 'message': 'Stitching failed - insufficient feature matches'}), 400
            
            print(f"[Module 4] Stitching successful: {stitched.shape}")
            
            # Get shape before encoding
            result_shape = stitched.shape[:2]
            
            # Encode result
            encoded = encode_image(stitched, quality=90)
            
            # Free memory
            del images
            del stitched
            gc.collect()
            
            return jsonify({
                'success': True,
                'stitched': encoded,
                'shape': result_shape
            })
        except Exception as e:
            print(f"[Module 4] Exception: {str(e)}")
            import traceback
            traceback.print_exc()
            return jsonify({'success': False, 'message': f'Server error: {str(e)}'}), 500
    
    @app.route('/api/module4/sift_detect', methods=['POST'])
    def api_module4_sift_detect():
        """SIFT keypoint detection"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            use_custom = data.get('use_custom', False)
            
            image = decode_image(image_data)
            
            if use_custom:
                sift = SIFTFromScratch()
                keypoints, descriptors = sift.detect_and_compute(image)
                
                # Draw keypoints
                result = image.copy()
                for kp in keypoints:
                    x, y = int(kp['x']), int(kp['y'])
                    size = int(kp['size'])
                    cv2.circle(result, (x, y), size, (0, 255, 0), 2)
                
                num_keypoints = len(keypoints)
            else:
                sift = cv2.SIFT_create()
                keypoints, descriptors = sift.detectAndCompute(image, None)
                result = cv2.drawKeypoints(image, keypoints, None, 
                                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
                num_keypoints = len(keypoints)
            
            return jsonify({
                'success': True,
                'image': encode_image(result),
                'num_keypoints': num_keypoints,
                'method': 'custom' if use_custom else 'opencv'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    # ==================== MODULE 5-6 APIs ====================
    
    @app.route('/api/module5/init_tracker', methods=['POST'])
    def api_module5_init_tracker():
        """Initialize tracker"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            tracker_type = data.get('tracker_type', 'aruco')
            image_data = data.get('image')
            bbox = data.get('bbox')  # [x, y, w, h] for markerless
            
            user_id = session['user_id']
            
            if tracker_type == 'aruco' or tracker_type == 'qr':
                trackers[user_id] = MarkerBasedTracker(tracker_type)
            elif tracker_type == 'markerless':
                tracker = MarkerlessTracker()
                if bbox:
                    image = decode_image(image_data)
                    tracker.initialize_tracking(image, bbox)
                trackers[user_id] = tracker
            elif tracker_type == 'color':
                tracker = ColorBasedTracker()
                if bbox:
                    image = decode_image(image_data)
                    tracker.initialize_tracking(image, bbox)
                trackers[user_id] = tracker
            
            return jsonify({
                'success': True,
                'message': f'{tracker_type} tracker initialized'
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module5/track', methods=['POST'])
    def api_module5_track():
        """Track in frame"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            user_id = session['user_id']
            
            if user_id not in trackers:
                return jsonify({'success': False, 'message': 'Tracker not initialized'}), 400
            
            data = request.json
            image_data = data.get('image')
            
            image = decode_image(image_data)
            tracker = trackers[user_id]
            
            result_frame, tracking_data = tracker.track(image)
            
            return jsonify({
                'success': True,
                'image': encode_image(result_frame),
                'tracking_data': tracking_data
            })
        except Exception as e:
            return jsonify({'success': False, 'message': str(e)}), 500
    
    # ==================== MODULE 7 APIs ====================
    
    @app.route('/api/module7/pose', methods=['POST'])
    def api_module7_pose():
        """Pose estimation"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            
            image = decode_image(image_data)
            
            # Resize for faster processing
            image = resize_for_processing(image, max_width=960)
            
            annotated, landmarks, pose_data = get_pose_estimator().process_frame(image)
            
            # Add summary statistics
            summary = {
                'landmark_count': 33 if landmarks else 0,
                'detected': landmarks is not None
            }
            
            return jsonify({
                'success': True,
                'image': encode_image(annotated, quality=80),
                'pose_detected': landmarks is not None,
                'pose_data': pose_data,
                'summary': summary
            })
        except Exception as e:
            print(f"Pose error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module7/hand', methods=['POST'])
    def api_module7_hand():
        """Hand tracking"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            data = request.json
            image_data = data.get('image')
            
            image = decode_image(image_data)
            
            # Resize for faster processing
            image = resize_for_processing(image, max_width=960)
            
            annotated, landmarks, hand_data = get_hand_tracker().process_frame(image)
            
            # Detect gestures
            gesture = None
            num_hands = 0
            if landmarks and len(landmarks) > 0:
                gesture = get_hand_tracker().detect_gestures(landmarks[0])
                num_hands = len(landmarks)
            
            # Add summary statistics
            summary = {
                'num_hands': num_hands,
                'landmark_count': num_hands * 21,
                'detected': landmarks is not None and len(landmarks) > 0
            }
            
            return jsonify({
                'success': True,
                'image': encode_image(annotated, quality=80),
                'hands_detected': landmarks is not None and len(landmarks) > 0,
                'hand_data': hand_data,
                'gesture': gesture,
                'summary': summary
            })
        except Exception as e:
            print(f"Hand tracking error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module7/save_pose_csv', methods=['POST'])
    def api_module7_save_pose_csv():
        """Save pose data to CSV"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            filename = f'pose_data_{session["user_id"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            filepath = os.path.join('uploads', filename)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            
            success = get_pose_estimator().save_to_csv(filepath)
            
            if success and os.path.exists(filepath):
                return send_file(
                    filepath, 
                    mimetype='text/csv',
                    as_attachment=True, 
                    download_name=filename
                )
            else:
                return jsonify({'success': False, 'message': 'No pose data to save. Start tracking first!'}), 400
        except Exception as e:
            app.logger.error(f"CSV save error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module7/save_hand_csv', methods=['POST'])
    def api_module7_save_hand_csv():
        """Save hand data to CSV"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            filename = f'hand_data_{session["user_id"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            filepath = os.path.join('uploads', filename)
            
            # Ensure uploads directory exists
            os.makedirs('uploads', exist_ok=True)
            
            success = get_hand_tracker().save_to_csv(filepath)
            
            if success and os.path.exists(filepath):
                return send_file(
                    filepath, 
                    mimetype='text/csv',
                    as_attachment=True, 
                    download_name=filename
                )
            else:
                return jsonify({'success': False, 'message': 'No hand data to save. Start tracking first!'}), 400
        except Exception as e:
            app.logger.error(f"CSV save error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500
    
    @app.route('/api/module7/reset_data', methods=['POST'])
    def api_module7_reset_data():
        """Reset pose and hand tracking data"""
        if 'user_id' not in session:
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        
        try:
            get_pose_estimator().reset_data()
            get_hand_tracker().reset_data()
            return jsonify({'success': True, 'message': 'Data reset successfully'})
        except Exception as e:
            app.logger.error(f"Reset data error: {str(e)}")
            return jsonify({'success': False, 'message': str(e)}), 500


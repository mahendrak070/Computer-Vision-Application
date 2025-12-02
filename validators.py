"""
Input Validation and Sanitization
Author: Mahendra Krishna Koneru
"""

import re
import base64
from functools import wraps
from flask import request, jsonify, session
import numpy as np
import cv2

def validate_email(email):
    """Validate email format"""
    if not email or not isinstance(email, str):
        return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email.strip()) is not None

def validate_username(username):
    """Validate username format"""
    if not username or not isinstance(username, str):
        return False
    username = username.strip()
    if len(username) < 3 or len(username) > 50:
        return False
    pattern = r'^[a-zA-Z0-9_-]+$'
    return re.match(pattern, username) is not None

def validate_base64_image(image_data):
    """Validate base64 image data"""
    try:
        if not image_data or not isinstance(image_data, str):
            return False, "Invalid image data"
        
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        decoded = base64.b64decode(image_data)
        
        if len(decoded) == 0:
            return False, "Empty image data"
        
        if len(decoded) > 50 * 1024 * 1024:
            return False, "Image too large (max 50MB)"
        
        nparr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return False, "Invalid image format"
        
        if img.shape[0] < 10 or img.shape[1] < 10:
            return False, "Image too small"
        
        if img.shape[0] > 10000 or img.shape[1] > 10000:
            return False, "Image dimensions too large"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Image validation error: {str(e)}"

def validate_number(value, min_val=None, max_val=None):
    """Validate numeric value"""
    try:
        num = float(value)
        if min_val is not None and num < min_val:
            return False, f"Value must be at least {min_val}"
        if max_val is not None and num > max_val:
            return False, f"Value must be at most {max_val}"
        return True, num
    except (ValueError, TypeError):
        return False, "Invalid number format"

def sanitize_string(text, max_length=1000):
    """Sanitize string input"""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text[:max_length]
    text = re.sub(r'[<>\"\'&]', '', text)
    return text

def require_auth(f):
    """Decorator to require authentication"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            if request.path.startswith('/api/'):
                return jsonify({
                    'success': False,
                    'error': 'Unauthorized',
                    'message': 'Authentication required'
                }), 401
            return jsonify({'success': False, 'message': 'Not authenticated'}), 401
        return f(*args, **kwargs)
    return decorated_function

def validate_request_json(*required_fields):
    """Decorator to validate required JSON fields"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            if not request.is_json:
                return jsonify({
                    'success': False,
                    'error': 'Bad Request',
                    'message': 'Request must be JSON'
                }), 400
            
            data = request.get_json()
            if not data:
                return jsonify({
                    'success': False,
                    'error': 'Bad Request',
                    'message': 'Empty request body'
                }), 400
            
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return jsonify({
                    'success': False,
                    'error': 'Bad Request',
                    'message': f'Missing required fields: {", ".join(missing_fields)}'
                }), 400
            
            return f(*args, **kwargs)
        return decorated_function
    return decorator

def validate_image_list(images_data, min_count=1, max_count=10):
    """Validate list of base64 images"""
    if not isinstance(images_data, list):
        return False, "Images must be a list"
    
    if len(images_data) < min_count:
        return False, f"Need at least {min_count} image(s)"
    
    if len(images_data) > max_count:
        return False, f"Maximum {max_count} images allowed"
    
    for i, img_data in enumerate(images_data):
        valid, msg = validate_base64_image(img_data)
        if not valid:
            return False, f"Image {i+1}: {msg}"
    
    return True, "Valid"

def validate_points(points):
    """Validate list of coordinate points"""
    if not isinstance(points, list):
        return False, "Points must be a list"
    
    if len(points) < 2:
        return False, "Need at least 2 points"
    
    for i, point in enumerate(points):
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            return False, f"Point {i+1} must be [x, y] coordinates"
        
        try:
            x, y = float(point[0]), float(point[1])
            if x < 0 or y < 0 or x > 10000 or y > 10000:
                return False, f"Point {i+1} coordinates out of range"
        except (ValueError, TypeError):
            return False, f"Point {i+1} has invalid coordinates"
    
    return True, "Valid"

def validate_bbox(bbox):
    """Validate bounding box [x, y, w, h]"""
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return False, "Bounding box must be [x, y, width, height]"
    
    try:
        x, y, w, h = [float(v) for v in bbox]
        if w <= 0 or h <= 0:
            return False, "Width and height must be positive"
        if x < 0 or y < 0:
            return False, "Coordinates must be non-negative"
        return True, [x, y, w, h]
    except (ValueError, TypeError):
        return False, "Invalid bounding box values"




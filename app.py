"""
Flask Backend for Computer Vision Project Website
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

from flask import Flask, render_template, request, jsonify, session, send_from_directory, abort
from flask_cors import CORS
import cv2
import numpy as np
import face_recognition
import os
import json
import sqlite3
from datetime import datetime, timedelta
import base64
from werkzeug.utils import secure_filename
import secrets
import logging
from logging.handlers import RotatingFileHandler

from config import config
from error_handlers import register_error_handlers
from validators import (validate_email, validate_username, validate_base64_image,
                        require_auth, validate_request_json)

env = os.environ.get('FLASK_ENV', 'development')
app = Flask(__name__)
app.config.from_object(config[env])
app.secret_key = app.config['SECRET_KEY']

CORS(app, supports_credentials=True, origins="*")

if not app.config['DEBUG']:
    if not os.path.exists('logs'):
        os.mkdir('logs')
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240000, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Computer Vision App startup')

register_error_handlers(app)

# Import and register API routes
from api_routes import register_api_routes
register_api_routes(app)

# Ensure directories exist
os.makedirs('database', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs('face_encodings', exist_ok=True)
os.makedirs('templates', exist_ok=True)

# Database initialization with error handling
def init_db():
    """Initialize database with proper error handling"""
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (id INTEGER PRIMARY KEY AUTOINCREMENT,
                      username TEXT UNIQUE NOT NULL,
                      email TEXT UNIQUE NOT NULL,
                      face_encoding BLOB NOT NULL,
                      created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
        conn.commit()
        conn.close()
        app.logger.info('Database initialized successfully')
    except Exception as e:
        app.logger.error(f'Database initialization error: {str(e)}')
        raise

def get_db_connection():
    """Get database connection with error handling"""
    try:
        conn = sqlite3.connect(app.config['DATABASE_PATH'])
        conn.row_factory = sqlite3.Row
        return conn
    except Exception as e:
        app.logger.error(f'Database connection error: {str(e)}')
        raise

init_db()

# ==================== AUTHENTICATION ROUTES ====================

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')

@app.route('/register', methods=['GET'])
def register_page():
    """Registration page"""
    return render_template('register.html')

@app.route('/login', methods=['GET'])
def login_page():
    """Login page"""
    return render_template('login.html')

@app.route('/dev-login')
def dev_login():
    """Quick development login bypass - for testing only"""
    session['user_id'] = 'dev_user'
    session['username'] = 'Developer'
    session.permanent = True
    return '''
    <html>
    <head>
        <meta http-equiv="refresh" content="1;url=/dashboard">
        <style>
            body { font-family: Arial; text-align: center; padding: 100px; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; }
        </style>
    </head>
    <body>
        <h1>✅ Development Login Successful!</h1>
        <p>Redirecting to dashboard...</p>
        <p><a href="/dashboard" style="color: white;">Click here if not redirected</a></p>
    </body>
    </html>
    '''

@app.route('/dashboard')
def dashboard():
    """Main dashboard - requires authentication"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('dashboard.html')

@app.route('/api/register', methods=['POST'])
@validate_request_json('username', 'email', 'image')
def api_register():
    """Register new user with face encoding - PRODUCTION READY"""
    try:
        data = request.json
        username = data.get('username', '').strip()
        email = data.get('email', '').strip()
        image_data = data.get('image')
        
        if not validate_username(username):
            return jsonify({
                'success': False, 
                'message': 'Invalid username. Use 3-50 characters, letters, numbers, underscore, hyphen only.'
            }), 400
        
        if not validate_email(email):
            return jsonify({
                'success': False, 
                'message': 'Invalid email format.'
            }), 400
        
        valid, msg = validate_base64_image(image_data)
        if not valid:
            return jsonify({
                'success': False, 
                'message': f'Invalid image: {msg}'
            }), 400
        
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False, 
                'message': 'Could not process image'
            }), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            return jsonify({
                'success': False, 
                'message': 'No face detected. Please ensure your face is clearly visible.'
            }), 400
        
        if len(face_locations) > 1:
            return jsonify({
                'success': False, 
                'message': 'Multiple faces detected. Please ensure only one face is visible.'
            }), 400
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            return jsonify({
                'success': False, 
                'message': 'Could not generate face encoding. Please try again.'
            }), 400
        
        face_encoding = face_encodings[0]
        
        conn = get_db_connection()
        c = conn.cursor()
        
        try:
            c.execute('INSERT INTO users (username, email, face_encoding) VALUES (?, ?, ?)',
                     (username, email, face_encoding.tobytes()))
            conn.commit()
            user_id = c.lastrowid
            
            os.makedirs(app.config['FACE_ENCODINGS_PATH'], exist_ok=True)
            cv2.imwrite(f"{app.config['FACE_ENCODINGS_PATH']}/{user_id}.jpg", image)
            
            app.logger.info(f'New user registered: {username}')
            return jsonify({
                'success': True, 
                'message': 'Registration successful! You can now login.'
            }), 200
            
        except sqlite3.IntegrityError as e:
            conn.rollback()
            if 'username' in str(e).lower():
                msg = 'Username already exists. Please choose another.'
            elif 'email' in str(e).lower():
                msg = 'Email already registered. Please login or use another email.'
            else:
                msg = 'Registration failed. Username or email may already exist.'
            return jsonify({'success': False, 'message': msg}), 400
            
        except Exception as e:
            conn.rollback()
            app.logger.error(f"Database error during registration: {str(e)}")
            return jsonify({
                'success': False, 
                'message': 'Database error. Please try again.'
            }), 500
        finally:
            conn.close()
            
    except Exception as e:
        app.logger.error(f"Registration error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'Registration failed. Please try again.'
        }), 500

@app.route('/api/login', methods=['POST'])
@validate_request_json('image')
def api_login():
    """Login using face recognition - PRODUCTION READY"""
    try:
        data = request.json
        image_data = data.get('image')
        
        valid, msg = validate_base64_image(image_data)
        if not valid:
            return jsonify({
                'success': False, 
                'message': f'Invalid image: {msg}'
            }), 400
        
        image_data = image_data.split(',')[1] if ',' in image_data else image_data
        image_bytes = base64.b64decode(image_data)
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return jsonify({
                'success': False, 
                'message': 'Could not process image'
            }), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_image)
        if len(face_locations) == 0:
            return jsonify({
                'success': False, 
                'message': 'No face detected. Please ensure your face is clearly visible.'
            }), 400
        
        face_encodings = face_recognition.face_encodings(rgb_image, face_locations)
        if len(face_encodings) == 0:
            return jsonify({
                'success': False, 
                'message': 'Could not generate face encoding. Please try again.'
            }), 400
        
        login_face_encoding = face_encodings[0]
        
        conn = None
        try:
            conn = get_db_connection()
            c = conn.cursor()
            c.execute('SELECT id, username, email, face_encoding FROM users')
            users = c.fetchall()
            
            if not users:
                return jsonify({
                    'success': False, 
                    'message': 'No registered users found. Please register first.'
                }), 404
            
            for user in users:
                user_id, username, email, stored_encoding_bytes = user
                stored_encoding = np.frombuffer(stored_encoding_bytes, dtype=np.float64)
                
                matches = face_recognition.compare_faces([stored_encoding], login_face_encoding, tolerance=0.6)
                face_distance = face_recognition.face_distance([stored_encoding], login_face_encoding)[0]
                
                if matches[0] and face_distance < 0.6:
                    session['user_id'] = user_id
                    session['username'] = username
                    session['email'] = email
                    session.permanent = True
                    
                    app.logger.info(f'User logged in: {username}')
                    
                    return jsonify({
                        'success': True,
                        'message': f'Welcome back, {username}!',
                        'username': username,
                        'confidence': float(1 - face_distance)
                }), 200
            
            return jsonify({
                'success': False, 
                'message': 'Face not recognized. Please register first or try again.'
            }), 401
            
        except Exception as e:
            app.logger.error(f"Database error during login: {str(e)}")
            return jsonify({
                'success': False, 
                'message': 'Database error. Please try again.'
            }), 500
        finally:
            if conn:
                conn.close()
        
    except Exception as e:
        app.logger.error(f"Login error: {str(e)}")
        return jsonify({
            'success': False, 
            'message': 'Login failed. Please try again.'
        }), 500

@app.route('/api/logout', methods=['POST'])
def api_logout():
    """Logout user"""
    session.clear()
    return jsonify({'success': True, 'message': 'Logged out successfully'}), 200

@app.route('/api/check_auth', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if 'user_id' in session:
        return jsonify({
            'authenticated': True,
            'username': session.get('username'),
            'email': session.get('email')
        }), 200
    return jsonify({'authenticated': False}), 401

# ==================== MODULE ROUTES ====================

@app.route('/module1')
def module1():
    """Module 1: Real-World Dimension Estimation"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module1.html')

@app.route('/module2')
def module2():
    """Module 2: Template Matching & Fourier Restoration"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module2.html')

@app.route('/module3')
def module3():
    """Module 3: Gradients, LoG, Edge/Corner Detection"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module3.html')

@app.route('/module4')
def module4():
    """Module 4: Image Stitching + SIFT"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module4.html')

@app.route('/module5')
def module5():
    """Module 5-6: Real-Time Object Trackers"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module5.html')

@app.route('/module7')
def module7():
    """Module 7: Stereo Calibration + Pose + Hand Tracking"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('module7.html')

@app.route('/documentation')
def documentation():
    """Documentation page"""
    if 'user_id' not in session:
        return render_template('login.html')
    return render_template('documentation.html')

# ==================== UTILITY ROUTES ====================

@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Upload file endpoint"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': 'Not authenticated'}), 401
    
    if 'file' not in request.files:
        return jsonify({'success': False, 'message': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'success': False, 'message': 'No file selected'}), 400
    
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)
    
    return jsonify({'success': True, 'filename': filename, 'path': filepath}), 200

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# ==================== HEALTH CHECK & MONITORING ====================

@app.route('/health')
def health_check():
    """Health check endpoint for monitoring"""
    try:
        conn = get_db_connection()
        c = conn.cursor()
        c.execute('SELECT COUNT(*) FROM users')
        user_count = c.fetchone()[0]
        conn.close()
        
        return jsonify({
            'status': 'healthy',
            'service': 'Computer Vision Project',
            'database': 'connected',
            'users': user_count,
            'timestamp': datetime.now().isoformat()
        }), 200
    except Exception as e:
        app.logger.error(f'Health check failed: {str(e)}')
        return jsonify({
            'status': 'unhealthy',
            'service': 'Computer Vision Project',
            'database': 'disconnected',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 503

@app.route('/api/status')
def api_status():
    """API status endpoint"""
    return jsonify({
        'api': 'operational',
        'version': '1.0',
        'endpoints': {
            'authentication': '/api/register, /api/login, /api/logout',
            'modules': '/api/module1/*, /api/module2/*, /api/module3/*, /api/module4/*, /api/module5/*, /api/module7/*',
            'health': '/health',
            'status': '/api/status'
        },
        'timestamp': datetime.now().isoformat()
    }), 200

@app.route('/ping')
def ping():
    """Simple ping endpoint"""
    return jsonify({'ping': 'pong', 'timestamp': datetime.now().isoformat()}), 200

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    debug_mode = os.environ.get('FLASK_ENV') != 'production'
    
    print("=" * 60)
    print("Computer Vision Project Website")
    print("Authors: Mahendra Krishna Koneru and Sai Leenath Jampala")
    print("=" * 60)
    print(f"\nStarting Flask server on port {port}")
    print(f"Environment: {'Development' if debug_mode else 'Production'}")
    print("Press Ctrl+C to stop the server\n")
    app.run(debug=debug_mode, host='0.0.0.0', port=port)


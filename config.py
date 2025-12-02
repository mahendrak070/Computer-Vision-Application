"""
Production Configuration
Author: Mahendra Krishna Koneru
"""

import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    MAX_CONTENT_LENGTH = 50 * 1024 * 1024
    
    UPLOAD_FOLDER = 'uploads'
    PERMANENT_SESSION_LIFETIME = timedelta(hours=24)
    
    DATABASE_PATH = 'database/users.db'
    FACE_ENCODINGS_PATH = 'face_encodings'
    
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'gif'}
    MAX_IMAGES_PER_UPLOAD = 10
    
    CORS_ORIGINS = '*'
    
    LOG_LEVEL = 'INFO'
    LOG_FILE = 'app.log'
    
    DEBUG = os.environ.get('FLASK_ENV') == 'development'
    
    REQUEST_TIMEOUT = 120
    
    MIN_USERNAME_LENGTH = 3
    MAX_USERNAME_LENGTH = 50
    MIN_PASSWORD_LENGTH = 8

class ProductionConfig(Config):
    DEBUG = False
    TESTING = False
    LOG_LEVEL = 'WARNING'

class DevelopmentConfig(Config):
    DEBUG = True
    TESTING = False
    LOG_LEVEL = 'DEBUG'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


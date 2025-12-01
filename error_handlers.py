"""
Comprehensive Error Handlers
Authors: Mahendra Krishna Koneru and Sai Leenath Jampala
"""

from flask import jsonify, render_template, request
import logging
import traceback

logger = logging.getLogger(__name__)

def register_error_handlers(app):
    """Register all error handlers for the Flask app"""
    
    @app.errorhandler(400)
    def bad_request(error):
        """Handle 400 Bad Request"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Bad Request',
                'message': str(error.description) if hasattr(error, 'description') else 'Invalid request'
            }), 400
        return render_template('error.html', 
                             error_code=400, 
                             error_title='Bad Request',
                             error_message='Your request could not be understood.'), 400
    
    @app.errorhandler(401)
    def unauthorized(error):
        """Handle 401 Unauthorized"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Unauthorized',
                'message': 'Authentication required'
            }), 401
        return render_template('error.html',
                             error_code=401,
                             error_title='Unauthorized',
                             error_message='Please log in to access this page.'), 401
    
    @app.errorhandler(403)
    def forbidden(error):
        """Handle 403 Forbidden"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Forbidden',
                'message': 'Access denied'
            }), 403
        return render_template('error.html',
                             error_code=403,
                             error_title='Forbidden',
                             error_message='You do not have permission to access this resource.'), 403
    
    @app.errorhandler(404)
    def not_found(error):
        """Handle 404 Not Found"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Not Found',
                'message': 'The requested resource was not found'
            }), 404
        return render_template('error.html',
                             error_code=404,
                             error_title='Page Not Found',
                             error_message='The page you are looking for does not exist.'), 404
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        """Handle 413 Payload Too Large"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Payload Too Large',
                'message': 'File size exceeds the maximum limit (50MB)'
            }), 413
        return render_template('error.html',
                             error_code=413,
                             error_title='File Too Large',
                             error_message='The uploaded file is too large. Maximum size is 50MB.'), 413
    
    @app.errorhandler(429)
    def too_many_requests(error):
        """Handle 429 Too Many Requests"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Too Many Requests',
                'message': 'Rate limit exceeded. Please try again later.'
            }), 429
        return render_template('error.html',
                             error_code=429,
                             error_title='Too Many Requests',
                             error_message='You have made too many requests. Please wait a moment.'), 429
    
    @app.errorhandler(500)
    def internal_server_error(error):
        """Handle 500 Internal Server Error"""
        logger.error(f'Internal Server Error: {str(error)}')
        logger.error(traceback.format_exc())
        
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Internal Server Error',
                'message': 'An unexpected error occurred. Please try again later.'
            }), 500
        return render_template('error.html',
                             error_code=500,
                             error_title='Internal Server Error',
                             error_message='Something went wrong on our end. Please try again later.'), 500
    
    @app.errorhandler(502)
    def bad_gateway(error):
        """Handle 502 Bad Gateway"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Bad Gateway',
                'message': 'Service temporarily unavailable'
            }), 502
        return render_template('error.html',
                             error_code=502,
                             error_title='Bad Gateway',
                             error_message='The service is temporarily unavailable.'), 502
    
    @app.errorhandler(503)
    def service_unavailable(error):
        """Handle 503 Service Unavailable"""
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Service Unavailable',
                'message': 'Service is under maintenance'
            }), 503
        return render_template('error.html',
                             error_code=503,
                             error_title='Service Unavailable',
                             error_message='The service is currently under maintenance.'), 503
    
    @app.errorhandler(Exception)
    def handle_exception(error):
        """Handle any unhandled exceptions"""
        logger.error(f'Unhandled Exception: {str(error)}')
        logger.error(traceback.format_exc())
        
        if request.path.startswith('/api/'):
            return jsonify({
                'success': False,
                'error': 'Internal Error',
                'message': 'An unexpected error occurred. Please try again.'
            }), 500
        return render_template('error.html',
                             error_code=500,
                             error_title='Unexpected Error',
                             error_message='An unexpected error occurred. Our team has been notified.'), 500




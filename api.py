#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
EnvironmentTagger API - RESTful API for the EnvironmentTagger tool

This module provides a Flask-based RESTful API for interacting with the
EnvironmentTagger for media environment analysis and tagging.

Author: Media Automation Team
Version: 1.0.0
Last Updated: 2025-06-24
"""

import os
import json
import uuid
import logging
import tempfile
from typing import Dict, List, Any, Optional
from werkzeug.utils import secure_filename

from flask import Flask, request, jsonify, send_file, abort
from flask_restful import Api, Resource
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

import environment_tagger

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
def load_config():
    """Load configuration from file or environment variables."""
    config_path = os.environ.get('CONFIG_PATH', 'config.json')
    try:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                return json.load(f)
        else:
            logger.warning(f"Config file {config_path} not found, using defaults")
            return {}
    except Exception as e:
        logger.error(f"Error loading config: {str(e)}")
        return {}

config = load_config()
api_config = config.get('api', {})

# Initialize Flask app
app = Flask(__name__)
CORS(app)
api = Api(app)

# Configure rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=[
        f"{api_config.get('rate_limits', {}).get('requests_per_minute', 60)} per minute",
        "1000 per day"
    ],
    strategy="fixed-window"
)

# Initialize EnvironmentTagger
tagger = environment_tagger.EnvironmentTagger(
    api_key=os.environ.get('GEMINI_API_KEY'),
    config_path=os.environ.get('CONFIG_PATH', 'config.json')
)

# API Authentication middleware
def require_api_key(func):
    """Decorator to require API key authentication."""
    def wrapper(*args, **kwargs):
        if not api_config.get('authentication', {}).get('require_api_key', False):
            return func(*args, **kwargs)
            
        api_key = request.headers.get('X-API-Key')
        if not api_key:
            return {"error": "API key required"}, 401
            
        # In a real implementation, validate the API key against a database
        # For this example, we'll accept any non-empty key
        if not api_key:
            return {"error": "Invalid API key"}, 403
            
        return func(*args, **kwargs)
    return wrapper

# API Resources
class HealthCheck(Resource):
    def get(self):
        """API health check endpoint."""
        return {
            "status": "ok",
            "version": "1.0.0",
            "service": "EnvironmentTagger API"
        }

class AnalyzeMedia(Resource):
    @limiter.limit("10 per minute")
    @require_api_key
    def post(self):
        """
        Analyze media file for environment detection.
        
        Accepts:
        - Uploaded file
        - URL
        - File path (for server-side files)
        """
        try:
            data = request.json or {}
            detail_level = data.get('detail_level', 'standard')
            include_palette = data.get('include_palette', True)
            include_objects = data.get('include_objects', True)
            include_scene_type = data.get('include_scene_type', True)
            
            # Check if there's a file in the request
            if 'file' in request.files:
                file = request.files['file']
                filename = secure_filename(file.filename)
                
                # Save to temporary file
                temp_dir = tempfile.mkdtemp()
                file_path = os.path.join(temp_dir, filename)
                file.save(file_path)
                
                # Analyze file
                results = tagger.analyze_image(
                    file_path,
                    detail_level=detail_level,
                    include_palette=include_palette,
                    include_objects=include_objects,
                    include_scene_type=include_scene_type
                )
                
                # Clean up
                os.remove(file_path)
                os.rmdir(temp_dir)
                
                return jsonify(results)
                
            # Check if there's a URL
            elif 'url' in data:
                url = data['url']
                # In a real implementation, download the file first
                # For this example, we'll just return a placeholder
                return {
                    "error": "URL processing not implemented in this example"
                }, 501
                
            # Check if there's a path
            elif 'path' in data:
                path = data['path']
                
                # For security, restrict to specific directories
                allowed_dirs = config.get('allowed_media_dirs', ['/media', '/data'])
                if not any(path.startswith(allowed_dir) for allowed_dir in allowed_dirs):
                    return {
                        "error": "Path not allowed for security reasons"
                    }, 403
                
                # Check if file exists
                if not os.path.exists(path):
                    return {
                        "error": f"File not found: {path}"
                    }, 404
                
                # Analyze file
                results = tagger.analyze_image(
                    path,
                    detail_level=detail_level,
                    include_palette=include_palette,
                    include_objects=include_objects,
                    include_scene_type=include_scene_type
                )
                
                return jsonify(results)
                
            else:
                return {
                    "error": "No media provided. Please upload a file, provide a URL, or specify a path."
                }, 400
                
        except Exception as e:
            logger.error(f"Error in analyze endpoint: {str(e)}")
            return {
                "error": f"Analysis failed: {str(e)}"
            }, 500

class BatchAnalyze(Resource):
    @limiter.limit("5 per minute")
    @require_api_key
    def post(self):
        """Batch analyze multiple media files."""
        try:
            data = request.json
            if not data or not isinstance(data, dict):
                return {
                    "error": "Invalid request format"
                }, 400
            
            media_paths = data.get('media_paths', [])
            if not media_paths or not isinstance(media_paths, list):
                return {
                    "error": "media_paths is required and must be a list"
                }, 400
            
            detail_level = data.get('detail_level', 'standard')
            parallel_processing = data.get('parallel_processing', True)
            max_concurrent = data.get('max_concurrent', 5)
            
            # Create a job ID
            job_id = str(uuid.uuid4())
            
            # In a real implementation, this would start an async job
            # For this example, we'll process synchronously
            results = tagger.batch_analyze(
                media_paths=media_paths,
                detail_level=detail_level,
                parallel_processing=parallel_processing,
                max_concurrent=max_concurrent
            )
            
            return jsonify({
                "job_id": job_id,
                "status": "completed",
                "results": results
            })
            
        except Exception as e:
            logger.error(f"Error in batch analyze endpoint: {str(e)}")
            return {
                "error": f"Batch analysis failed: {str(e)}"
            }, 500

class BatchStatus(Resource):
    @require_api_key
    def get(self, job_id):
        """Get status of a batch job."""
        # In a real implementation, this would check job status
        # For this example, we'll return a placeholder
        return {
            "job_id": job_id,
            "status": "not_implemented",
            "message": "Batch job status checking not implemented in this example"
        }, 501

# Register API routes
api.add_resource(HealthCheck, '/health')
api.add_resource(AnalyzeMedia, '/analyze')
api.add_resource(BatchAnalyze, '/batch')
api.add_resource(BatchStatus, '/batch/<string:job_id>')

if __name__ == '__main__':
    # Get host and port from config or defaults
    host = api_config.get('host', '0.0.0.0')
    port = api_config.get('port', 5000)
    debug = api_config.get('debug', False)
    
    logger.info(f"Starting EnvironmentTagger API on {host}:{port}")
    app.run(host=host, port=port, debug=debug)
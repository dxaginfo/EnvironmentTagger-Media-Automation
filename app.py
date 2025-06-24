#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment Tagger - Web Application

This module provides a Flask web application for the EnvironmentTagger service.
"""

import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import tempfile
import uuid

from flask import Flask, request, jsonify, send_file
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

from environmenttagger import EnvironmentTagger, TagResult, BatchJob, AnalysisDepth

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize environment tagger
tagger = EnvironmentTagger(
    api_key=os.environ.get("GEMINI_API_KEY"),
    enable_drive=os.environ.get("ENABLE_DRIVE", "false").lower() == "true",
    drive_credentials=os.environ.get("DRIVE_CREDENTIALS"),
    storage_bucket=os.environ.get("STORAGE_BUCKET")
)

# Create upload directory if it doesn't exist
UPLOAD_FOLDER = os.path.join(tempfile.gettempdir(), "environmenttagger_uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route('/analyze', methods=['POST'])
def analyze():
    """Analyze media and generate environment tags."""
    try:
        # Check if request includes a file or a URL
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return jsonify({'error': 'No file selected'}), 400
            
            # Save file to temporary location
            filename = secure_filename(file.filename)
            file_path = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}_{filename}")
            file.save(file_path)
            media_source = file_path
        elif 'url' in request.form:
            media_source = request.form['url']
        elif 'drive_id' in request.form:
            media_source = f"drive://{request.form['drive_id']}"
        else:
            return jsonify({'error': 'No media source provided'}), 400
        
        # Get analysis parameters
        analysis_depth = request.form.get('analysis_depth', 'standard')
        if analysis_depth not in ['basic', 'standard', 'detailed']:
            return jsonify({'error': 'Invalid analysis depth'}), 400
        
        # Determine if media is image or video
        is_video = False
        if 'file' in request.files:
            is_video = file.filename.lower().endswith(('.mp4', '.avi', '.mov', '.mkv'))
        elif 'media_type' in request.form:
            is_video = request.form['media_type'].lower() == 'video'
        
        # Process media
        if is_video:
            frame_interval = int(request.form.get('frame_interval', '10'))
            results = tagger.tag_video(media_source, analysis_depth, frame_interval)
            response = {
                'media_id': media_source if 'file' not in request.files else file.filename,
                'media_type': 'video',
                'frame_count': len(results),
                'analysis_timestamp': datetime.now().isoformat(),
                'frames': [result.dict() for result in results]
            }
        else:
            result = tagger.tag_image(media_source, analysis_depth)
            response = result.dict()
        
        # Clean up temporary file if needed
        if 'file' in request.files and os.path.exists(file_path):
            os.remove(file_path)
        
        # Return results in requested format
        output_format = request.form.get('output_format', 'json')
        if output_format == 'json':
            return jsonify(response)
        elif output_format == 'csv':
            # Convert to CSV (simple implementation)
            if is_video:
                csv_data = "frame,primary_environment,secondary_environments,confidence_score\n"
                for i, frame in enumerate(response['frames']):
                    csv_data += f"{i},{frame.get('primary_environment', '')},\"{','.join(frame.get('secondary_environments', []))}\",{frame.get('confidence_score', '')}\n"
            else:
                csv_data = "primary_environment,secondary_environments,confidence_score\n"
                csv_data += f"{response.get('primary_environment', '')},\"{','.join(response.get('secondary_environments', []))}\",{response.get('confidence_score', '')}\n"
            
            # Create temporary CSV file
            csv_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.csv")
            with open(csv_path, 'w') as f:
                f.write(csv_data)
            
            return send_file(csv_path, as_attachment=True, download_name="environment_tags.csv")
        elif output_format == 'xml':
            # Simple XML conversion
            xml_data = '<?xml version="1.0" encoding="UTF-8"?>\n'
            xml_data += '<EnvironmentTags>\n'
            
            if is_video:
                for i, frame in enumerate(response['frames']):
                    xml_data += f'  <Frame number="{i}">\n'
                    xml_data += f'    <PrimaryEnvironment>{frame.get("primary_environment", "")}</PrimaryEnvironment>\n'
                    xml_data += '    <SecondaryEnvironments>\n'
                    for env in frame.get('secondary_environments', []):
                        xml_data += f'      <Environment>{env}</Environment>\n'
                    xml_data += '    </SecondaryEnvironments>\n'
                    xml_data += f'    <ConfidenceScore>{frame.get("confidence_score", "")}</ConfidenceScore>\n'
                    xml_data += '  </Frame>\n'
            else:
                xml_data += f'  <PrimaryEnvironment>{response.get("primary_environment", "")}</PrimaryEnvironment>\n'
                xml_data += '  <SecondaryEnvironments>\n'
                for env in response.get('secondary_environments', []):
                    xml_data += f'    <Environment>{env}</Environment>\n'
                xml_data += '  </SecondaryEnvironments>\n'
                xml_data += f'  <ConfidenceScore>{response.get("confidence_score", "")}</ConfidenceScore>\n'
            
            xml_data += '</EnvironmentTags>'
            
            # Create temporary XML file
            xml_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}.xml")
            with open(xml_path, 'w') as f:
                f.write(xml_data)
            
            return send_file(xml_path, as_attachment=True, download_name="environment_tags.xml")
        else:
            return jsonify({'error': 'Invalid output format'}), 400
    
    except Exception as e:
        logging.error(f"Error in analyze endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch', methods=['POST'])
def batch():
    """Process multiple media files."""
    try:
        # Validate request parameters
        if not request.json or 'media_sources' not in request.json:
            return jsonify({'error': 'media_sources is required'}), 400
        
        media_sources = request.json['media_sources']
        if not isinstance(media_sources, list) or len(media_sources) == 0:
            return jsonify({'error': 'media_sources must be a non-empty array'}), 400
        
        analysis_depth = request.json.get('analysis_depth', 'standard')
        if analysis_depth not in ['basic', 'standard', 'detailed']:
            return jsonify({'error': 'Invalid analysis depth'}), 400
        
        extract_frames = request.json.get('extract_frames', False)
        frame_interval = request.json.get('frame_interval', 10)
        
        # Create batch job
        batch_job = tagger.create_batch_job(
            media_sources=media_sources,
            analysis_depth=analysis_depth,
            extract_frames=extract_frames,
            frame_interval=frame_interval
        )
        
        # Start processing in background (in a real-world application, this would be done asynchronously)
        batch_job = tagger.start_batch_job(batch_job)
        
        # Return job information
        return jsonify({
            'job_id': batch_job.job_id,
            'status': batch_job.status,
            'created_at': batch_job.created_at,
            'started_at': batch_job.started_at,
            'completed_at': batch_job.completed_at,
            'processed_count': batch_job.processed_count,
            'total_count': batch_job.total_count
        })
    
    except Exception as e:
        logging.error(f"Error in batch endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/batch/<job_id>', methods=['GET'])
def get_batch_job(job_id):
    """Get status and results of a batch job."""
    try:
        batch_job = tagger.get_batch_job(job_id)
        if not batch_job:
            return jsonify({'error': 'Job not found'}), 404
        
        # Return job information and results
        response = {
            'job_id': batch_job.job_id,
            'status': batch_job.status,
            'created_at': batch_job.created_at,
            'started_at': batch_job.started_at,
            'completed_at': batch_job.completed_at,
            'processed_count': batch_job.processed_count,
            'total_count': batch_job.total_count
        }
        
        # Include results if requested
        if request.args.get('include_results', 'false').lower() == 'true':
            response['results'] = batch_job.results
        
        # Include errors if present
        if batch_job.errors:
            response['errors'] = batch_job.errors
        
        return jsonify(response)
    
    except Exception as e:
        logging.error(f"Error in get_batch_job endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/tags/custom', methods=['POST'])
def custom_tags():
    """Define custom tagging templates."""
    try:
        # Validate request parameters
        if not request.json:
            return jsonify({'error': 'Invalid request'}), 400
        
        if 'template_name' not in request.json:
            return jsonify({'error': 'template_name is required'}), 400
        
        if 'tags' not in request.json or not isinstance(request.json['tags'], list):
            return jsonify({'error': 'tags must be an array'}), 400
        
        # For now, this is a placeholder implementation
        # In a real application, this would store the custom template in a database
        
        return jsonify({
            'status': 'success',
            'message': 'Custom tagging template created',
            'template_name': request.json['template_name']
        })
    
    except Exception as e:
        logging.error(f"Error in custom_tags endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/export', methods=['POST'])
def export():
    """Export tagging results."""
    try:
        # Validate request parameters
        if not request.json or 'job_id' not in request.json:
            return jsonify({'error': 'job_id is required'}), 400
        
        job_id = request.json['job_id']
        export_format = request.json.get('format', 'json')
        destination = request.json.get('destination', 'download')
        
        # Get batch job results
        batch_job = tagger.get_batch_job(job_id)
        if not batch_job:
            return jsonify({'error': 'Job not found'}), 404
        
        if batch_job.status != 'completed':
            return jsonify({'error': 'Job is not completed yet'}), 400
        
        # Handle export based on format and destination
        if export_format == 'json':
            if destination == 'download':
                # Create JSON file for download
                json_path = os.path.join(tempfile.gettempdir(), f"{job_id}.json")
                with open(json_path, 'w') as f:
                    json.dump(batch_job.results, f, indent=2)
                
                return send_file(json_path, as_attachment=True, download_name=f"environment_tags_{job_id}.json")
            elif destination == 'drive' and tagger.enable_drive:
                # Export to Google Drive
                # This would be implemented in a real application
                return jsonify({'error': 'Export to Drive not implemented yet'}), 501
            else:
                return jsonify({'error': 'Invalid destination'}), 400
        
        elif export_format == 'sheet' and tagger.enable_drive:
            # Export to Google Sheet
            sheet_name = request.json.get('sheet_name', f"EnvironmentTags_{job_id}")
            spreadsheet_id = tagger.export_to_sheet(
                results=batch_job.results,
                sheet_name=sheet_name,
                create_new=True
            )
            
            return jsonify({
                'status': 'success',
                'message': 'Exported to Google Sheet',
                'spreadsheet_id': spreadsheet_id
            })
        
        else:
            return jsonify({'error': 'Invalid export format or destination'}), 400
    
    except Exception as e:
        logging.error(f"Error in export endpoint: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'ok',
        'version': '0.1.0',
        'drive_integration': tagger.enable_drive,
        'gemini_api': tagger.gemini_model is not None
    })


if __name__ == '__main__':
    # Run the Flask application
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port, debug=False)
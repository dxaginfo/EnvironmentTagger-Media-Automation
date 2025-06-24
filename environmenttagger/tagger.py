#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment Tagger - Core Module

This module provides the main functionality for detecting and tagging environments in media.
"""

import os
import uuid
import logging
from typing import List, Dict, Union, Optional, Any
from datetime import datetime
import json
import tempfile
import time

import google.generativeai as genai
from google.cloud import vision, storage
from PIL import Image
import numpy as np
import requests
import cv2

from .models import TagResult, BatchJob, AnalysisDepth, MediaSource, Environment
from .media_processor import MediaProcessor
from .utils import download_media, extract_frames, get_file_extension


class EnvironmentTagger:
    """Main class for environment detection and tagging."""

    def __init__(
        self,
        api_key: str = None,
        enable_drive: bool = False,
        drive_credentials: str = None,
        storage_bucket: str = None,
    ):
        """Initialize the EnvironmentTagger.

        Args:
            api_key: API key for Gemini API
            enable_drive: Whether to enable Google Drive integration
            drive_credentials: Path to Google Drive credentials file
            storage_bucket: Google Cloud Storage bucket for processing
        """
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        self.enable_drive = enable_drive
        self.drive_credentials = drive_credentials
        self.storage_bucket = storage_bucket
        
        # Initialize Google Vision client
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize Gemini API
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro-vision')
        else:
            logging.warning("No Gemini API key provided. Gemini integration disabled.")
            self.gemini_model = None
            
        # Initialize media processor
        self.media_processor = MediaProcessor()
        
        # Initialize storage client if bucket is provided
        if self.storage_bucket:
            self.storage_client = storage.Client()
            self.bucket = self.storage_client.bucket(self.storage_bucket)
        else:
            self.storage_client = None
            self.bucket = None
            
        # Initialize Google Drive integration if enabled
        if self.enable_drive:
            try:
                from googleapiclient.discovery import build
                from google.oauth2 import service_account
                
                if drive_credentials:
                    self.drive_credentials = service_account.Credentials.from_service_account_file(
                        drive_credentials, scopes=['https://www.googleapis.com/auth/drive']
                    )
                    self.drive_service = build('drive', 'v3', credentials=self.drive_credentials)
                else:
                    logging.warning("Drive credentials not provided. Drive integration limited.")
                    self.drive_service = None
            except ImportError:
                logging.error("Google Drive integration requires google-api-python-client")
                self.enable_drive = False
                self.drive_service = None

    def tag_image(self, image_path: str, analysis_depth: str = "standard") -> TagResult:
        """Tag a single image with environment information.

        Args:
            image_path: Path or URL to the image
            analysis_depth: Depth of analysis (basic, standard, detailed)

        Returns:
            TagResult object containing environment tags
        """
        # Determine if path is local file, URL, or Drive ID
        media_source = self._determine_media_source(image_path)
        
        # Download/load the image
        local_path = self._get_local_media(media_source)
        
        # Process based on analysis depth
        analysis_depth_enum = AnalysisDepth(analysis_depth)
        
        # Create base result
        result = TagResult(
            media_id=os.path.basename(image_path),
            analysis_timestamp=datetime.now().isoformat(),
        )
        
        # Perform Vision API analysis
        vision_results = self._analyze_with_vision(local_path)
        result.primary_tags = self._extract_primary_tags(vision_results)
        
        # For standard and detailed analysis, use Gemini API
        if analysis_depth_enum in [AnalysisDepth.STANDARD, AnalysisDepth.DETAILED]:
            gemini_results = self._analyze_with_gemini(local_path, vision_results)
            result.secondary_tags = gemini_results.get('secondary_tags', [])
            result.attributes = gemini_results.get('attributes', {})
            
            # For detailed analysis, add more comprehensive information
            if analysis_depth_enum == AnalysisDepth.DETAILED:
                result.objects = gemini_results.get('objects', [])
                result.color_palette = gemini_results.get('color_palette', [])
                result.suggested_tags = gemini_results.get('suggested_tags', [])
        
        # Clean up temporary files
        if local_path != image_path and os.path.exists(local_path):
            os.remove(local_path)
            
        return result
    
    def tag_video(self, video_path: str, analysis_depth: str = "standard", frame_interval: int = 10) -> List[TagResult]:
        """Tag a video by extracting and analyzing frames.

        Args:
            video_path: Path or URL to the video
            analysis_depth: Depth of analysis (basic, standard, detailed)
            frame_interval: Interval between frames to extract (in seconds)

        Returns:
            List of TagResult objects for each analyzed frame
        """
        # Determine if path is local file, URL, or Drive ID
        media_source = self._determine_media_source(video_path)
        
        # Download/load the video
        local_path = self._get_local_media(media_source)
        
        # Extract frames from video
        frames = extract_frames(local_path, frame_interval)
        
        results = []
        for i, frame in enumerate(frames):
            # Save frame temporarily
            frame_path = os.path.join(tempfile.gettempdir(), f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Tag the frame
            frame_result = self.tag_image(frame_path, analysis_depth)
            frame_result.frame_number = i
            frame_result.timestamp = i * frame_interval
            results.append(frame_result)
            
            # Remove temporary frame file
            os.remove(frame_path)
        
        # Clean up downloaded video if necessary
        if local_path != video_path and os.path.exists(local_path):
            os.remove(local_path)
            
        return results
    
    def create_batch_job(
        self,
        media_sources: List[str],
        analysis_depth: str = "standard",
        extract_frames: bool = False,
        frame_interval: int = 10
    ) -> BatchJob:
        """Create a batch job for processing multiple media files.

        Args:
            media_sources: List of paths or URLs to media files
            analysis_depth: Depth of analysis (basic, standard, detailed)
            extract_frames: Whether to extract frames from videos
            frame_interval: Interval between frames to extract (in seconds)

        Returns:
            BatchJob object for tracking progress
        """
        job_id = str(uuid.uuid4())
        job = BatchJob(
            job_id=job_id,
            media_sources=media_sources,
            analysis_depth=analysis_depth,
            extract_frames=extract_frames,
            frame_interval=frame_interval,
            status="created",
            created_at=datetime.now().isoformat(),
            processed_count=0,
            total_count=len(media_sources),
            results={}
        )
        
        # Save job to a temporary file or storage
        self._save_job(job)
        
        return job
    
    def start_batch_job(self, job: BatchJob) -> BatchJob:
        """Start processing a batch job.

        Args:
            job: BatchJob object to process

        Returns:
            Updated BatchJob object
        """
        job.status = "processing"
        job.started_at = datetime.now().isoformat()
        self._save_job(job)
        
        for i, media_source in enumerate(job.media_sources):
            # Determine file type
            extension = get_file_extension(media_source)
            is_video = extension.lower() in ['.mp4', '.avi', '.mov', '.mkv']
            
            try:
                if is_video and job.extract_frames:
                    # Process video
                    results = self.tag_video(
                        media_source,
                        analysis_depth=job.analysis_depth,
                        frame_interval=job.frame_interval
                    )
                    job.results[media_source] = [r.dict() for r in results]
                else:
                    # Process image
                    result = self.tag_image(media_source, analysis_depth=job.analysis_depth)
                    job.results[media_source] = result.dict()
                
                job.processed_count += 1
                self._save_job(job)
            except Exception as e:
                job.errors[media_source] = str(e)
                logging.error(f"Error processing {media_source}: {e}")
        
        job.status = "completed"
        job.completed_at = datetime.now().isoformat()
        self._save_job(job)
        
        return job
    
    def get_batch_job(self, job_id: str) -> Optional[BatchJob]:
        """Get a batch job by ID.

        Args:
            job_id: ID of the job to retrieve

        Returns:
            BatchJob object if found, None otherwise
        """
        # Retrieve job from storage
        job_data = self._load_job(job_id)
        if not job_data:
            return None
        
        return BatchJob(**job_data)
    
    def tag_drive_folder(
        self,
        folder_id: str,
        recursive: bool = False,
        file_types: List[str] = None,
    ) -> Dict[str, TagResult]:
        """Tag all media files in a Google Drive folder.

        Args:
            folder_id: ID of the Google Drive folder
            recursive: Whether to process subfolders
            file_types: List of file types to process (e.g., ['image', 'video'])

        Returns:
            Dictionary mapping file IDs to TagResult objects
        """
        if not self.enable_drive or not self.drive_service:
            raise ValueError("Google Drive integration not enabled or configured.")
        
        file_types = file_types or ['image', 'video']
        mime_types = []
        
        if 'image' in file_types:
            mime_types.extend(['image/jpeg', 'image/png', 'image/tiff', 'image/webp'])
        if 'video' in file_types:
            mime_types.extend(['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/x-matroska'])
        
        # Build query for files in the folder
        query = f"'{folder_id}' in parents and trashed = false"
        if mime_types:
            mime_query = ' or '.join([f"mimeType = '{mime}'" for mime in mime_types])
            query += f" and ({mime_query})"
        
        # Get files in the folder
        response = self.drive_service.files().list(
            q=query,
            spaces='drive',
            fields='files(id, name, mimeType)'
        ).execute()
        
        files = response.get('files', [])
        
        # Create batch job for processing
        media_sources = [f"drive://{file['id']}" for file in files]
        batch_job = self.create_batch_job(
            media_sources=media_sources,
            analysis_depth="standard",
            extract_frames=True
        )
        
        # Start processing
        batch_job = self.start_batch_job(batch_job)
        
        # Return results
        return batch_job.results
    
    def export_to_sheet(self, results: Dict[str, TagResult], sheet_name: str, create_new: bool = True) -> str:
        """Export tagging results to a Google Sheet.

        Args:
            results: Dictionary of tagging results
            sheet_name: Name for the Google Sheet
            create_new: Whether to create a new sheet or update existing

        Returns:
            ID of the created/updated Google Sheet
        """
        if not self.enable_drive or not self.drive_service:
            raise ValueError("Google Drive integration not enabled or configured.")
        
        try:
            from googleapiclient.discovery import build
            from google.oauth2 import service_account
            
            # Create Sheets service
            sheets_service = build('sheets', 'v4', credentials=self.drive_credentials)
            
            # Create new spreadsheet if needed
            if create_new:
                spreadsheet_body = {
                    'properties': {'title': sheet_name}
                }
                request = sheets_service.spreadsheets().create(body=spreadsheet_body)
                response = request.execute()
                spreadsheet_id = response['spreadsheetId']
            else:
                # TODO: Implement finding existing sheet
                raise NotImplementedError("Updating existing sheets not implemented yet")
            
            # Prepare data for the sheet
            header_row = [
                "Media ID", "Primary Environment", "Secondary Environments",
                "Lighting", "Size", "Occupancy", "Mood", "Period",
                "Confidence Score", "Analysis Timestamp"
            ]
            
            data = [header_row]
            
            # Add rows for each result
            for media_id, result in results.items():
                # Handle if result is a dict or TagResult object
                if isinstance(result, dict):
                    r = result
                else:
                    r = result.dict()
                
                # Extract attributes
                attributes = r.get('attributes', {})
                
                row = [
                    r.get('media_id', media_id),
                    r.get('primary_environment', ''),
                    ", ".join(r.get('secondary_environments', [])),
                    attributes.get('lighting', ''),
                    attributes.get('size', ''),
                    attributes.get('occupancy', ''),
                    attributes.get('mood', ''),
                    attributes.get('period', ''),
                    r.get('confidence_score', ''),
                    r.get('analysis_timestamp', '')
                ]
                
                data.append(row)
            
            # Update the sheet
            body = {
                'values': data
            }
            result = sheets_service.spreadsheets().values().update(
                spreadsheetId=spreadsheet_id,
                range='Sheet1!A1',
                valueInputOption='RAW',
                body=body
            ).execute()
            
            return spreadsheet_id
            
        except ImportError:
            logging.error("Google Sheets integration requires google-api-python-client")
            raise
    
    def _determine_media_source(self, path: str) -> MediaSource:
        """Determine the type of media source from the path.

        Args:
            path: Path or URL to the media

        Returns:
            MediaSource object with source type and path
        """
        if path.startswith('http://') or path.startswith('https://'):
            return MediaSource(type='url', path=path)
        elif path.startswith('drive://'):
            drive_id = path.replace('drive://', '')
            return MediaSource(type='drive', path=drive_id)
        else:
            return MediaSource(type='local', path=path)
    
    def _get_local_media(self, media_source: MediaSource) -> str:
        """Get local path to media from various sources.

        Args:
            media_source: MediaSource object with source information

        Returns:
            Local path to the media file
        """
        if media_source.type == 'local':
            return media_source.path
        elif media_source.type == 'url':
            return download_media(media_source.path)
        elif media_source.type == 'drive':
            if not self.enable_drive or not self.drive_service:
                raise ValueError("Google Drive integration not enabled or configured.")
            
            # Download from Drive
            file_id = media_source.path
            request = self.drive_service.files().get_media(fileId=file_id)
            
            # Save to temporary file
            file_metadata = self.drive_service.files().get(fileId=file_id, fields='name').execute()
            file_name = file_metadata.get('name', 'drive_file')
            
            temp_path = os.path.join(tempfile.gettempdir(), file_name)
            with open(temp_path, 'wb') as f:
                downloader = requests.Request(request, f)
                downloader.next_chunk()
            
            return temp_path
        else:
            raise ValueError(f"Unsupported media source type: {media_source.type}")
    
    def _analyze_with_vision(self, image_path: str) -> Dict[str, Any]:
        """Analyze image with Google Cloud Vision API.

        Args:
            image_path: Local path to the image

        Returns:
            Dictionary with Vision API results
        """
        with open(image_path, 'rb') as image_file:
            content = image_file.read()
        
        image = vision.Image(content=content)
        
        # Request all relevant features
        features = [
            vision.Feature(type_=vision.Feature.Type.LABEL_DETECTION),
            vision.Feature(type_=vision.Feature.Type.LANDMARK_DETECTION),
            vision.Feature(type_=vision.Feature.Type.OBJECT_LOCALIZATION),
            vision.Feature(type_=vision.Feature.Type.IMAGE_PROPERTIES),
            vision.Feature(type_=vision.Feature.Type.SAFE_SEARCH_DETECTION),
        ]
        
        request = vision.AnnotateImageRequest(image=image, features=features)
        response = self.vision_client.annotate_image(request=request)
        
        # Convert to dictionary
        response_json = type(response).to_json(response)
        return json.loads(response_json)
    
    def _extract_primary_tags(self, vision_results: Dict[str, Any]) -> List[str]:
        """Extract primary environment tags from Vision API results.

        Args:
            vision_results: Dictionary with Vision API results

        Returns:
            List of primary environment tags
        """
        # Extract labels
        labels = [label.get('description') for label in vision_results.get('labelAnnotations', [])]
        
        # Filter for environment-related labels
        environment_keywords = [
            'indoor', 'outdoor', 'interior', 'exterior', 'room', 'building',
            'office', 'home', 'house', 'apartment', 'studio', 'kitchen',
            'bathroom', 'bedroom', 'living room', 'hallway', 'corridor',
            'garden', 'park', 'forest', 'beach', 'mountain', 'street',
            'urban', 'rural', 'natural', 'artificial', 'commercial', 'residential',
            'industrial', 'agricultural', 'public', 'private'
        ]
        
        environment_tags = [label for label in labels 
                         if any(keyword in label.lower() for keyword in environment_keywords)]
        
        # Add landmark if detected
        landmarks = [landmark.get('description') for landmark in vision_results.get('landmarkAnnotations', [])]
        if landmarks:
            environment_tags.extend(landmarks)
        
        return environment_tags[:5]  # Return top 5 environment tags
    
    def _analyze_with_gemini(self, image_path: str, vision_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze image with Gemini API for enhanced understanding.

        Args:
            image_path: Local path to the image
            vision_results: Dictionary with Vision API results

        Returns:
            Dictionary with Gemini API results
        """
        if not self.gemini_model:
            raise ValueError("Gemini API not configured.")
        
        # Load image for Gemini
        image = Image.open(image_path)
        
        # Create prompt with Vision API results for context
        labels = [label.get('description') for label in vision_results.get('labelAnnotations', [])]
        label_text = ", ".join(labels[:10])
        
        prompt = f"""Analyze this image as an environment for media production. 
        Google Vision labels detected: {label_text}
        
        Provide detailed environment information in JSON format with these fields:
        1. secondary_tags: List of specific environment descriptors (e.g., 'corporate meeting room', 'beachfront resort')
        2. attributes: Object with lighting, size, occupancy, mood, and period of the environment
        3. objects: List of key objects in the environment
        4. color_palette: List of 5 hex color codes that represent the environment's palette
        5. suggested_tags: List of 5 media production relevant tags for this environment
        
        Return ONLY valid JSON with these fields.
        """
        
        try:
            response = self.gemini_model.generate_content([prompt, image])
            response_text = response.text
            
            # Extract JSON from response (handling possible text before/after JSON)
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            
            if json_start >= 0 and json_end >= 0:
                json_text = response_text[json_start:json_end+1]
                result = json.loads(json_text)
                return result
            else:
                logging.error("Gemini response did not contain valid JSON")
                return {
                    "secondary_tags": [],
                    "attributes": {},
                    "objects": [],
                    "color_palette": [],
                    "suggested_tags": []
                }
                
        except Exception as e:
            logging.error(f"Error analyzing with Gemini: {e}")
            return {
                "secondary_tags": [],
                "attributes": {},
                "objects": [],
                "color_palette": [],
                "suggested_tags": []
            }
    
    def _save_job(self, job: BatchJob) -> None:
        """Save batch job data.

        Args:
            job: BatchJob object to save
        """
        job_data = job.dict()
        
        if self.storage_bucket:
            # Save to Google Cloud Storage
            blob = self.bucket.blob(f"jobs/{job.job_id}.json")
            blob.upload_from_string(json.dumps(job_data))
        else:
            # Save to local file system
            job_dir = os.path.join(tempfile.gettempdir(), "environmenttagger_jobs")
            os.makedirs(job_dir, exist_ok=True)
            job_path = os.path.join(job_dir, f"{job.job_id}.json")
            
            with open(job_path, 'w') as f:
                json.dump(job_data, f)
    
    def _load_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Load batch job data.

        Args:
            job_id: ID of the job to load

        Returns:
            Dictionary with job data if found, None otherwise
        """
        if self.storage_bucket:
            # Load from Google Cloud Storage
            blob = self.bucket.blob(f"jobs/{job_id}.json")
            if not blob.exists():
                return None
            
            job_data = json.loads(blob.download_as_string())
            return job_data
        else:
            # Load from local file system
            job_dir = os.path.join(tempfile.gettempdir(), "environmenttagger_jobs")
            job_path = os.path.join(job_dir, f"{job_id}.json")
            
            if not os.path.exists(job_path):
                return None
            
            with open(job_path, 'r') as f:
                job_data = json.load(f)
            
            return job_data
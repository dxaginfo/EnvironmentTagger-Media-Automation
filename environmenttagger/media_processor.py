#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment Tagger - Media Processor

This module provides functionality for processing different types of media files.
"""

import os
import tempfile
import logging
from typing import List, Dict, Any, Union, Optional, Tuple
import uuid

import cv2
import numpy as np
from PIL import Image
import requests


class MediaProcessor:
    """Class for processing different types of media files."""

    def __init__(self):
        """Initialize the MediaProcessor."""
        self.supported_image_formats = ['.jpg', '.jpeg', '.png', '.tiff', '.webp']
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
    
    def is_supported_format(self, file_path: str) -> bool:
        """Check if file format is supported.

        Args:
            file_path: Path to the file

        Returns:
            Boolean indicating if format is supported
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_image_formats + self.supported_video_formats
    
    def is_video(self, file_path: str) -> bool:
        """Check if file is a video.

        Args:
            file_path: Path to the file

        Returns:
            Boolean indicating if file is a video
        """
        _, ext = os.path.splitext(file_path.lower())
        return ext in self.supported_video_formats
    
    def extract_frames(self, video_path: str, interval: int = 10) -> List[np.ndarray]:
        """Extract frames from a video at specified intervals.

        Args:
            video_path: Path to the video file
            interval: Interval between frames (in seconds)

        Returns:
            List of frames as numpy arrays
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Unable to open video file: {video_path}")
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_interval = int(fps * interval)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()
        return frames
    
    def download_from_url(self, url: str) -> str:
        """Download media from URL to temporary file.

        Args:
            url: URL of the media file

        Returns:
            Path to the downloaded file
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get file extension from URL or content type
            content_type = response.headers.get('Content-Type', '')
            ext = self._get_extension_from_content_type(content_type)
            
            if not ext and '.' in url.split('/')[-1]:
                ext = os.path.splitext(url.split('/')[-1])[-1]
            
            if not ext:
                ext = '.jpg'  # Default to jpg if extension can't be determined
            
            # Create temporary file with appropriate extension
            temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
            
            with open(temp_file, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            return temp_file
            
        except Exception as e:
            logging.error(f"Error downloading from URL {url}: {e}")
            raise
    
    def preprocess_image(self, image_path: str, target_size: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Preprocess image for analysis.

        Args:
            image_path: Path to the image file
            target_size: Optional target size (width, height)

        Returns:
            Preprocessed image as numpy array
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                raise ValueError(f"Unable to read image: {image_path}")
            
            # Convert from BGR to RGB (standard for most ML models)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Resize if target size specified
            if target_size:
                image = cv2.resize(image, target_size)
            
            return image
            
        except Exception as e:
            logging.error(f"Error preprocessing image {image_path}: {e}")
            raise
    
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract metadata from media file.

        Args:
            file_path: Path to the media file

        Returns:
            Dictionary with metadata
        """
        metadata = {}
        
        try:
            if self.is_video(file_path):
                # Extract video metadata
                cap = cv2.VideoCapture(file_path)
                metadata['width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                metadata['height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                metadata['fps'] = cap.get(cv2.CAP_PROP_FPS)
                metadata['frame_count'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                metadata['duration'] = metadata['frame_count'] / metadata['fps'] if metadata['fps'] > 0 else 0
                cap.release()
            else:
                # Extract image metadata
                with Image.open(file_path) as img:
                    metadata['width'], metadata['height'] = img.size
                    metadata['format'] = img.format
                    metadata['mode'] = img.mode
                    
                    # Extract EXIF data if available
                    if hasattr(img, '_getexif') and img._getexif():
                        exif = {ExifTags.TAGS.get(k, k): v for k, v in img._getexif().items()}
                        metadata['exif'] = exif
        except Exception as e:
            logging.warning(f"Error extracting metadata from {file_path}: {e}")
        
        return metadata
    
    def _get_extension_from_content_type(self, content_type: str) -> str:
        """Get file extension from content type.

        Args:
            content_type: Content type string

        Returns:
            File extension including the dot
        """
        content_type = content_type.lower()
        
        if 'jpeg' in content_type or 'jpg' in content_type:
            return '.jpg'
        elif 'png' in content_type:
            return '.png'
        elif 'tiff' in content_type:
            return '.tiff'
        elif 'webp' in content_type:
            return '.webp'
        elif 'mp4' in content_type or 'mpeg4' in content_type:
            return '.mp4'
        elif 'quicktime' in content_type:
            return '.mov'
        elif 'x-msvideo' in content_type:
            return '.avi'
        elif 'x-matroska' in content_type:
            return '.mkv'
        
        return ''
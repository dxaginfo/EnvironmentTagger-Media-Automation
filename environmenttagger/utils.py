#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment Tagger - Utilities

This module provides utility functions for the EnvironmentTagger application.
"""

import os
import tempfile
import uuid
from typing import List, Dict, Any, Optional
import logging
import requests
import cv2
import numpy as np


def download_media(url: str) -> str:
    """Download media from a URL.

    Args:
        url: URL of the media to download

    Returns:
        Local path to the downloaded media
    """
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        # Try to determine file extension from content type
        content_type = response.headers.get('Content-Type', '')
        if 'image' in content_type:
            ext = '.jpg'
            if 'png' in content_type:
                ext = '.png'
            elif 'tiff' in content_type:
                ext = '.tiff'
            elif 'webp' in content_type:
                ext = '.webp'
        elif 'video' in content_type:
            ext = '.mp4'
            if 'quicktime' in content_type:
                ext = '.mov'
            elif 'x-msvideo' in content_type:
                ext = '.avi'
            elif 'x-matroska' in content_type:
                ext = '.mkv'
        else:
            # Try to get extension from URL
            url_path = url.split('?')[0]  # Remove query parameters
            if '.' in url_path:
                ext = os.path.splitext(url_path)[1].lower()
            else:
                ext = '.dat'  # Generic extension if can't determine
        
        # Create temporary file with appropriate extension
        temp_file = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4()}{ext}")
        
        with open(temp_file, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        return temp_file
    except Exception as e:
        logging.error(f"Error downloading media from {url}: {e}")
        raise


def extract_frames(video_path: str, interval_seconds: int = 10) -> List[np.ndarray]:
    """Extract frames from a video at specified intervals.

    Args:
        video_path: Path to the video file
        interval_seconds: Interval between frames in seconds

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
    if fps <= 0:
        logging.warning(f"Invalid FPS value: {fps}, using default of 30")
        fps = 30
        
    frame_interval = int(fps * interval_seconds)
    
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


def get_file_extension(file_path: str) -> str:
    """Get the file extension from a path.

    Args:
        file_path: Path to the file

    Returns:
        File extension (including the dot)
    """
    if '?' in file_path:
        # Remove query parameters
        file_path = file_path.split('?')[0]
    
    return os.path.splitext(file_path)[1].lower()


def format_timestamp(seconds: float) -> str:
    """Format seconds as HH:MM:SS.MS.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted timestamp string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:06.3f}"


def create_color_palette_image(color_codes: List[str], size: int = 100) -> np.ndarray:
    """Create an image displaying a color palette.

    Args:
        color_codes: List of hex color codes
        size: Size of each color square in pixels

    Returns:
        Image as numpy array
    """
    num_colors = len(color_codes)
    if num_colors == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)
    
    # Create image wide enough for all colors
    palette = np.zeros((size, size * num_colors, 3), dtype=np.uint8)
    
    for i, color_code in enumerate(color_codes):
        # Convert hex to RGB
        hex_code = color_code.lstrip('#')
        rgb = tuple(int(hex_code[j:j+2], 16) for j in (0, 2, 4))
        
        # Fill the region with the color
        palette[:, i*size:(i+1)*size, :] = rgb
    
    return palette
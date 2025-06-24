#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Environment Tagger - Data Models

This module defines the data models used throughout the EnvironmentTagger application.
"""

from enum import Enum
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class AnalysisDepth(str, Enum):
    """Depth of analysis for environment tagging."""
    BASIC = "basic"
    STANDARD = "standard"
    DETAILED = "detailed"


class MediaSource(BaseModel):
    """Source information for media files."""
    type: str = Field(..., description="Type of media source (local, url, drive)")
    path: str = Field(..., description="Path or identifier for the media")


class Environment(BaseModel):
    """Environment information."""
    primary: str = Field(..., description="Primary environment category")
    secondary: List[str] = Field(default_factory=list, description="Secondary environment categories")
    description: Optional[str] = Field(None, description="Detailed environment description")


class LightingInfo(BaseModel):
    """Lighting information for an environment."""
    type: str = Field(..., description="Type of lighting (natural, artificial, mixed)")
    quality: Optional[str] = Field(None, description="Quality of lighting (bright, dim, etc.)")
    source: Optional[str] = Field(None, description="Source of lighting (daylight, tungsten, etc.)")
    direction: Optional[str] = Field(None, description="Direction of lighting (overhead, side, etc.)")


class TagResult(BaseModel):
    """Result of an environment tagging operation."""
    media_id: str = Field(..., description="Identifier for the media")
    analysis_timestamp: str = Field(..., description="Timestamp of the analysis")
    primary_environment: Optional[str] = Field(None, description="Primary environment detected")
    primary_tags: List[str] = Field(default_factory=list, description="Primary environment tags")
    secondary_environments: List[str] = Field(default_factory=list, description="Secondary environments detected")
    secondary_tags: List[str] = Field(default_factory=list, description="Secondary tags for the environment")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Environment attributes")
    objects: List[str] = Field(default_factory=list, description="Objects detected in the environment")
    confidence_score: Optional[float] = Field(None, description="Confidence score for the detection")
    color_palette: List[str] = Field(default_factory=list, description="Color palette of the environment")
    suggested_tags: List[str] = Field(default_factory=list, description="Suggested tags for the environment")
    frame_number: Optional[int] = Field(None, description="Frame number (for video analysis)")
    timestamp: Optional[float] = Field(None, description="Timestamp within video (in seconds)")


class BatchJob(BaseModel):
    """Batch job for processing multiple media files."""
    job_id: str = Field(..., description="Unique identifier for the job")
    media_sources: List[str] = Field(..., description="List of media sources to process")
    analysis_depth: str = Field("standard", description="Depth of analysis")
    extract_frames: bool = Field(False, description="Whether to extract frames from videos")
    frame_interval: int = Field(10, description="Interval between frames to extract (in seconds)")
    status: str = Field("created", description="Status of the job (created, processing, completed, failed)")
    created_at: str = Field(..., description="Timestamp when the job was created")
    started_at: Optional[str] = Field(None, description="Timestamp when the job was started")
    completed_at: Optional[str] = Field(None, description="Timestamp when the job was completed")
    processed_count: int = Field(0, description="Number of media sources processed")
    total_count: int = Field(..., description="Total number of media sources")
    results: Dict[str, Any] = Field(default_factory=dict, description="Results of the job")
    errors: Dict[str, str] = Field(default_factory=dict, description="Errors encountered during processing")
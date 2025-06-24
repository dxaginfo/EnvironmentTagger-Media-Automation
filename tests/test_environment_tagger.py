#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Test suite for EnvironmentTagger module
"""

import os
import json
import pytest
from unittest.mock import patch, MagicMock
import tempfile
from PIL import Image
import numpy as np

# Import module to test
import sys
sys.path.append('..')
from environment_tagger import EnvironmentTagger, MediaProcessingError, AnalysisError

# Test data and fixtures
@pytest.fixture
def sample_config():
    """Return a sample configuration for testing."""
    return {
        "analysis_detail_levels": {
            "basic": ["object_detection", "label_detection", "color_analysis"],
            "standard": ["object_detection", "label_detection", "color_analysis", "scene_detection"],
            "detailed": ["object_detection", "label_detection", "color_analysis", "scene_detection", 
                         "lighting_analysis", "spatial_analysis", "gemini_enhanced"]
        },
        "use_firebase": False,
        "use_custom_models": False,
        "allow_temporary_files": True,
        "temp_directory": tempfile.mkdtemp()
    }

@pytest.fixture
def sample_test_image():
    """Create a sample test image for testing."""
    # Create a simple test image
    img_path = os.path.join(tempfile.mkdtemp(), "test_image.jpg")
    img = Image.new('RGB', (100, 100), color=(73, 109, 137))
    img.save(img_path)
    yield img_path
    # Cleanup
    if os.path.exists(img_path):
        os.remove(img_path)

@pytest.fixture
def mock_vision_response():
    """Mock response from Google Cloud Vision."""
    mock_response = MagicMock()
    
    # Mock label annotations
    mock_label = MagicMock()
    mock_label.description = "office"
    mock_label.score = 0.95
    mock_response.label_annotations = [mock_label]
    
    # Mock object annotations
    mock_object = MagicMock()
    mock_object.name = "desk"
    mock_object.score = 0.92
    mock_response.localized_object_annotations = [mock_object]
    
    # Mock image properties
    mock_color = MagicMock()
    mock_color.color.red = 73
    mock_color.color.green = 109
    mock_color.color.blue = 137
    mock_color.pixel_fraction = 0.8
    mock_response.image_properties_annotation.dominant_colors.colors = [mock_color]
    
    return mock_response

@pytest.fixture
def mock_gemini_response():
    """Mock response from Gemini API."""
    return {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "text": """
                            {
                              "environmentType": ["indoor", "office", "corporate"],
                              "lighting": {
                                "type": "artificial",
                                "direction": "overhead",
                                "intensity": "moderate"
                              },
                              "sceneComposition": {
                                "depth": "shallow",
                                "layout": "symmetrical"
                              },
                              "timeOfDay": "daytime",
                              "suggestedTags": ["office space", "workspace", "corporate environment"],
                              "environmentDescription": "Modern corporate office environment with natural lighting from large windows, predominantly blue and neutral color palette."
                            }
                            """
                        }
                    ]
                }
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 100,
            "candidatesTokenCount": 200
        }
    }

# Test cases
class TestEnvironmentTagger:
    """Test suite for EnvironmentTagger class."""
    
    def test_initialization(self, sample_config):
        """Test initialization of EnvironmentTagger."""
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value=sample_config):
            tagger = EnvironmentTagger(api_key="fake_key")
            assert tagger.api_key == "fake_key"
            assert tagger.use_gemini is True
            assert tagger.use_cloud_vision is True
            assert tagger.db is None
    
    def test_initialization_no_api_key(self, sample_config):
        """Test initialization fails without API key."""
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value=sample_config):
            with pytest.raises(Exception):
                tagger = EnvironmentTagger()
    
    @patch('environment_tagger.vision.ImageAnnotatorClient')
    @patch('environment_tagger.requests.post')
    def test_analyze_image_basic(self, mock_post, mock_vision_client, sample_config, sample_test_image, mock_vision_response):
        """Test basic image analysis functionality."""
        # Setup mocks
        mock_vision_instance = MagicMock()
        mock_vision_instance.annotate_image.return_value = mock_vision_response
        mock_vision_client.return_value = mock_vision_instance
        
        # Create tagger with mocked config
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value=sample_config):
            tagger = EnvironmentTagger(api_key="fake_key")
            
            # Test analysis
            results = tagger.analyze_image(
                sample_test_image,
                detail_level="basic",
                include_palette=True,
                include_objects=True,
                include_scene_type=True
            )
            
            # Verify results
            assert "mediaAnalysis" in results
            assert "objects" in results["mediaAnalysis"]
            assert results["mediaAnalysis"]["objects"][0]["name"] == "desk"
            assert results["mediaAnalysis"]["objects"][0]["confidence"] == 0.92
    
    @patch('environment_tagger.vision.ImageAnnotatorClient')
    @patch('environment_tagger.requests.post')
    def test_analyze_image_detailed_with_gemini(self, mock_post, mock_vision_client, sample_config, sample_test_image, mock_vision_response, mock_gemini_response):
        """Test detailed image analysis with Gemini integration."""
        # Setup mocks
        mock_vision_instance = MagicMock()
        mock_vision_instance.annotate_image.return_value = mock_vision_response
        mock_vision_client.return_value = mock_vision_instance
        
        # Mock Gemini API response
        mock_response = MagicMock()
        mock_response.json.return_value = mock_gemini_response
        mock_response.raise_for_status = MagicMock()
        mock_post.return_value = mock_response
        
        # Create tagger with mocked config
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value=sample_config):
            tagger = EnvironmentTagger(api_key="fake_key")
            
            # Test analysis
            results = tagger.analyze_image(
                sample_test_image,
                detail_level="detailed",
                include_palette=True,
                include_objects=True,
                include_scene_type=True
            )
            
            # Verify results
            assert "mediaAnalysis" in results
            assert "metadataGeneration" in results
            assert "environmentType" in results["mediaAnalysis"]
            assert results["mediaAnalysis"]["environmentType"] == ["indoor", "office", "corporate"]
            assert "lighting" in results["mediaAnalysis"]
            assert results["mediaAnalysis"]["lighting"]["type"] == "artificial"
            assert "suggestedTags" in results["metadataGeneration"]
            assert "office space" in results["metadataGeneration"]["suggestedTags"]
    
    def test_extract_environment_from_labels(self, sample_config):
        """Test environment extraction from labels."""
        # Setup test data
        labels = [
            {"description": "office", "confidence": 0.95},
            {"description": "indoor", "confidence": 0.90},
            {"description": "desk", "confidence": 0.85},
            {"description": "bright", "confidence": 0.80}
        ]
        
        # Create tagger with mocked config and vocabulary
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value=sample_config):
            with patch('environment_tagger.EnvironmentTagger._load_environment_vocabulary', return_value={
                "location_type": ["indoor", "outdoor"],
                "setting": ["office", "home", "kitchen"],
                "lighting": ["bright", "dim", "dark"],
                "atmosphere": ["cozy", "spacious"]
            }):
                tagger = EnvironmentTagger(api_key="fake_key")
                
                # Test extraction
                env_types = tagger._extract_environment_from_labels(labels)
                
                # Verify results
                assert "indoor" in env_types
                assert "office" in env_types
                assert "bright" in env_types
    
    def test_rgb_to_hex(self):
        """Test RGB to HEX color conversion."""
        with patch('environment_tagger.EnvironmentTagger._load_config', return_value={}):
            with patch('environment_tagger.EnvironmentTagger.__init__', return_value=None):
                tagger = EnvironmentTagger()
                tagger.config = {}
                
                # Test conversion
                hex_color = tagger._rgb_to_hex(255, 0, 0)
                assert hex_color == "#ff0000"
                
                hex_color = tagger._rgb_to_hex(0, 255, 0)
                assert hex_color == "#00ff00"
                
                hex_color = tagger._rgb_to_hex(0, 0, 255)
                assert hex_color == "#0000ff"

if __name__ == "__main__":
    pytest.main()
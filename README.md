# EnvironmentTagger: Media Environment Detection & Tagging Tool

EnvironmentTagger is an AI-powered tool that automatically identifies, tags, and categorizes environments in media content. Using computer vision and natural language processing, it analyzes video frames and images to extract detailed environment information, enabling better content organization, search, and production planning.

## 🚀 Features

- **Intelligent Environment Detection**: Analyzes video frames or images to identify shooting environments, locations, and setting details
- **AI-Powered Tagging**: Uses Gemini API to generate comprehensive and context-aware tags for detected environments
- **Media Metadata Enhancement**: Enriches media files with standardized environment metadata
- **Batch Processing**: Supports processing of multiple media files in a single operation
- **Integration Support**: Seamlessly connects with other media automation tools

## 🛠️ Technologies

- **Python** for core processing engine
- **TensorFlow** for custom vision models
- **Google Cloud Vision API** for base image analysis
- **Gemini API** for intelligent content understanding
- **Flask** for API endpoints
- **Firebase** for user authentication and data persistence
- **Google Cloud Storage** for media file handling
- **Google Cloud Functions** for serverless operations

## 📋 Requirements

- Python 3.8+
- Google Cloud account with Vision API and Storage enabled
- Gemini API access
- Firebase project (for user authentication)
- TensorFlow 2.x

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/dxaginfo/EnvironmentTagger-Media-Automation.git
cd EnvironmentTagger-Media-Automation

# Set up a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
export GEMINI_API_KEY="your-gemini-api-key"
```

## 📖 Usage

### Basic Usage

```python
import environmenttagger as et

# Initialize the tagger
tagger = et.EnvironmentTagger(api_key="YOUR_API_KEY")

# Tag a single image
results = tagger.tag_image("path/to/image.jpg")

# Print the detected environment tags
print(results.primary_tags)
print(results.secondary_tags)
print(results.confidence_scores)
```

### Batch Processing

```python
import environmenttagger as et

# Initialize the tagger
tagger = et.EnvironmentTagger(api_key="YOUR_API_KEY")

# Define media sources
media_sources = [
    "path/to/image1.jpg",
    "path/to/video1.mp4",
    "https://example.com/media/image2.jpg",
    "drive://folder_id/image3.png"
]

# Process batch with custom options
batch_job = tagger.create_batch_job(
    media_sources=media_sources,
    analysis_depth="detailed",
    extract_frames=True,
    frame_interval=5
)

# Start processing
batch_job.start()

# Check status
status = batch_job.get_status()
print(f"Processed: {status.processed_count}/{status.total_count}")

# Get results when complete
if status.is_complete:
    results = batch_job.get_results()
    for media_id, tags in results.items():
        print(f"Media {media_id}: {tags}")
```

## 🌐 API Endpoints

### `/analyze`
Process media and generate environment tags
- Method: POST
- Params:
  - `media_source`: File, URL, or Drive ID
  - `analysis_depth`: basic, standard, or detailed
  - `output_format`: json, csv, xml
  - `include_confidence`: boolean

### `/batch`
Process multiple media files
- Method: POST
- Params:
  - `media_sources`: Array of media sources
  - `analysis_depth`: basic, standard, or detailed
  - `output_format`: json, csv, xml

### `/tags/custom`
Define custom tagging templates
- Method: POST
- Params:
  - `template_name`: String
  - `tags`: Array of tag definitions
  - `hierarchy`: Tag relationship map

### `/export`
Export tagging results
- Method: POST
- Params:
  - `job_id`: Analysis job ID
  - `format`: export format
  - `destination`: export destination

## 🔄 Integration

EnvironmentTagger is designed to integrate seamlessly with other media automation tools:

- **SceneValidator**: Provides validated scene data for context-aware tagging
- **TimelineAssembler**: Consumes environment tags for timeline organization
- **StoryboardGen**: Uses environment data to improve storyboard generation
- **ContinuityTracker**: Integrates environment data for continuity checking

## 📊 Sample Output

```json
{
  "media_id": "image1.jpg",
  "analysis_timestamp": "2025-06-24T15:30:45Z",
  "primary_environment": "Interior - Commercial - Office",
  "secondary_environments": ["Meeting Room", "Corporate"],
  "attributes": {
    "lighting": "Artificial - Fluorescent",
    "size": "Medium",
    "occupancy": "Sparse",
    "mood": "Professional",
    "period": "Modern"
  },
  "objects": ["Desk", "Chair", "Computer", "Whiteboard", "Plant"],
  "confidence_score": 0.92,
  "color_palette": ["#F5F5F5", "#0A2463", "#3E92CC", "#D8E1E9", "#B2B2B2"],
  "suggested_tags": ["corporate environment", "workplace", "office space", "meeting area", "business setting"]
}
```

## 🔐 Security Considerations

- OAuth 2.0 authentication for Google API access
- API key management for third-party services
- Input validation to prevent injection attacks
- Rate limiting to prevent abuse
- Data encryption in transit and at rest

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request
# PaksaTalker Examples

This directory contains example scripts that demonstrate how to use the PaksaTalker API for various tasks.

## Getting Started

1. Install the required dependencies:
   ```bash
   pip install requests python-dotenv
   ```

2. Create a `.env` file in the project root with your API credentials:
   ```env
   API_BASE_URL=http://localhost:8000/api/v1
   API_KEY=your_api_key_here
   ```

3. Run any of the example scripts:
   ```bash
   python notebooks/01_quickstart.py
   ```

## Example Scripts

### 1. Quickstart (`01_quickstart.py`)
A simple example that shows how to generate a basic talking head video from an image and text.

**Usage:**
```bash
python notebooks/01_quickstart.py
```

### 2. Advanced Usage (`02_advanced_usage.py`)
A more comprehensive example demonstrating:
- Authentication
- Task status checking
- Error handling
- Downloading results

**Usage:**
```bash
python notebooks/02_advanced_usage.py
```

### 3. Integrated Demo (`03_integrated_demo.py`)
A complete workflow that showcases all PaksaTalker features:
1. Text-to-speech generation with Qwen
2. Talking head video creation with SadTalker
3. Lip-sync enhancement with Wav2Lip
4. Gesture addition based on speech content

**Usage:**
```bash
python notebooks/03_integrated_demo.py
```

## API Reference

### Authentication
All API requests require an API key in the `Authorization` header:
```
Authorization: Bearer YOUR_API_KEY
```

### Endpoints

#### Health Check
```
GET /health
```
Check API status and available models.

#### Generate Video
```
POST /generate/video
```
Generate a talking head video from an image and text/audio.

#### Get Task Status
```
GET /tasks/{task_id}
```
Check the status of a background task.

## Error Handling
All API errors follow this format:
```json
{
  "error": "Error message",
  "code": 400,
  "details": {
    "field": "additional error details"
  }
}
```

## Support
For support, please contact support@paksatalker.com

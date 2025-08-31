# Video Processing Backend

This is the backend service for the Video Processing App, built with FastAPI.

## Features

- Video file upload with size and type validation
- Background video processing
- Progress tracking
- Download processed videos
- RESTful API

## Prerequisites

- Python 3.8+
- pip
- FFmpeg (for video processing)

## Installation

1. Clone the repository
2. Navigate to the backend directory:
   ```bash
   cd backend
   ```
3. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the Server

1. Start the FastAPI server:
   ```bash
   uvicorn app.main:app --reload
   ```
2. The API will be available at `http://localhost:8000`
3. Access the interactive API documentation at `http://localhost:8000/docs`

## API Endpoints

- `POST /api/videos/upload` - Upload a video file
- `POST /api/videos/{video_id}/process` - Start processing a video
- `GET /api/videos/{video_id}/status` - Get video processing status
- `GET /api/videos/{video_id}/download` - Download the processed video

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
# Server configuration
HOST=0.0.0.0
PORT=8000

# File storage
UPLOAD_FOLDER=uploads
PROCESSED_FOLDER=processed

# Security
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

## Development

To run the development server with auto-reload:

```bash
uvicorn app.main:app --reload
```

## Testing

To run the test suite:

```bash
pytest
```

## Deployment

For production deployment, consider using:

- Gunicorn with Uvicorn workers
- Nginx as a reverse proxy
- Environment variables for configuration
- Proper logging and monitoring

## License

MIT

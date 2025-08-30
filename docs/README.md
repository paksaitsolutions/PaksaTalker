# PaksaTalker: Advanced AI-Powered Talking Head Video Generation

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PaksaTalker is an enterprise-grade AI framework for generating hyper-realistic talking head videos with perfect lip-sync, natural facial expressions, and life-like gestures. Built on cutting-edge AI research, it seamlessly integrates multiple state-of-the-art models to deliver production-ready video synthesis.

PaksaTalker is an advanced AI-powered platform that creates hyper-realistic talking avatars with synchronized facial expressions and natural body gestures. The system combines multiple state-of-the-art AI models including Qwen for language processing, SadTalker for facial animation, and PantoMatrix/EMAGE for full-body gesture generation, delivering production-ready video synthesis with unprecedented realism.


## 🌟 Key Features

### 🎭 Natural Animation

- **Precise Lip-Sync**: Frame-accurate audio-visual synchronization
- **Expressive Faces**: Emotionally aware facial animations
- **Natural Gestures**: Context-appropriate head movements and expressions
- **High Fidelity**: 4K resolution support with minimal artifacts

### 🛠️ Technical Capabilities
- Multi-model architecture (SadTalker, Wav2Lip, Qwen)
- GPU-accelerated processing
- Batch processing support
- Real-time preview
- RESTful API for easy integration

### 🧩 Extensible Architecture
- Modular design for easy model swapping
- Plugin system for custom integrations
- Support for custom voice models
- Multi-language support

- **Precise Lip-Sync**: Frame-accurate audio-visual synchronization using SadTalker
- **Expressive Faces**: Emotionally aware facial animations with micro-expressions
- **Full-Body Gestures**: Context-appropriate body language and hand movements
- **High Fidelity**: 4K resolution support with DSLR-quality rendering

### 🛠️ Technical Capabilities
- **Multi-Model Architecture**: Integrates Qwen LLM, SadTalker, and PantoMatrix
- **GPU-Accelerated**: Optimized for NVIDIA GPUs with CUDA support
- **Modular Design**: Swappable components for customization
- **High-Quality Output**: 1080p+ resolution with advanced rendering
- **RESTful API**: Easy integration with existing systems

### 🧩 Extensible Architecture
- **Modular Pipeline**: Independent components for face, body, and voice
- **Custom Avatars**: Support for 3D models and 2D images
- **Plugin System**: Extend with custom models and effects
- **Multi-Language**: Support for multiple languages and accents
- **Customization**: Fine-tune animation styles and rendering parameters


## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA 11.3+ (for GPU acceleration)
- ffmpeg 4.4+
- 8GB+ VRAM recommended

- **Python 3.9+** with pip
- **Node.js 16+** and npm 8+ (for web interface)
- **CUDA 11.8+** (for GPU acceleration)
- **ffmpeg 4.4+** for video processing
- **NVIDIA GPU** with 16GB+ VRAM recommended
- **Docker** (optional, for containerized deployment


### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/paksatalker.git
   cd paksatalker
   ```

2. **Set up Python environment**:
   ```bash
   # Create and activate virtual environment
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   # source venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

3. **Install AI Models**:
   ```bash
   # Download SadTalker models
   python -c "from models.sadtalker import download_models; download_models()"
   
   # Download PantoMatrix/EMAGE models
   python -c "from models.gesture import download_models; download_models()"
   
   # Download Qwen model weights (optional, can use API)
   # python -c "from models.qwen import download_models; download_models()"
   ```

4. **Set up frontend** (for web interface):
   ```bash
   cd frontend
   npm install
   npm run build
   cd ..

   ```

## 🖥️ Quick Start


### Command Line Interface

```bash
# Basic usage
python -m PaksaTalker.cli \
    --image input/face.jpg \
    --audio input/speech.wav \
    --output output/result.mp4 \
    --enhance_face True \
    --expression_intensity 0.8

# Advanced options
python -m PaksaTalker.cli \
    --image input/face.jpg \
    --audio input/speech.wav \
    --output output/result.mp4 \
    --resolution 1080 \
    --fps 30 \
    --background blur \
    --gesture_level medium
```

### Python API

```python
from PaksaTalker import PaksaTalker

### Generate a Talking Avatar from Text

```bash
# Generate speech and animate avatar from text
python -m cli.generate \
    --text "Hello, I'm your AI assistant. Welcome to PaksaTalker!" \
    --image assets/avatars/default.jpg \
    --voice en-US-JennyNeural \
    --output output/welcome.mp4 \
    --gesture-style natural \
    --resolution 1080
```

### Animate with Custom Audio

```bash
# Animate avatar with existing audio
python -m cli.animate \
    --image assets/avatars/presenter.jpg \
    --audio input/presentation.wav \
    --output output/presentation.mp4 \
    --expression excited \
    --background blur \
    --lighting studio
```

### Advanced Options

```bash
# Full pipeline with custom settings
python -m cli.pipeline \
    --prompt "Explain quantum computing in simple terms" \
    --avatar assets/avatars/scientist.jpg \
    --voice en-US-ChristopherNeural \
    --style professional \
    --gesture-level high \
    --output output/quantum_explainer.mp4 \
    --resolution 4k \
    --fps 30 \
    --enhance-face \
    --background office
```

## 🐍 Python API

### Basic Usage

```python
from paksatalker import Pipeline

# Initialize the pipeline
pipeline = Pipeline(
    model_dir="models",
    device="cuda"  # or "cpu" if no GPU
)

# Generate a talking avatar video
result = pipeline.generate(
    text="Welcome to PaksaTalker, the future of digital avatars.",
    image_path="assets/avatars/host.jpg",
    voice="en-US-JennyNeural",
    output_path="output/welcome.mp4",
    gesture_style="casual",
    resolution=1080
)

print(f"Video generated at: {result['output_path']}")
```

### Advanced Usage

```python
from paksatalker import (
    TextToSpeech,
    FaceAnimator,
    GestureGenerator,
    VideoRenderer
)

# Initialize components
tts = TextToSpeech(voice="en-US-ChristopherNeural")
animator = FaceAnimator(model_path="models/sadtalker")
gesture = GestureGenerator(model_path="models/pantomatrix")
renderer = VideoRenderer(resolution=1080, fps=30)

# Process pipeline
text = "Let me show you how this works..."
audio = tts.generate(text)
face_animation = animator.animate("assets/avatars/assistant.jpg", audio)
body_animation = gesture.generate(audio, style="presentation")

# Render final video
video = renderer.combine(
    face_animation=face_animation,
    body_animation=body_animation,
    audio=audio,
    output_path="output/demo.mp4"
)
```

from pathlib import Path

# Initialize with custom settings
pt = PaksaTalker(
    device="cuda",  # or "cpu"
    model_dir="models/",
    temp_dir="temp/"
)

# Generate video with enhanced settings
result = pt.generate(
    image_path="input/face.jpg",
    audio_path="input/speech.wav",
    output_path="output/result.mp4",
    config={
        "resolution": 1080,
        "fps": 30,
        "expression_scale": 0.9,
        "head_pose": "natural",
        "background": {
            "type": "blur",
            "blur_strength": 0.7
        },
        "post_processing": {
            "denoise": True,
            "color_correction": True,
            "stabilization": True
        }
    }
)
```

## 🏗️ Architecture

```
PaksaTalker/
├── api/                  # REST API endpoints
│   ├── routes/          # API route definitions
│   ├── schemas/         # Pydantic models
│   └── utils/           # API utilities
│
├── config/              # Configuration management
│   ├── __init__.py
│   └── config.py
│
├── core/                # Core functionality
│   ├── engine.py       # Main processing pipeline
│   ├── video.py        # Video processing
│   └── audio.py        # Audio processing
│
├── integrations/        # Model integrations
│   ├── sadtalker/      # SadTalker implementation
│   ├── wav2lip/        # Wav2Lip integration
│   ├── qwen/           # Qwen language model
│   └── gesture/        # Gesture generation
│
├── models/             # Model architectures
│   ├── base.py         # Base model interface
│   └── registry.py     # Model registry
│
├── static/             # Static files
│   ├── css/
│   ├── js/
│   └── templates/
│
├── tests/              # Test suite
│   ├── unit/
│   └── integration/
│
├── utils/              # Utility functions
│   ├── audio_utils.py
│   ├── video_utils.py
│   └── face_utils.py
│
├── app.py              # Main application
├── cli.py              # Command-line interface
└── requirements.txt    # Dependencies

## 🏃‍♂️ Usage

### Development Mode

1. **Start the development servers**:
   ```bash
   # In the project root directory
   python run_dev.py
   ```
   This will start:
   - Frontend at http://localhost:5173
   - Backend API at http://localhost:8000
   - API Docs at http://localhost:8000/api/docs

### Production Build

1. **Build the frontend**:
   ```bash
   cd frontend
   npm run build
   cd ..
   ```

2. **Start the production server**:
   ```bash
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```
   The application will be available at http://localhost:8000

### Command Line (Direct API)

```bash
python app.py --input "Hello world" --output output/video.mp4
```

## 🔧 Configuration

PaksaTalker is highly configurable. Here's an example configuration:

```yaml
# config/config.yaml
models:
  sadtalker:
    checkpoint: "models/sadtalker/checkpoints"
    config: "models/sadtalker/configs"

  wav2lip:
    checkpoint: "models/wav2lip/checkpoints"

  qwen:
    model_name: "Qwen/Qwen-7B-Chat"

processing:
  resolution: 1080
  fps: 30
  batch_size: 4
  device: "cuda"

api:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  debug: false
```

## 🧪 Testing

Run the test suite:

```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests
pytest tests/
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

## 📚 Documentation

For detailed documentation, please visit our [Documentation](https://paksatalker.readthedocs.io/).

### Project Structure

```
paksatalker/
├── frontend/           # React + TypeScript frontend
│   ├── src/            # Source files
│   ├── public/         # Static files
│   └── package.json    # Frontend dependencies
├── api/                # API endpoints
├── config/             # Configuration files
├── models/             # AI models
├── static/             # Static files (served by FastAPI)
├── app.py              # Main application entry point
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

### Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# Backend
DEBUG=True
PORT=8000

# Database
DATABASE_URL=sqlite:///./paksatalker.db

# JWT
SECRET_KEY=your-secret-key
ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

For development, you can also create a `.env.development` file in the `frontend` directory.

### API Documentation

Once the server is running, visit `/api/docs` for interactive API documentation (Swagger UI).

For detailed documentation, please visit our [documentation website](https://paksatalker.readthedocs.io).


## 📧 Contact

Project Link: [https://github.com/yourusername/paksatalker](https://github.com/yourusername/paksatalker)

## 🙏 Acknowledgments

- [SadTalker](https://github.com/OpenTalker/SadTalker) - For the amazing talking head generation
- [Wav2Lip](https://github.com/Rudrabha/Wav2Lip) - For lip-sync technology
- [Qwen](https://github.com/QwenLM/Qwen) - For advanced language modeling
- All contributors and open-source maintainers who made this project possible

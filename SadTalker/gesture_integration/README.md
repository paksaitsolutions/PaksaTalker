# SadTalker with Gesture Generation

This project integrates SadTalker's talking head animation with gesture generation capabilities, creating more expressive and natural-looking talking avatars.

## Features

- **Facial Animation**: High-quality talking head animation using SadTalker
- **Gesture Generation**: Natural co-speech gestures using StyleGestures
- **Synchronization**: Seamless coordination between facial expressions and body movements
- **Customization**: Control over style and intensity of gestures

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/OpenTalker/SadTalker.git
   cd SadTalker/gesture_integration
   ```

2. Create and activate a conda environment:
   ```bash
   conda create -n sadgesture python=3.10
   conda activate sadgesture
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Basic Usage

```bash
python -m src.gesture_integrator \
    --image path/to/source/image.jpg \
    --audio path/to/input/audio.wav \
    --output path/to/output/video.mp4
```

### Advanced Options

```bash
python -m src.gesture_integrator \
    --image path/to/source/image.jpg \
    --audio path/to/input/audio.wav \
    --output path/to/output/video.mp4 \
    --config path/to/custom/config.yaml
```

### Configuration

Edit `config/config.yaml` to customize the behavior:

```yaml
# Paths
paths:
  sadtalker_root: "path/to/SadTalker"
  stylegestures_root: "path/to/StyleGestures"
  output_dir: "data/output"
  temp_dir: "data/temp"

# Model parameters
sadtalker:
  checkpoint: "checkpoints/SadTalker_V0.0.2_256.safetensors"
  still: true
  preprocess: "full"
  # ... other parameters

stylegestures:
  model_path: "path/to/stylegestures/checkpoint.pt"
  style: "neutral"  # neutral, excited, calm, etc.
  intensity: 0.5    # 0.0 to 1.0
```

## Project Structure

```
gesture_integration/
├── config/               # Configuration files
│   └── config.yaml       # Main configuration
├── data/                 # Data directories
│   ├── input/            # Input files
│   ├── output/           # Output files
│   └── temp/             # Temporary files
├── src/                  # Source code
│   ├── __init__.py
│   └── gesture_integrator.py  # Main integration script
├── tests/                # Test files
├── requirements.txt      # Python dependencies
└── README.md            # This file
```

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**:
   - Reduce the batch size in the configuration
   - Lower the resolution of the output video

2. **Missing Dependencies**:
   - Make sure all dependencies are installed: `pip install -r requirements.txt`
   - Check for any platform-specific requirements

3. **Model Loading Errors**:
   - Verify the paths to model checkpoints in the config file
   - Ensure the model files are downloaded and accessible

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [SadTalker](https://github.com/OpenTalker/SadTalker) - Talking head animation
- [StyleGestures](https://github.com/simonalexanderson/StyleGestures) - Gesture generation

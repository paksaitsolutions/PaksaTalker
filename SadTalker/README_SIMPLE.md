# SadTalker Simplified Interface

This directory contains simplified scripts to run SadTalker for generating talking head videos.

## Prerequisites

- Python 3.8+
- Required packages (install with `pip install -r requirements.txt`)
- FFmpeg installed and added to system PATH

## Quick Start

1. **Run with default settings**:
   ```bash
   python run_simple.py --source_image examples/source_image/art_0.png --driven_audio examples/driven_audio/RD_Radio31_000.wav
   ```

2. **Custom output directory**:
   ```bash
   python run_simple.py --source_image path/to/your/image.jpg --driven_audio path/to/your/audio.wav --result_dir ./custom_output
   ```

## Advanced Usage

For more control, you can modify the `run_simple.py` script to adjust parameters like:
- `pose_style` (0-45)
- `size` (128, 256, or 512)
- `expression_scale` (0.5-2.0)
- `still` (True/False for head movement)

## Notes

- First run will download necessary models (may take time)
- Processing time depends on video length and system specs
- For best results, use square images (1:1 aspect ratio)

## Troubleshooting

- If you get CUDA errors, make sure to set `device = 'cpu'` in the script
- For memory issues, reduce the `size` parameter
- Ensure all file paths are correct and accessible

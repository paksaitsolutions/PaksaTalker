# SadTalker Cleanup and Setup Guide

This guide will help you clean up your SadTalker installation and set up a clean environment with all necessary dependencies.

## 1. Backup Important Files

Before proceeding with the cleanup, make sure to back up any important files, especially:
- Custom models in the `checkpoints` directory
- Any modified configuration files
- Your own scripts or modifications

## 2. Run the Cleanup Script

Run the cleanup script to remove unnecessary files and directories:

```bash
python cleanup.py
```

This script will:
- Show you a list of files and directories to be removed
- Display the total disk space that will be freed
- Ask for confirmation before deleting anything

## 3. Set Up the Environment

Run the setup script to create a clean Python virtual environment and install all dependencies:

```bash
setup_environment.bat
```

This will:
1. Create a Python virtual environment
2. Install PyTorch with CUDA support (if available)
3. Install all required Python packages
4. Install additional dependencies for PaksaTalker

## 4. Verify the Installation

After setup, verify that everything is working correctly:

```bash
python verify_installation.py
```

This will check:
- Python version compatibility
- Required Python packages
- Essential files and directories
- FFmpeg installation

## 5. Start Using SadTalker

Once everything is set up, you can start using SadTalker with:

```bash
webui.bat
```

## Directory Structure After Cleanup

```
SadTalker/
├── PaksaTalker/           # PaksaTalker integration
│   ├── __init__.py
│   ├── config.py
│   ├── core.py
│   └── integrations/      # Integration modules
├── checkpoints/          # Model checkpoints
├── configs/              # Configuration files
├── gfpgan/               # GFP-GAN for face enhancement
├── src/                  # Source code
├── docs/                 # Documentation
├── cleanup.py            # Cleanup script
├── setup_environment.bat # Environment setup script
├── verify_installation.py # Installation verification
├── inference.py          # Main inference script
├── launcher.py           # Application launcher
├── webui.bat             # Web UI launcher for Windows
├── webui.sh              # Web UI launcher for Linux/macOS
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Troubleshooting

### Missing Dependencies
If you encounter missing dependencies, install them with:

```bash
pip install -r requirements.txt
```

### CUDA Errors
If you get CUDA-related errors, try reinstalling PyTorch with CPU-only support:

```bash
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### FFmpeg Issues
Make sure FFmpeg is installed and added to your system PATH.

## Next Steps

- Check out the [documentation](docs/) for advanced usage
- Try the example scripts in the `examples` directory
- Refer to the [PaksaTalker documentation](PaksaTalker/README.md) for integration details

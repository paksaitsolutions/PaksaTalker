import subprocess
import sys
from pathlib import Path

def generate_video(image_path, audio_path, output_path):
    """Minimal SadTalker video generation."""
    
    # Install SadTalker if needed
    if not Path("SadTalker").exists():
        subprocess.run(["git", "clone", "https://github.com/OpenTalker/SadTalker.git"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-r", "SadTalker/requirements.txt"], check=True)
    
    # Run SadTalker
    cmd = [
        sys.executable, "inference.py",
        "--driven_audio", audio_path,
        "--source_image", image_path,
        "--result_dir", str(Path(output_path).parent),
        "--enhancer", "gfpgan"
    ]
    
    result = subprocess.run(cmd, cwd="SadTalker", capture_output=True, text=True)
    
    if result.returncode == 0:
        # Find generated video
        for f in Path(output_path).parent.glob("*.mp4"):
            f.rename(output_path)
            return str(output_path)
    
    raise Exception(f"Failed: {result.stderr}")
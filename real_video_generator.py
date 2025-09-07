import subprocess
import sys
import os
from pathlib import Path


def _resolve_sadtalker_dir() -> Path:
    """Pick a local SadTalker directory; never clones/installs.

    Honors PAKSA_SADTALKER_DIR if set.
    """
    env = os.getenv('PAKSA_SADTALKER_DIR')
    candidates = []
    if env:
        candidates.append(Path(env))
    candidates.append(Path('SadTalker'))
    candidates.append(Path('PaksaTalker') / 'SadTalker')
    for p in candidates:
        try:
            if p.exists() and (p / 'inference.py').exists():
                return p
        except Exception:
            continue
    raise RuntimeError("SadTalker not found locally. Please place the SadTalker repo at ./SadTalker or set PAKSA_SADTALKER_DIR.")


def generate_real_video(image_path, audio_path, output_path, preprocess: str | None = None):
    """Generate video using local SadTalker CLI.

    This function never performs network operations. It expects a local
    SadTalker checkout with its own dependencies satisfied in the current
    Python environment.
    """

    sadtalker_dir = _resolve_sadtalker_dir()

    # Helper to build command with a given preprocess mode
    def _build_cmd(preprocess: str) -> list[str]:
        cmd = [
            sys.executable, "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", str(Path(output_path).parent),
            "--preprocess", preprocess
        ]
        # Optional face enhancer can be enabled via env if dependencies are installed
        enhancer = os.getenv("PAKSA_FACE_ENHANCER")
        if enhancer:
            cmd.extend(["--enhancer", enhancer])
        return cmd

    # Run SadTalker inference, prefer explicit arg, then env override then default
    preferred = (preprocess or os.getenv("PAKSA_PREPROCESS") or "full").strip().lower()
    if preferred not in ("full", "crop", "resize", "extfull", "extcrop"):
        preferred = "full"
    cmd = _build_cmd(preferred)

    print(f"Running SadTalker at {sadtalker_dir}: {' '.join(cmd)}")
    # No timeout: allow slow CPU runs to complete
    result = subprocess.run(cmd, cwd=str(sadtalker_dir), capture_output=True, text=True)
    print(f"SadTalker stdout: {result.stdout[:2000]}")
    print(f"SadTalker stderr: {result.stderr[:2000]}")

    if result.returncode != 0:
        strict = os.getenv("PAKSA_STRICT_PREPROCESS", "0").strip().lower() in ("1","true","yes")
        if not strict:
            alt = "crop" if preferred != "crop" else "full"
            print(f"SadTalker failed with '{preferred}' preprocess; retrying with '{alt}'.")
            cmd = _build_cmd(alt)
            print(f"Running SadTalker at {sadtalker_dir}: {' '.join(cmd)}")
            # No timeout on retry
            result = subprocess.run(cmd, cwd=str(sadtalker_dir), capture_output=True, text=True)
            print(f"SadTalker stdout: {result.stdout[:2000]}")
            print(f"SadTalker stderr: {result.stderr[:2000]}")

    if result.returncode == 0:
        # SadTalker creates videos in its results folder
        sadtalker_results = sadtalker_dir / "results"
        if sadtalker_results.exists():
            video_files = list(sadtalker_results.rglob("*.mp4"))
            if video_files:
                latest_video = max(video_files, key=lambda x: x.stat().st_mtime)
                import shutil
                shutil.copy2(latest_video, output_path)
                return output_path

        # Fallback: check result_dir
        result_dir = Path(output_path).parent
        for video_file in result_dir.glob("*.mp4"):
            if video_file != Path(output_path):
                video_file.rename(output_path)
                return output_path

    raise Exception(f"SadTalker failed (exit {result.returncode}). Stderr: {result.stderr[:500]}")

"""
Scene Composition Endpoints (Basic)
- Layer management (base + overlay on top)
- Alpha channel support (honors overlay alpha if present)
- Resolution scaling and frame rate control
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from pathlib import Path
import uuid
import os
import subprocess

from config import config

router = APIRouter(prefix="/compose", tags=["composition"])


def _ffmpeg_cmd() -> str:
    return os.environ.get('FFMPEG_BIN', 'ffmpeg')


@router.post("/basic")
async def compose_basic(
    base: UploadFile = File(...),               # image or video
    overlay: UploadFile = File(...),            # image (PNG with alpha) or video
    x: int = Form(0),
    y: int = Form(0),
    overlay_width: Optional[int] = Form(None),
    overlay_height: Optional[int] = Form(None),
    resolution: Optional[str] = Form("720x720"),  # e.g., 1280x720
    fps: Optional[int] = Form(25),
    output_ext: Optional[str] = Form("mp4")      # mp4|webm
):
    """Compose a base layer with an overlay.

    - If overlay has alpha (e.g., PNG RGBA or video with alpha), transparency is preserved.
    - If overlay_width/height provided, overlay is scaled before compositing.
    - Output is scaled to `resolution` and frame rate set to `fps`.
    """
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        base_path = temp_dir / f"{sid}_{base.filename}"
        overlay_path = temp_dir / f"{sid}_{overlay.filename}"

        with open(base_path, 'wb') as f:
            f.write(await base.read())
        with open(overlay_path, 'wb') as f:
            f.write(await overlay.read())

        output_path = out_dir / f"{sid}_composed.{output_ext}"

        # Build filter_complex
        filters = []
        map_video = None
        map_audio = None

        # Input labels: [0:v][0:a] base, [1:v][1:a] overlay (if present)
        overlay_chain = "[1:v]format=rgba"
        if overlay_width and overlay_height:
            overlay_chain += f",scale={overlay_width}:{overlay_height}"
        overlay_chain += "[ovr]"

        filters.append(overlay_chain)
        filters.append(f"[0:v][ovr]overlay={x}:{y}[vout]")

        # Output scaling and fps control
        if resolution:
            filters.append(f"[vout]scale={resolution},fps={fps}[vfinal]")
            map_video = "[vfinal]"
        else:
            filters.append(f"[vout]fps={fps}[vfinal]")
            map_video = "[vfinal]"

        # Prefer base audio if present, else overlay audio, else none
        # We don't enforce audio presence here; ffmpeg will complain if mapped incorrectly.

        fc = ";".join(filters)

        cmd = [
            _ffmpeg_cmd(), '-y',
            '-i', str(base_path),
            '-i', str(overlay_path),
            '-filter_complex', fc,
            '-map', map_video,
        ]

        # Try map base audio if any, otherwise skip audio mapping
        # We will not probe here; simpler path is to let ffmpeg pick default audio stream when present
        # and otherwise omit explicit -map for audio.
        cmd += [
            '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
            '-pix_fmt', 'yuv420p',
            '-preset', 'veryfast',
            '-crf', '20',
            str(output_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {
            'success': True,
            'output': f"/api/download/{output_path.name}",
            'task_id': sid
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


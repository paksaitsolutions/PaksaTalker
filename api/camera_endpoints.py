"""
Camera and Cinematic Effects Endpoints
- Depth of field (global blur)
- Motion blur (frame blending)
- Lens distortion (barrel/pincushion)
- Chromatic aberration (RGB/chroma shift)
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from pathlib import Path
import os
import uuid
import subprocess

from config import config

router = APIRouter(prefix="/camera", tags=["camera-effects"])


def _ffmpeg() -> str:
    return os.environ.get('FFMPEG_BIN', 'ffmpeg')


@router.post("/effects")
async def apply_cinematic_effects(
    video: UploadFile = File(...),
    dof_sigma: float = Form(0.0),                # global blur strength
    motion_blur_frames: int = Form(0),          # tmix frames
    lens_k1: float = Form(0.0),                 # lens correction k1
    lens_k2: float = Form(0.0),                 # lens correction k2
    ca_shift: int = Form(0),                    # chroma shift pixels
    fps: Optional[int] = Form(None),
    resolution: Optional[str] = Form(None),
    output_ext: str = Form("mp4")
):
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        in_path = temp_dir / f"{sid}_{video.filename}"
        with open(in_path, 'wb') as f:
            f.write(await video.read())

        out_path = out_dir / f"{sid}_cine.{output_ext}"

        filters = []
        if abs(lens_k1) > 1e-6 or abs(lens_k2) > 1e-6:
            filters.append(f"lenscorrection=0.5:0.5:{lens_k1}:{lens_k2}")
        if dof_sigma > 1e-3:
            rad = max(1, int(dof_sigma))
            filters.append(f"boxblur={rad}:1")
        if motion_blur_frames and motion_blur_frames > 1:
            frames = max(2, min(15, int(motion_blur_frames)))
            weights = " ".join(["1"] * frames)
            filters.append(f"tmix=frames={frames}:weights='{weights}'")
        if ca_shift:
            s = max(0, min(10, int(ca_shift)))
            filters.append(f"chromashift=cbh={s}:cbv={s}:crh=-{s}:crv=-{s}")
        if resolution:
            filters.append(f"scale={resolution}")
        if fps:
            filters.append(f"fps={int(fps)}")

        vf = ",".join(filters) if filters else None
        cmd = [_ffmpeg(), '-y', '-i', str(in_path)]
        if vf:
            cmd += ['-vf', vf]
        cmd += [
            '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
            '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
            str(out_path)
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


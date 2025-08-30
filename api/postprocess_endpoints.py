"""
Post-Processing Endpoints
- Color grading (curves/eq presets)
- Noise reduction (hqdn3d)
- Sharpening (unsharp)
- Glow/Bloom (blur + blend)
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from pathlib import Path
import os
import uuid
import subprocess

from config import config

router = APIRouter(prefix="/post", tags=["post-processing"])


def _ffmpeg() -> str:
    return os.environ.get('FFMPEG_BIN', 'ffmpeg')


@router.post("/process")
async def process_video(
    video: UploadFile = File(...),
    # Color grading
    grade_preset: str = Form("neutral"),   # neutral|film|vivid|mono
    contrast: float = Form(1.0),
    saturation: float = Form(1.0),
    # Noise reduction
    denoise: float = Form(0.0),             # 0..1 maps to hqdn3d strengths
    # Sharpen
    sharpen: float = Form(0.0),             # 0..1 unsharp amount
    # Glow / bloom
    bloom: float = Form(0.0),               # 0..1 blend strength
    bloom_sigma: float = Form(8.0),         # gaussian blur sigma
    # Output
    fps: Optional[int] = Form(None),
    resolution: Optional[str] = Form(None),
    output_ext: str = Form("mp4")
):
    """Apply visual enhancements to a video using ffmpeg filters."""
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        in_path = temp_dir / f"{sid}_{video.filename}"
        with open(in_path, 'wb') as f:
            f.write(await video.read())

        out_path = out_dir / f"{sid}_post.{output_ext}"

        # Build filter graph
        filters = []

        # Start with labeled source
        src = "[0:v]"
        chain_label = "v0"
        current = f"{src}format=yuv420p[{chain_label}]"
        filters.append(current)

        # Color grading via curves/eq/hue
        next_label = "v1"
        grade_filters = []
        preset = (grade_preset or "neutral").lower()
        if preset == "film":
            grade_filters.append("curves=preset=medium_contrast")
            if saturation != 1.0:
                grade_filters.append(f"hue=s={saturation}")
            grade_filters.append(f"eq=contrast={max(0.5,min(1.5,contrast))}")
        elif preset == "vivid":
            grade_filters.append("curves=preset=strong_contrast")
            grade_filters.append(f"hue=s={max(0.5,min(2.0,saturation))}")
            grade_filters.append(f"eq=contrast={max(0.5,min(1.5,contrast))}")
        elif preset == "mono":
            grade_filters.append("hue=s=0")
            grade_filters.append(f"eq=contrast={max(0.5,min(1.8,contrast))}")
        else:
            # neutral: just apply simple eq/hue if changed
            if abs(contrast-1.0) > 1e-3:
                grade_filters.append(f"eq=contrast={contrast}")
            if abs(saturation-1.0) > 1e-3:
                grade_filters.append(f"hue=s={saturation}")

        if grade_filters:
            filters.append(f"[{chain_label}]{','.join(grade_filters)}[{next_label}]")
            chain_label = next_label
            next_label = "v2"

        # Denoise
        if denoise > 1e-3:
            # Map 0..1 to moderate strengths
            s = max(0.0, min(1.0, denoise))
            lsp = round(1.5 + 4.0 * s, 2)
            csp = round(1.0 + 3.0 * s, 2)
            ltmp = round(3.0 + 6.0 * s, 2)
            ctmp = round(2.0 + 4.0 * s, 2)
            filters.append(f"[{chain_label}]hqdn3d={lsp}:{csp}:{ltmp}:{ctmp}[{next_label}]")
            chain_label = next_label
            next_label = "v3"

        # Sharpen
        if sharpen > 1e-3:
            amt = max(0.0, min(2.0, sharpen*2.0))
            filters.append(f"[{chain_label}]unsharp=5:5:{amt}[{next_label}]")
            chain_label = next_label
            next_label = "v4"

        # Bloom/glow: split, blur, blend (screen-like)
        if bloom > 1e-3:
            b = max(0.0, min(1.0, bloom))
            sigma = max(1.0, min(50.0, bloom_sigma))
            filters.append(f"[{chain_label}]split[a][b]")
            filters.append(f"[b]gblur=sigma={sigma}[bb]")
            # Use blend with screen-like effect via lighten + opacity
            filters.append(f"[a][bb]blend=all_mode='screen':all_opacity={b}[{next_label}]")
            chain_label = next_label
            next_label = "v5"

        # Output fps/scale
        if resolution or fps:
            ops = []
            if resolution:
                ops.append(f"scale={resolution}")
            if fps:
                ops.append(f"fps={int(fps)}")
            filters.append(f"[{chain_label}]{','.join(ops)}[vout]")
            vmap = "[vout]"
        else:
            vmap = f"[{chain_label}]"

        fc = ";".join(filters)

        cmd = [_ffmpeg(), '-y', '-i', str(in_path), '-filter_complex', fc, '-map', vmap,
               '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
               '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
               str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


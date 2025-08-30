"""
Background Processing Endpoints
- Green screen removal (chroma key) with optional background composite
- Virtual sets (place keyed/alpha video over background)
- Background blur (with optional key/mask)
- Environment mapping (simple color/exposure alignment)
"""

from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from typing import Optional
from pathlib import Path
import os
import uuid
import subprocess

from config import config

router = APIRouter(prefix="/background", tags=["background-processing"])


def _ffmpeg() -> str:
    return os.environ.get('FFMPEG_BIN', 'ffmpeg')


@router.post("/green-screen")
async def green_screen(
    video: UploadFile = File(...),
    bg_image: Optional[UploadFile] = File(None),
    bg_video: Optional[UploadFile] = File(None),
    key_color: str = Form("0x00FF00"),          # hex like 0x00FF00
    similarity: float = Form(0.20),              # 0..1
    blend: float = Form(0.00),                   # 0..1
    x: int = Form(0), y: int = Form(0),
    resolution: Optional[str] = Form(None),
    fps: Optional[int] = Form(None),
    output_ext: str = Form("mp4")
):
    """Remove green screen using chromakey and optionally composite over a background image/video."""
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        in_path = temp_dir / f"{sid}_{video.filename}"
        with open(in_path, 'wb') as f:
            f.write(await video.read())

        bg_path = None
        if bg_image is not None:
            bg_path = temp_dir / f"{sid}_{bg_image.filename}"
            with open(bg_path, 'wb') as f:
                f.write(await bg_image.read())
        elif bg_video is not None:
            bg_path = temp_dir / f"{sid}_{bg_video.filename}"
            with open(bg_path, 'wb') as f:
                f.write(await bg_video.read())

        out_path = out_dir / f"{sid}_keyed.{output_ext}"

        filters = []
        # Key the input producing alpha
        # chromakey=color:similarity:blend
        filters.append(f"[0:v]chromakey={key_color}:{similarity}:{blend},format=rgba[fg]")

        # If background available, overlay; else export RGBA over black
        if bg_path:
            # Scale background to match resolution if requested
            if resolution:
                filters.append(f"[1:v]scale={resolution}[bg]")
                overlay_bg = "[bg]"
            else:
                overlay_bg = "[1:v]"
            filters.append(f"{overlay_bg}[fg]overlay={x}:{y}:format=auto[vout]")
            vmap = "[vout]"
        else:
            vmap = "[fg]"

        # Add fps/scale controls last if needed
        if fps and resolution:
            filters.append(f"{vmap},fps={int(fps)},scale={resolution}[vfinal]")
            vmap = "[vfinal]"
        elif fps:
            filters.append(f"{vmap},fps={int(fps)}[vfinal]")
            vmap = "[vfinal]"
        elif resolution:
            filters.append(f"{vmap},scale={resolution}[vfinal]")
            vmap = "[vfinal]"

        fc = ";".join(filters)

        cmd = [_ffmpeg(), '-y', '-i', str(in_path)]
        if bg_path:
            cmd += ['-i', str(bg_path)]
        cmd += ['-filter_complex', fc, '-map', vmap,
                '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
                '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
                str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/virtual-set")
async def virtual_set(
    subject: UploadFile = File(...),            # video with alpha or green screen
    background: UploadFile = File(...),
    x: int = Form(0), y: int = Form(0),
    subject_width: Optional[int] = Form(None),
    subject_height: Optional[int] = Form(None),
    key_color: Optional[str] = Form(None),      # if provided, key the subject first
    similarity: float = Form(0.20),
    blend: float = Form(0.00),
    resolution: Optional[str] = Form(None),
    output_ext: str = Form("mp4")
):
    """Place the subject over a virtual set background. If key_color provided, performs keying first."""
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        sub_path = temp_dir / f"{sid}_{subject.filename}"
        bg_path = temp_dir / f"{sid}_{background.filename}"
        with open(sub_path, 'wb') as f:
            f.write(await subject.read())
        with open(bg_path, 'wb') as f:
            f.write(await background.read())

        out_path = out_dir / f"{sid}_virtualset.{output_ext}"

        filters = []
        # Optionally key subject
        if key_color:
            filters.append(f"[0:v]chromakey={key_color}:{similarity}:{blend},format=rgba[sub]")
        else:
            filters.append("[0:v]format=rgba[sub]")

        # Scale subject if needed
        if subject_width and subject_height:
            filters.append(f"[sub]scale={subject_width}:{subject_height}[sub2]")
            sub_label = "[sub2]"
        else:
            sub_label = "[sub]"

        # Scale background if resolution requested
        if resolution:
            filters.append(f"[1:v]scale={resolution}[bg]")
            bg_label = "[bg]"
        else:
            bg_label = "[1:v]"

        filters.append(f"{bg_label}{sub_label}overlay={x}:{y}:format=auto[vout]")
        fc = ";".join(filters)

        cmd = [_ffmpeg(), '-y', '-i', str(sub_path), '-i', str(bg_path),
               '-filter_complex', fc, '-map', '[vout]',
               '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
               '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
               str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/blur")
async def background_blur(
    video: UploadFile = File(...),
    strength: int = Form(10),                    # blur radius
    mask: Optional[UploadFile] = File(None),     # optional alpha/grayscale mask: 255 keep sharp, 0 blur
    resolution: Optional[str] = Form(None),
    output_ext: str = Form("mp4")
):
    """Apply background blur. If a mask is provided, preserves subject areas; otherwise blurs the whole frame."""
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        in_path = temp_dir / f"{sid}_{video.filename}"
        with open(in_path, 'wb') as f:
            f.write(await video.read())

        mask_path = None
        if mask is not None:
            mask_path = temp_dir / f"{sid}_{mask.filename}"
            with open(mask_path, 'wb') as f:
                f.write(await mask.read())

        out_path = out_dir / f"{sid}_blur.{output_ext}"

        filters = []
        if mask_path:
            # Create blurred background from source, then use mask to composite
            filters.append(f"[0:v]boxblur={max(1,int(strength))}:1[blur]")
            filters.append(f"[0:v][blur][1:v]alphamerge,overlay,format=yuv420p[vout]")
            vmap = "[vout]"
            cmd = [_ffmpeg(), '-y', '-i', str(in_path), '-i', str(mask_path)]
        else:
            filters.append(f"[0:v]boxblur={max(1,int(strength))}:1[vout]")
            vmap = "[vout]"
            cmd = [_ffmpeg(), '-y', '-i', str(in_path)]

        if resolution:
            filters.append(f"{vmap},scale={resolution}[vfinal]")
            vmap = "[vfinal]"

        fc = ";".join(filters)

        cmd += ['-filter_complex', fc, '-map', vmap,
                '-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
                '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
                str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/env-map")
async def environment_map(
    video: UploadFile = File(...),
    temperature_k: int = Form(6500),
    tint: float = Form(0.0),                      # -1..1 (mapped via hue)
    exposure: float = Form(0.0),
    saturation: float = Form(1.0),
    contrast: float = Form(1.0),
    output_ext: str = Form("mp4")
):
    """Simple environment mapping by aligning color temperature/tint and exposure/contrast/saturation."""
    try:
        temp_dir = Path(config.get('paths.temp', 'temp'))
        out_dir = Path(config.get('paths.output', 'output'))
        temp_dir.mkdir(exist_ok=True)
        out_dir.mkdir(exist_ok=True)

        sid = str(uuid.uuid4())
        in_path = temp_dir / f"{sid}_{video.filename}"
        with open(in_path, 'wb') as f:
            f.write(await video.read())

        out_path = out_dir / f"{sid}_env.{output_ext}"

        filters = []
        if temperature_k != 6500:
            filters.append(f"colortemperature=t={int(temperature_k)}")
        if abs(tint) > 1e-3:
            # hue=h: degrees, we map tint -1..1 to -10..10 degrees
            hdeg = int(max(-10, min(10, tint * 10)))
            filters.append(f"hue=h={hdeg}*PI/180")
        if any([abs(exposure) > 1e-3, abs(contrast - 1.0) > 1e-3, abs(saturation - 1.0) > 1e-3]):
            filters.append(f"eq=brightness={exposure}:contrast={contrast}")
            filters.append(f"hue=s={saturation}")

        vf = ",".join(filters) if filters else None
        cmd = [_ffmpeg(), '-y', '-i', str(in_path)]
        if vf:
            cmd += ['-vf', vf]
        cmd += ['-c:v', 'libx264' if output_ext == 'mp4' else 'libvpx-vp9',
                '-pix_fmt', 'yuv420p', '-preset', 'veryfast', '-crf', '20',
                str(out_path)]

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=240)
        if result.returncode != 0:
            raise RuntimeError(result.stderr or 'ffmpeg failed')

        return {"success": True, "output": f"/api/download/{out_path.name}", "task_id": sid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


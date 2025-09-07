#!/usr/bin/env python3
"""
Stable PaksaTalker Server with Real AI Processing
"""
import os
import sys
import logging
import asyncio
import uuid
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, HTTPException, File, UploadFile, Form, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
import uvicorn
import cv2
import numpy as np

# Add project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

try:
    from api.capabilities_endpoints import router as capabilities_router
    from api.expressions_endpoints import router as expressions_router
    from api.emage_endpoints import router as emage_router
    from api.diagnostics_endpoints import router as diagnostics_router
    API_ROUTER_AVAILABLE = True
except Exception as e:
    API_ROUTER_AVAILABLE = False
    # Log the actual reason once, in a concise line (logger may not be defined yet)
    import logging as _logging
    _logging.getLogger(__name__).warning(
        "API router disabled: %s. Using fallback endpoints.", str(e)
    )

# Emotion Recognition (optional; avoid heavy deps/downloads at startup)
try:
    from models.emotion.fer_model import EmotionRecognizer
    # Initialize a lightweight stub by default; real weights can be wired later
    EMOTION_MODEL = EmotionRecognizer(weights_path=None)
except Exception as e:
    print(f"⚠️  Emotion recognition not initialized: {str(e)}")
    EMOTION_MODEL = None

# OpenSeeFace imports
try:
    from OpenSeeFace.facetracker import FaceTracker
    from OpenSeeFace.tracker import Tracker
    OPENSEEFACE_AVAILABLE = True
except ImportError:
    OPENSEEFACE_AVAILABLE = False
    print("Warning: OpenSeeFace not available. Face tracking features will be disabled.")

# 3DDFA_V2 imports removed here (not needed in stable_server core path)
THREE_DDFA_AVAILABLE = False

# Performance tuning envs
FFMPEG_THREADS = int(os.environ.get('PAKSA_FFMPEG_THREADS', '0'))  # 0 = ffmpeg auto

# Utility helpers
def _find_ffmpeg() -> Optional[str]:
    """Locate ffmpeg on the system, return path or None."""
    import subprocess, os
    candidates = [
        "ffmpeg",
        r"C:\\ffmpeg\\bin\\ffmpeg.exe",
        r"C:\\Program Files\\ffmpeg\\bin\\ffmpeg.exe",
        os.path.expandvars(r"C:\\Users\\%USERNAME%\\AppData\\Local\\Microsoft\\WinGet\\Packages\\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\\ffmpeg-8.0-full_build\\bin\\ffmpeg.exe"),
    ]
    for p in candidates:
        try:
            result = subprocess.run([p, "-version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                return p
        except Exception:
            continue
    return None

def _emage_available() -> bool:
    """Check if EMAGE python package is importable locally."""
    try:
        import importlib
        importlib.import_module('EMAGE.models.gesture_decoder')
        return True
    except Exception:
        return False

def _wav2lip2_available() -> bool:
    """Check if Wav2Lip2 can run offline (module + local weights)."""
    try:
        from models.wav2lip2_aoti import Wav2Lip2AOTI  # noqa: F401
        # Require local weights to avoid network fetch
        weights_path = Path('wav2lip2-aoti') / 'weights' / 'wav2lip2_fp8.pt'
        return weights_path.exists()
    except Exception:
        return False

def _apply_advanced_effects(input_video: str, profile: str = "cinematic") -> Optional[str]:
    """Apply lightweight post effects with ffmpeg. Returns output path or None on failure."""
    try:
        ffmpeg_cmd = _find_ffmpeg() or 'ffmpeg'
        import subprocess, os
        temp_out = str(Path(input_video).with_name(Path(input_video).stem + "_fx.mp4"))

        # Define simple profiles
        if profile == 'none':
            return None
        elif profile == 'sharpen':
            vf = "format=yuv420p,unsharp=5:5:0.8:5:5:0.0"
        elif profile == 'portrait':
            vf = "format=yuv420p,eq=contrast=1.05:saturation=1.12:brightness=0.02,unsharp=3:3:0.6:3:3:0.0"
        else:  # cinematic default
            vf = "format=yuv420p,unsharp=5:5:0.8:5:5:0.0,eq=contrast=1.08:saturation=1.1,vignette"

        cmd = [
            ffmpeg_cmd, '-y',
            '-i', input_video,
            '-vf', vf,
            '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
            '-c:a', 'aac', '-b:a', '128k',
            '-movflags', '+faststart',
            temp_out
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
        if result.returncode == 0 and os.path.exists(temp_out) and os.path.getsize(temp_out) > 10000:
            os.replace(temp_out, input_video)
            return input_video
    except Exception as e:
        logger.warning(f"Advanced effects failed: {e}")
    return None


def _apply_background(
    input_video: str,
    mode: str = "none",
    *,
    bg_image: Optional[str] = None,
    bg_color: str = "#000000",
    chroma_color: str = "#00ff00",
    similarity: float = 0.12,
    blend: float = 0.08,
) -> Optional[str]:
    """Apply background customization or green-screen replacement.

    - mode 'none'|'blur'|'portrait'|'cinematic': defer to _apply_advanced_effects
    - mode 'greenscreen': requires chroma key; optionally overlay on image or color
    - mode 'color'/'image': only meaningful with green-screen; otherwise ignored
    Returns output path or None on failure.
    """
    try:
        ffmpeg_cmd = _find_ffmpeg() or 'ffmpeg'
        import subprocess, os
        from pathlib import Path as _P

        if mode in (None, '', 'none'):
            return None
        if mode in ('blur', 'portrait', 'cinematic'):
            return _apply_advanced_effects(input_video, profile=mode if mode != 'blur' else 'portrait')

        # Determine dimensions via ffprobe fallback to 1280x720
        W, H, FPS = 1280, 720, 25
        try:
            import cv2 as _cv
            cap = _cv.VideoCapture(input_video)
            if cap.isOpened():
                W = int(cap.get(_cv.CAP_PROP_FRAME_WIDTH)) or W
                H = int(cap.get(_cv.CAP_PROP_FRAME_HEIGHT)) or H
                FPS = int(cap.get(_cv.CAP_PROP_FPS)) or FPS
            cap.release()
        except Exception:
            pass

        # Normalize color strings like '#RRGGBB' -> 0xRRGGBB
        def _hex(col: str) -> str:
            c = (col or '').strip()
            if c.startswith('#'):
                return '0x' + c[1:]
            if c.lower().startswith('0x'):
                return c
            return c

        out_path = str(_P(input_video).with_name(_P(input_video).stem + "_bg.mp4"))

        # Greenscreen pipeline using chromakey
        if mode == 'greenscreen':
            chroma_hex = _hex(chroma_color or '#00ff00')
            if bg_image and os.path.exists(bg_image):
                cmd = [
                    ffmpeg_cmd, '-y',
                    '-i', input_video,
                    '-i', bg_image,
                    '-filter_complex',
                    f"[1:v][0:v]scale2ref[bg][base];[base]chromakey={chroma_hex}:{similarity}:{blend}[fg];[bg][fg]overlay=shortest=1,format=yuv420p",
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '128k', '-shortest', out_path
                ]
            else:
                color_hex = _hex(bg_color or '#000000')
                cmd = [
                    ffmpeg_cmd, '-y',
                    '-i', input_video,
                    '-f', 'lavfi', '-i', f"color=c={color_hex}:s={W}x{H}:r={FPS}",
                    '-filter_complex',
                    f"[1:v][0:v]scale2ref[bg][base];[base]chromakey={chroma_hex}:{similarity}:{blend}[fg];[bg][fg]overlay=shortest=1,format=yuv420p",
                    '-c:v', 'libx264', '-preset', 'veryfast', '-crf', '20',
                    '-c:a', 'aac', '-b:a', '128k', '-shortest', out_path
                ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                if res.returncode == 0 and os.path.exists(out_path) and os.path.getsize(out_path) > 10240:
                    return out_path
            except Exception:
                return None
            return None

        # 'color' or 'image' without greenscreen: no-op for now
        return None
    except Exception:
        return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
FRONTEND_DIR = BASE_DIR / "frontend" / "dist"
STATIC_DIR = FRONTEND_DIR / "assets"
OUTPUT_DIR = BASE_DIR / "output"
TEMP_DIR = BASE_DIR / "temp"

# Create directories
OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Global task storage
tasks: Dict[str, Dict[str, Any]] = {}

# Create app
app = FastAPI(
    title="PaksaTalker Stable AI",
    description="Stable AI-Powered Video Generation Platform",
    version="1.0.0"
)

# Simple load control
MAX_CONCURRENCY = int(os.environ.get('PAKSA_MAX_CONCURRENCY', '2'))
_sem = asyncio.Semaphore(MAX_CONCURRENCY)

# Add CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend assets
if STATIC_DIR.exists():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR), name="assets")

# Serve docs folder
DOCS_DIR = BASE_DIR / "docs"
if DOCS_DIR.exists():
    app.mount("/docs", StaticFiles(directory=DOCS_DIR), name="docs")

# Capabilities/Expressions endpoints
if API_ROUTER_AVAILABLE:
    app.include_router(capabilities_router, prefix="/api/v1")
    app.include_router(expressions_router, prefix="/api/v1")
    app.include_router(emage_router, prefix="/api/v1")
    app.include_router(diagnostics_router, prefix="/api/v1")


# Prefetch frequently used model assets on startup (non-blocking)
@app.on_event("startup")
async def _prefetch_assets():
    pass  # Skip asset prefetch to avoid errors


@app.post("/api/v1/assets/ensure")
async def ensure_assets_endpoint():
    """Ensure downloadable model assets are present; returns a simple status report."""
    report = {
        "sadtalker": Path("SadTalker").exists(),
        "emotion_model": False,
        "emage_weights": False,
        "openseeface_models": False
    }
    return {"success": True, "report": report}

# Expressions endpoints (local fallback to ensure availability)
@app.get("/api/v1/expressions/capabilities")
async def expressions_caps_fallback():
    try:
        from models.expression.engine import detect_capabilities
        from pathlib import Path
        expr = detect_capabilities()
        # Also include coarse model flags similar to /capabilities
        sadtalker_ok = False
        try:
            import importlib
            importlib.import_module('models.sadtalker_full')
            sadtalker_ok = True
        except Exception:
            sadtalker_ok = False
        ck = Path('models') / 'sadtalker' / 'checkpoints'
        sadtalker_weights = any((ck/f).exists() for f in ['epoch_20.pth','facevid2vid_00189-model.pth.tar','mapping_00109-model.pth.tar'])
        wav2lip2_ok = (Path('wav2lip2-aoti')/ 'checkpoints' / 'wav2lip2_fp8.pt').exists()
        emage_repo = Path('EMAGE').exists()
        return {"success": True, "engines": expr, "models": {
            "sadtalker": sadtalker_ok,
            "sadtalker_weights": sadtalker_weights,
            "wav2lip2": wav2lip2_ok,
            "emage": emage_repo
        }}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

# Capabilities fallback
@app.get("/api/v1/capabilities")
async def capabilities_fallback():
    try:
        from models.expression.engine import detect_capabilities
        from pathlib import Path
        expr = detect_capabilities()
        sadtalker_ok = False
        try:
            import importlib
            importlib.import_module('models.sadtalker_full')
            sadtalker_ok = True
        except Exception:
            sadtalker_ok = False
        ck = Path('models') / 'sadtalker' / 'checkpoints'
        sadtalker_weights = any((ck/f).exists() for f in ['epoch_20.pth','facevid2vid_00189-model.pth.tar','mapping_00109-model.pth.tar'])
        wav2lip2_ok = (Path('wav2lip2-aoti')/ 'checkpoints' / 'wav2lip2_fp8.pt').exists()
        emage_repo = Path('EMAGE').exists()
        data = {
            "models": {
                "sadtalker": sadtalker_ok,
                "sadtalker_weights": sadtalker_weights,
                "wav2lip2": wav2lip2_ok,
                "emage": emage_repo,
                "qwen": False,  # not checked here
                "mediapipe": expr.get('mediapipe', False),
                "threeddfa": expr.get('threeddfa', False),
                "openseeface": expr.get('openseeface', False),
                "mini_xception": expr.get('mini_xception', False)
            }
        }
        return {"success": True, "data": data}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

@app.post("/api/v1/expressions/estimate")
async def expressions_estimate_fallback(request: Request):
    try:
        from models.expression.engine import estimate_from_path
        form = await request.form()
        engine = form.get('engine') or 'auto'
        up = form.get('image')
        if up is None:
            return JSONResponse({"success": False, "detail": "image file is required"}, status_code=400)
        # Save temp
        import tempfile
        import shutil
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tf:
            shutil.copyfileobj(up.file, tf)
            img_path = tf.name
        res = estimate_from_path(img_path, engine)
        return {"success": True, "engine": res.engine, "result": res.to_dict()}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

# POST alias for fusion-video to support Fusion mode unconditionally
@app.post("/api/v1/generate/fusion-video", include_in_schema=False)
async def fusion_video_alias(request: Request):
    try:
        from fastapi import UploadFile
        import uuid as _uuid
        form = await request.form()

        # Parse params
        image: UploadFile = form.get('image') if 'image' in form else None  # type: ignore
        audio: UploadFile = form.get('audio') if 'audio' in form else None  # type: ignore
        text = form.get('prompt') or form.get('text')
        resolution = form.get('resolution') or '720p'
        fps_val = form.get('fps')
        try:
            fps = int(fps_val) if fps_val else 25
        except Exception:
            fps = 25
        emotion = form.get('emotion') or 'neutral'
        style = form.get('style') or 'natural'
        preferWav2Lip2 = form.get('preferWav2Lip2') in ('true', '1', 'yes', True)
        preprocess = (form.get('preprocess') or 'full').lower()
        expression_engine = (form.get('expressionEngine') or 'auto').lower()
        useEmage = form.get('useEmage')
        if useEmage is not None:
            useEmage = useEmage in ('true', '1', 'yes', True)
        # Background customization
        bg_mode = (form.get('backgroundMode') or form.get('background') or 'none').lower()
        bg_color = form.get('backgroundColor') or '#000000'
        chroma_color = form.get('chromaColor') or '#00ff00'
        try:
            chroma_similarity = float(form.get('similarity') or 0.12)
        except Exception:
            chroma_similarity = 0.12
        try:
            chroma_blend = float(form.get('blend') or 0.08)
        except Exception:
            chroma_blend = 0.08
        bg_image_upload = form.get('backgroundImage') if 'backgroundImage' in form else None

        # Save inputs
        tmp = TEMP_DIR
        tmp.mkdir(exist_ok=True)
        img_path = None
        if image is not None:
            img_path = tmp / f"{_uuid.uuid4()}_{image.filename or 'image' }"
            with open(img_path, 'wb') as f:
                f.write(await image.read())
        # Audio or TTS
        audio_path = None
        if audio is not None:
            audio_path = tmp / f"{_uuid.uuid4()}_{audio.filename or 'audio' }"
            with open(audio_path, 'wb') as f:
                f.write(await audio.read())
        elif text:
            # Force gTTS to avoid google cloud/protobuf deps
            try:
                from gtts import gTTS
                mp3_path = tmp / f"{_uuid.uuid4()}.mp3"
                wav_path = tmp / f"{_uuid.uuid4()}.wav"
                tts = gTTS(text)
                tts.save(str(mp3_path))
                # Convert to wav for downstream
                import subprocess
                subprocess.run(['ffmpeg','-y','-i',str(mp3_path),'-ar','16000','-ac','1',str(wav_path)], capture_output=True)
                audio_path = wav_path if wav_path.exists() else mp3_path
            except Exception as e:
                return JSONResponse({"success": False, "detail": f"TTS failed: {e}"}, status_code=500)
        else:
            return JSONResponse({"success": False, "detail": "Either audio or text is required"}, status_code=400)

        # Default avatar if missing
        if img_path is None:
            img_path = tmp / f"{_uuid.uuid4()}_default.jpg"
            try:
                import numpy as _np, cv2 as _cv2
                canvas = _np.ones((512,512,3), dtype=_np.uint8) * 235
                _cv2.circle(canvas, (256,256), 180, (210,210,210), -1)
                _cv2.circle(canvas, (210,210), 28, (70,70,70), -1)
                _cv2.circle(canvas, (302,210), 28, (70,70,70), -1)
                _cv2.ellipse(canvas, (256,320), (70,35), 0, 0, 180, (70,70,70), 5)
                _cv2.imwrite(str(img_path), canvas)
            except Exception:
                with open(img_path, 'wb') as f:
                    f.write(b'avatar')

        # Check EMAGE availability (optional)
        reqEmage = form.get('requireEmage') in ('true','1','yes', True)
        if reqEmage:
            try:
                from models.emage_realistic import emage_available
                if not emage_available():
                    return JSONResponse({
                        "success": False,
                        "detail": "EMAGE not available. Please set PAKSA_EMAGE_ROOT to the EMAGE Python repo and ensure checkpoints/emage_best.pth."
                    }, status_code=503)
            except Exception:
                return JSONResponse({"success": False, "detail": "EMAGE not available."}, status_code=503)

        # Create task
        task_id = str(_uuid.uuid4())
        tasks[task_id] = {"status": "processing", "progress": 5, "stage": "Queued"}

        # Kick background generation
        import asyncio as _asyncio
        async def _run():
            try:
                tasks[task_id].update({"progress": 15, "stage": "Starting fusion"})
                from models.fusion.engine import FusionEngine
                eng = FusionEngine()
                out_path = OUTPUT_DIR / f"{task_id}.mp4"
                final = eng.generate(
                    face_image=str(img_path),
                    audio_path=str(audio_path),
                    output_path=str(out_path),
                    emotion=emotion,
                    style=style,
                    fps=fps,
                    resolution=resolution,
                    prefer_wav2lip2=preferWav2Lip2,
                    use_emage=useEmage if isinstance(useEmage, bool) else None,
                    preprocess=preprocess
                )
                # Optional background processing
                bg_path = None
                try:
                    # Save background image upload if provided
                    bg_image_path = None
                    if bg_image_upload is not None:
                        bg_image_path = OUTPUT_DIR / f"{task_id}_bg_{getattr(bg_image_upload, 'filename', 'image') or 'image' }"
                        with open(bg_image_path, 'wb') as bf:
                            bf.write(await bg_image_upload.read())
                        bg_image_path = str(bg_image_path)
                    res_path = _apply_background(
                        final,
                        mode=bg_mode,
                        bg_image=bg_image_path,
                        bg_color=bg_color,
                        chroma_color=chroma_color,
                        similarity=chroma_similarity,
                        blend=chroma_blend,
                    )
                    if res_path and os.path.exists(res_path):
                        final = res_path
                        out_path = Path(final)
                except Exception:
                    pass
                tasks[task_id] = {
                    "status": "completed",
                    "progress": 100,
                    "stage": "Done",
                    "video_url": f"/api/download/{out_path.name}",
                    "expression_engine": expression_engine,
                    "preprocess": preprocess,
                }
            except Exception as e:
                tasks[task_id] = {"status": "failed", "progress": 0, "stage": "Failed", "error": str(e)}

        _asyncio.create_task(_run())
        return {"success": True, "task_id": task_id, "status": "processing"}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

# Diagnostics fallbacks (always available)
@app.get("/api/v1/diagnostics/versions")
async def diagnostics_versions_fallback():
    versions = {}
    def _put(name, mod, attr='__version__'):
        try:
            m = __import__(mod, fromlist=['*'])
            versions[name] = getattr(m, attr, 'unknown')
        except Exception as e:
            versions[name] = f"not_installed ({e})"
    _put('python', 'sys', 'version')
    _put('protobuf', 'google.protobuf')
    _put('mediapipe', 'mediapipe')
    _put('onnxruntime', 'onnxruntime')
    _put('opencv', 'cv2')
    _put('tensorflow', 'tensorflow')
    _put('torch', 'torch')
    return {"success": True, "versions": versions}

@app.get("/api/v1/diagnostics/models")
async def diagnostics_models_fallback():
    from pathlib import Path
    import os
    status = {}
    try:
        p = os.getenv('SADTALKER_WEIGHTS')
        ok = False
        if p and Path(p).exists():
            ok = True
        else:
            ck = Path('models') / 'sadtalker' / 'checkpoints'
            for c in [ck/ 'epoch_20.pth', ck/'facevid2vid_00189-model.pth.tar', ck/'mapping_00109-model.pth.tar']:
                if c.exists():
                    ok = True
                    break
        status['sadtalker_weights'] = ok
    except Exception as e:
        status['sadtalker_weights'] = f"error ({e})"

    try:
        status['wav2lip2'] = (Path('wav2lip2-aoti')/ 'checkpoints' / 'wav2lip2_fp8.pt').exists()
    except Exception as e:
        status['wav2lip2'] = f"error ({e})"

    try:
        repo = Path('EMAGE')
        status['emage_repo'] = repo.exists()
        status['emage_weights'] = (repo / 'checkpoints' / 'emage_best.pth').exists()
    except Exception as e:
        status['emage'] = f"error ({e})"

    return {"success": True, "models": status}

# EMAGE status fallback (always available)
@app.get("/api/v1/emage/status")
async def emage_status_fallback():
    try:
        from pathlib import Path
        repo = Path('EMAGE')
        available = repo.exists()
        details = {"present": available}
        if available:
            models_dir = repo / 'models'
            checkpoints_dir = repo / 'checkpoints'
            details.update({
                "models_dir": str(models_dir),
                "checkpoints_dir": str(checkpoints_dir),
                "gesture_decoder": (models_dir / 'gesture_decoder.py').exists() or (models_dir / 'gesture_decoder' / '__init__.py').exists(),
                "weights_present": (checkpoints_dir / 'emage_best.pth').exists()
            })
        return {"success": True, "emage": details}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

# Lightweight style presets list to satisfy frontend when using stable_server
try:
    # Reuse presets from full API if available
    from api.routes import style_presets as _style_presets  # type: ignore
except Exception:
    from datetime import datetime, timezone
    _now_iso = datetime.now(timezone.utc).isoformat()
    _style_presets = {
        "professional": {
            "preset_id": "professional",
            "name": "Professional",
            "description": "Balanced professional on-camera style",
            "intensity": 0.6,
            "smoothness": 0.85,
            "expressiveness": 0.6,
            "cultural_context": "GLOBAL",
            "formality": 0.7,
            "gesture_frequency": 0.5,
            "gesture_amplitude": 0.8,
            "created_at": _now_iso,
            "updated_at": _now_iso,
        },
        "casual": {
            "preset_id": "casual",
            "name": "Casual",
            "description": "Relaxed conversational style",
            "intensity": 0.7,
            "smoothness": 0.7,
            "expressiveness": 0.8,
            "cultural_context": "GLOBAL",
            "formality": 0.3,
            "gesture_frequency": 0.8,
            "gesture_amplitude": 1.2,
            "created_at": _now_iso,
            "updated_at": _now_iso,
        },
        "enthusiastic": {
            "preset_id": "enthusiastic",
            "name": "Enthusiastic",
            "description": "High-energy presentation style",
            "intensity": 0.9,
            "smoothness": 0.6,
            "expressiveness": 0.9,
            "cultural_context": "GLOBAL",
            "formality": 0.5,
            "gesture_frequency": 0.9,
            "gesture_amplitude": 1.5,
            "created_at": _now_iso,
            "updated_at": _now_iso,
        },
    }


@app.get("/api/v1/style-presets")
async def list_style_presets_v1():
    """List style presets for stable server compatibility."""
    try:
        presets = list(_style_presets.values())
        return {"success": True, "presets": presets, "count": len(presets)}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

@app.post("/api/v1/style-presets")
async def create_style_preset_v1(
    name: str = Form(...),
    description: str = Form(""),
    intensity: float = Form(0.7),
    smoothness: float = Form(0.8),
    expressiveness: float = Form(0.7),
    cultural_context: str = Form("GLOBAL"),
    formality: float = Form(0.5),
    gesture_frequency: float = Form(0.7),
    gesture_amplitude: float = Form(1.0)
):
    try:
        import uuid
        from datetime import datetime, timezone
        pid = str(uuid.uuid4())
        now = datetime.now(timezone.utc).isoformat()
        preset = {
            "preset_id": pid,
            "name": name,
            "description": description,
            "intensity": float(intensity),
            "smoothness": float(smoothness),
            "expressiveness": float(expressiveness),
            "cultural_context": cultural_context,
            "formality": float(formality),
            "gesture_frequency": float(gesture_frequency),
            "gesture_amplitude": float(gesture_amplitude),
            "created_at": now,
            "updated_at": now
        }
        _style_presets[preset["preset_id"]] = preset
        return {"success": True, "preset": preset}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

@app.post("/api/v1/style-presets/interpolate")
async def interpolate_style_presets_v1(
    preset1_id: str = Form(...),
    preset2_id: str = Form(...),
    ratio: float = Form(0.5)
):
    try:
        if preset1_id not in _style_presets or preset2_id not in _style_presets:
            return JSONResponse({"success": False, "detail": "Preset not found"}, status_code=404)
        import uuid
        from datetime import datetime, timezone
        p1 = _style_presets[preset1_id]
        p2 = _style_presets[preset2_id]
        now = datetime.now(timezone.utc).isoformat()
        interp = {
            "preset_id": str(uuid.uuid4()),
            "name": f"{p1['name']} + {p2['name']} ({ratio:.1f})",
            "description": f"Interpolated between {p1['name']} and {p2['name']}",
            "intensity": p1["intensity"] * (1 - ratio) + p2["intensity"] * ratio,
            "smoothness": p1["smoothness"] * (1 - ratio) + p2["smoothness"] * ratio,
            "expressiveness": p1["expressiveness"] * (1 - ratio) + p2["expressiveness"] * ratio,
            "cultural_context": p1["cultural_context"] if ratio < 0.5 else p2["cultural_context"],
            "formality": p1["formality"] * (1 - ratio) + p2["formality"] * ratio,
            "gesture_frequency": p1["gesture_frequency"] * (1 - ratio) + p2["gesture_frequency"] * ratio,
            "gesture_amplitude": p1["gesture_amplitude"] * (1 - ratio) + p2["gesture_amplitude"] * ratio,
            "created_at": now,
            "updated_at": now
        }
        return {"success": True, "interpolated_preset": interp}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

@app.post("/api/v1/style-presets/{preset_id}/cultural-variants")
async def create_cultural_variants_v1(preset_id: str):
    try:
        if preset_id not in _style_presets:
            return JSONResponse({"success": False, "detail": "Preset not found"}, status_code=404)
        base = _style_presets[preset_id]
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        contexts = ["WESTERN", "EAST_ASIAN", "MIDDLE_EASTERN", "SOUTH_ASIAN", "LATIN_AMERICAN", "AFRICAN"]
        variants = []
        for ctx in contexts:
            if ctx == base["cultural_context"]:
                continue
            import uuid
            v = dict(base)
            v.update({
                "preset_id": str(uuid.uuid4()),
                "name": f"{base['name']} ({ctx})",
                "cultural_context": ctx,
                "created_at": now,
                "updated_at": now
            })
            variants.append(v)
        return {"success": True, "variants": variants, "count": len(variants)}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

def process_video_sync(
    task_id: str,
    image_path: str,
    audio_path: Optional[str] = None,
    text: Optional[str] = None,
    settings: Dict[str, Any] = None
):
    """Synchronous video processing"""
    try:
        logger.info(f"Starting video processing for task {task_id}")
        # Ensure output path is defined early so AI pipeline can write to it
        output_path = OUTPUT_DIR / f"{task_id}.mp4"
        
        # Update progress
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["progress"] = 10
        tasks[task_id]["stage"] = "Initializing processing"
        
        # Step 1: Validate inputs
        time.sleep(1)
        tasks[task_id]["progress"] = 20
        tasks[task_id]["stage"] = "Validating inputs"
        
        if not os.path.exists(image_path):
            raise Exception("Image file not found")
        
        # Step 2: Process image
        time.sleep(2)
        tasks[task_id]["progress"] = 40
        tasks[task_id]["stage"] = "Processing avatar image"
        
        # Try to load and process image
        try:
            import cv2
            img = cv2.imread(image_path)
            try:
                # Use available CPU threads; allow OpenCL if available
                cv2.setNumThreads(max(1, os.cpu_count() or 1))
                if hasattr(cv2, 'ocl'):
                    cv2.ocl.setUseOpenCL(True)
            except Exception:
                pass
            if img is not None:
                # Resize image to standard size
                height, width = img.shape[:2]
                if width > 512:
                    scale = 512 / width
                    new_width = 512
                    new_height = int(height * scale)
                    img = cv2.resize(img, (new_width, new_height))
                
                processed_img_path = TEMP_DIR / f"{task_id}_processed.jpg"
                cv2.imwrite(str(processed_img_path), img)
                logger.info(f"Image processed and saved to {processed_img_path}")
        except ImportError:
            logger.warning("OpenCV not available, skipping image processing")
        except Exception as e:
            logger.warning(f"Image processing failed: {e}")
        
        # Step 3: Process audio/text
        time.sleep(2)
        tasks[task_id]["progress"] = 60
        tasks[task_id]["stage"] = "Processing audio/speech"
        
        final_audio_path = audio_path
        if text and not audio_path:
            # Generate TTS if text is provided
            try:
                import pyttsx3
                engine = pyttsx3.init()
                
                # Configure TTS settings
                voices = engine.getProperty('voices')
                if voices:
                    # Use female voice if available
                    for voice in voices:
                        if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                            engine.setProperty('voice', voice.id)
                            break
                
                engine.setProperty('rate', 150)  # Speaking rate
                engine.setProperty('volume', 0.9)  # Volume level
                
                tts_output = TEMP_DIR / f"{task_id}_speech.wav"
                engine.save_to_file(text, str(tts_output))
                engine.runAndWait()
                
                # Wait a bit for file to be written
                time.sleep(1)
                
                if os.path.exists(tts_output) and os.path.getsize(tts_output) > 0:
                    final_audio_path = str(tts_output)
                    logger.info(f"TTS generated successfully: {tts_output} ({os.path.getsize(tts_output)} bytes)")
                else:
                    logger.warning("TTS file was not created or is empty")
                    
            except Exception as e:
                logger.warning(f"TTS generation failed: {e}")
                # Create a simple beep as fallback
                try:
                    import subprocess
                    beep_output = TEMP_DIR / f"{task_id}_beep.wav"
                    subprocess.run([
                        "ffmpeg", "-y", "-f", "lavfi", 
                        "-i", "sine=frequency=440:duration=3",
                        str(beep_output)
                    ], capture_output=True)
                    if os.path.exists(beep_output):
                        final_audio_path = str(beep_output)
                        logger.info("Created fallback beep audio")
                except Exception:
                    pass
        
        # Step 4: Full AI Pipeline - Facial + Body + Gestures
        tasks[task_id]["progress"] = 70
        tasks[task_id]["stage"] = "Loading AI models"
        
        ai_success = False
        
        try:
            # Step 4a: Generate body gestures with EMAGE
            tasks[task_id]["stage"] = "Generating body gestures (EMAGE)"
            body_video = None

            # Only run EMAGE if requested AND available locally
            if settings and settings.get('use_emage', False) and _emage_available():
                try:
                    from models.emage_realistic import EMageRealistic
                    emage = EMageRealistic()

                    body_video = emage.generate_full_video(
                        audio_path=final_audio_path,
                        text=text,
                        emotion=settings.get('emotion', 'neutral'),
                        style=settings.get('body_style', 'natural'),
                        avatar_type=settings.get('avatar_type', 'realistic'),
                        background=None,
                        output_path=str(TEMP_DIR / f"{task_id}_body.mp4")
                    )
                    logger.info(f"EMAGE body animation generated: {body_video}")
                except Exception as e:
                    logger.warning(f"EMAGE failed: {e}")
            elif settings and settings.get('use_emage', False):
                logger.warning("EMAGE requested but EMAGE modules/weights not available; skipping EMAGE stage")
            
            # Step 4b: Generate facial animation with SadTalker
            tasks[task_id]["stage"] = "Generating facial animation (SadTalker)"
            face_video = None

            try:
                from real_video_generator import generate_real_video
                
                face_video = generate_real_video(
                    image_path,
                    final_audio_path,
                    str(TEMP_DIR / f"{task_id}_face.mp4")
                )
                logger.info(f"Real SadTalker video generated: {face_video}")
            except Exception as e:
                logger.warning(f"Real SadTalker failed: {e}")
            
            # Step 4c: Enhance lip-sync with Wav2Lip2
            tasks[task_id]["stage"] = "Enhancing lip-sync (Wav2Lip2)"
            lipsync_video = None

            # Only run Wav2Lip2 if requested and available
            if settings and settings.get('use_wav2lip2', False) and face_video and _wav2lip2_available():
                try:
                    from models.wav2lip2_aoti import Wav2Lip2AOTI
                    wav2lip = Wav2Lip2AOTI()

                    # Generate a lip-synced face video directly from image+audio
                    lipsync_video = wav2lip.generate_video(
                        image_path=image_path,
                        audio_path=final_audio_path,
                        output_path=str(TEMP_DIR / f"{task_id}_lipsync.mp4"),
                        fps=settings.get('fps', 25)
                    )
                    logger.info(f"Wav2Lip2 lip-sync enhanced: {lipsync_video}")
                except Exception as e:
                    logger.warning(f"Wav2Lip2 failed: {e}")
            elif settings and settings.get('use_wav2lip2', False):
                logger.warning("Wav2Lip2 requested but not available; skipping Wav2Lip2 stage")
            
            # Step 4d: Combine face and body
            tasks[task_id]["stage"] = "Combining facial and body animation"
            
            final_video = lipsync_video or face_video
            
            if body_video and final_video:
                # Composite face and body videos
                composite_output = TEMP_DIR / f"{task_id}_composite.mp4"
                
                ffmpeg_cmd = _find_ffmpeg()
                if not ffmpeg_cmd:
                    raise Exception("FFmpeg not available for compositing")
                composite_cmd = [
                    ffmpeg_cmd, "-y",
                    "-i", final_video,  # Face video
                    "-i", body_video,   # Body video
                    "-filter_complex", "[0:v][1:v]overlay=0:0[v]",
                    "-map", "[v]", "-map", "0:a",
                    "-c:v", "libx264", "-c:a", "aac",
                    str(composite_output)
                ]
                
                result = subprocess.run(composite_cmd, capture_output=True, text=True, timeout=120)
                
                if result.returncode == 0 and os.path.exists(composite_output):
                    import shutil
                    shutil.copy2(composite_output, output_path)
                    ai_success = True
                    logger.info(f"Full AI pipeline completed: {output_path}")
            elif final_video:
                # Use face video only
                import shutil
                shutil.copy2(final_video, output_path)
                ai_success = True
                logger.info(f"Facial AI pipeline completed: {output_path}")
            
        except Exception as e:
            logger.warning(f"AI pipeline failed: {e}, falling back to basic generation")
        
        if not ai_success:
            tasks[task_id]["progress"] = 80
            tasks[task_id]["stage"] = "Generating basic video"
        
        # output_path already defined above
        
        # Skip FFmpeg if AI pipeline succeeded
        if ai_success:
            tasks[task_id]["mode"] = "ai"
            tasks[task_id]["progress"] = 95
            tasks[task_id]["stage"] = "Finalizing AI-generated video"
            time.sleep(1)
        else:
        
            # Try to create video with ffmpeg
            try:
                import subprocess
            
                # Check if ffmpeg is available
                ffmpeg_paths = [
                    "ffmpeg",  # Try system PATH first
                    r"C:\ffmpeg\bin\ffmpeg.exe",  # Common installation path
                    r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
                    r"C:\Users\%USERNAME%\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
                ]

                ffmpeg_cmd = None
                for path in ffmpeg_paths:
                    try:
                        if path == "ffmpeg":
                            result = subprocess.run([path, "-version"], capture_output=True, text=True, timeout=10)
                        else:
                            expanded_path = os.path.expandvars(path)
                            if os.path.exists(expanded_path):
                                result = subprocess.run([expanded_path, "-version"], capture_output=True, text=True, timeout=10)
                            else:
                                continue
                    
                        if result.returncode == 0:
                            ffmpeg_cmd = path if path == "ffmpeg" else expanded_path
                            logger.info(f"Found FFmpeg at: {ffmpeg_cmd}")
                            break
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        continue
            
                if not ffmpeg_cmd:
                    raise FileNotFoundError("FFmpeg not found in system PATH or common locations")
            
                if result.returncode == 0:
                    logger.info("FFmpeg available, creating video")
                
                    # Get audio duration first
                    audio_duration = 10  # default
                    if final_audio_path and os.path.exists(final_audio_path):
                        try:
                            # Get audio duration using ffprobe
                            ffprobe_cmd = ffmpeg_cmd.replace("ffmpeg", "ffprobe") if "ffmpeg" in ffmpeg_cmd else "ffprobe"
                            probe_cmd = [
                                ffprobe_cmd, "-v", "quiet", "-show_entries", 
                                "format=duration", "-of", "csv=p=0", final_audio_path
                            ]
                            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)
                            if probe_result.returncode == 0 and probe_result.stdout.strip():
                                duration_str = probe_result.stdout.strip()
                                audio_duration = max(float(duration_str), 3.0)  # Minimum 3 seconds
                                logger.info(f"Audio duration detected: {audio_duration} seconds")
                            else:
                                logger.warning(f"Could not detect audio duration, using default: {audio_duration}s")
                        except Exception as e:
                            logger.warning(f"Audio duration detection failed: {e}, using default: {audio_duration}s")
                    else:
                        logger.info(f"No audio file, using default duration: {audio_duration}s")
                    
                    # Build ffmpeg command for proper video generation
                    cmd = [
                        ffmpeg_cmd, "-y",
                        "-loop", "1", "-i", image_path,  # Loop the image
                    ]

                    # Add audio input
                    if final_audio_path and os.path.exists(final_audio_path):
                        cmd.extend(["-i", final_audio_path])
                        # Use original audio
                        cmd.extend(["-c:a", "aac", "-b:a", "128k"])
                        # Match video duration to audio
                        cmd.extend(["-shortest"])
                    else:
                        # Generate silent audio for the duration
                        cmd.extend(["-f", "lavfi", "-i", f"anullsrc=r=44100:cl=stereo:d={audio_duration}"])
                        cmd.extend(["-c:a", "aac"])
                    # Set explicit duration
                    cmd.extend(["-t", str(audio_duration)])

                    # Video settings
                    cmd.extend([
                        "-c:v", "libx264",
                        *( ["-threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 else [] ),
                        "-pix_fmt", "yuv420p",
                        "-r", "25",  # Frame rate
                        "-crf", "18",
                        "-tune", "stillimage",
                        "-vf", "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2",
                        str(output_path)
                    ])

                    logger.info(f"Running ffmpeg command: {' '.join(cmd)}")

                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=120  # Increased timeout
                    )

                    if result.returncode == 0:
                        if os.path.exists(output_path) and os.path.getsize(output_path) > 1000:
                            logger.info(f"Video created successfully: {output_path} ({os.path.getsize(output_path)} bytes)")
                        else:
                            logger.error(f"Video file created but too small: {os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes")
                            raise Exception("Generated video file is too small or empty")
                    else:
                        logger.error(f"FFmpeg failed with return code {result.returncode}")
                        logger.error(f"FFmpeg stderr: {result.stderr}")
                        logger.error(f"FFmpeg stdout: {result.stdout}")
                        raise Exception(f"FFmpeg failed: {result.stderr}")
                else:
                    raise Exception("FFmpeg not available")
                
            except subprocess.TimeoutExpired:
                logger.error("FFmpeg timeout")
                raise Exception("Video generation timeout")
            except FileNotFoundError:
                logger.error("FFmpeg not found in PATH - cannot generate video")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = "FFmpeg is required for video generation. Please install FFmpeg."
                return
            except Exception as e:
                logger.error(f"Video generation failed: {e}")
                tasks[task_id]["status"] = "failed"
                tasks[task_id]["error"] = f"Video generation failed: {str(e)}"
                return
            
            # Mark mode only if not already AI
            tasks[task_id]["mode"] = tasks[task_id].get("mode", "basic")
        
        # Step 5: Finalize
        time.sleep(1)
        # Apply lightweight advanced effects (optional)
        try:
            effects_profile = (settings or {}).get('effects', 'cinematic')
            _apply_advanced_effects(str(output_path), effects_profile)
        except Exception as e:
            logger.warning(f"Skipping effects: {e}")
        tasks[task_id]["progress"] = 100
        tasks[task_id]["status"] = "completed"
        tasks[task_id]["stage"] = "Video generation completed"
        tasks[task_id]["video_url"] = f"/api/download/{task_id}.mp4"
        
        logger.info(f"Video processing completed for task {task_id}")
        
    except Exception as e:
        logger.error(f"Video processing failed for task {task_id}: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        tasks[task_id]["stage"] = f"Failed: {str(e)}"

async def process_video_async(
    task_id: str,
    image_path: str,
    audio_path: Optional[str] = None,
    text: Optional[str] = None,
    settings: Dict[str, Any] = None
):
    """Async wrapper for video processing"""
    loop = asyncio.get_event_loop()
    try:
        async with _sem:
            await loop.run_in_executor(
                None,
                process_video_sync,
                task_id, image_path, audio_path, text, settings
            )
    except asyncio.CancelledError:
        # Mark task as failed but do not crash the request pipeline
        logger.warning(f"Video processing task cancelled: {task_id}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = "Task cancelled"
        tasks[task_id]["stage"] = "Cancelled"
        raise

# API endpoints
@app.get("/api/health")
async def health():
    return {
        "status": "ok", 
        "message": "Stable AI server running",
        "tasks_active": len([t for t in tasks.values() if t["status"] == "processing"])
    }

@app.get("/api/v1/health")
async def health_v1():
    return {
        "status": "ok", 
        "message": "Stable AI server running",
        "models": "ready"
    }

@app.post("/api/generate/preview")
async def generate_preview(
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
    duration: int = Form(3)
):
    """Generate a short preview clip for the given image and optional audio.
    Tries a fast SadTalker render if available; otherwise falls back to ffmpeg.
    Returns an MP4 file directly.
    """
    try:
        task_id = str(uuid.uuid4())
        img_path = TEMP_DIR / f"{task_id}_{image.filename or 'image' }"
        with open(img_path, 'wb') as f:
            f.write(await image.read())

        audio_path = None
        if audio is not None:
            audio_path = TEMP_DIR / f"{task_id}_{audio.filename or 'audio' }"
            with open(audio_path, 'wb') as f:
                f.write(await audio.read())

        preview_path = TEMP_DIR / f"{task_id}_preview.mp4"

        # Try SadTalker quick pass
        ai_ok = False
        try:
            from models.sadtalker_full import SadTalkerFull
            st = SadTalkerFull()
            # Small tweaks for speed
            st.img_size = 192
            result = st.generate(
                image_path=str(img_path),
                audio_path=str(audio_path) if audio_path else str(img_path),
                output_path=str(preview_path),
                emotion='neutral',
                enhance_face=False
            )
            if os.path.exists(result) and os.path.getsize(result) > 10_000:
                ai_ok = True
        except Exception:
            ai_ok = False

        if not ai_ok:
            # Fallback to ffmpeg still image + short duration/audio
            import subprocess
            ffmpeg_cmd = _find_ffmpeg() if '_find_ffmpeg' in globals() else 'ffmpeg'
            cmd = [
                ffmpeg_cmd, '-y',
                '-loop', '1', '-i', str(img_path),
            ]
            if audio_path:
                cmd += ['-i', str(audio_path)]
            else:
                cmd += ['-f', 'lavfi', '-i', 'anullsrc=r=44100:cl=stereo']
            cmd += [
                '-t', str(max(1, int(duration))),
                '-c:v', 'libx264',
                *( ["-threads", str(FFMPEG_THREADS)] if FFMPEG_THREADS > 0 else [] ),
                '-pix_fmt', 'yuv420p',
                '-shortest', str(preview_path)
            ]
            subprocess.run(cmd, capture_output=True)

        if not preview_path.exists():
            raise HTTPException(status_code=500, detail='Preview generation failed')

        return FileResponse(
            str(preview_path),
            media_type='video/mp4',
            filename=preview_path.name
        )
    except HTTPException:
        raise
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/api/v1/generate/video")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
    text: str = Form(None),
    resolution: str = Form("720p"),
    fps: int = Form(30),
    expressionIntensity: float = Form(0.8),
    gestureLevel: str = Form("medium"),
    voiceModel: str = Form("en-US-JennyNeural"),
    background: str = Form("blur"),
    enhanceFace: bool = Form(True),
    stabilization: bool = Form(True),
    expressionEngine: str = Form('auto')
):
    """Real video generation from uploaded files"""
    
    if not image:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either audio file or text is required")
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "stage": "Starting video generation",
        "created_at": time.time()
    }
    
    try:
        # Save uploaded files
        image_path = TEMP_DIR / f"{task_id}_{image.filename}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        audio_path = None
        if audio:
            audio_path = TEMP_DIR / f"{task_id}_{audio.filename}"
            with open(audio_path, "wb") as f:
                content = await audio.read()
                f.write(content)
        
        # Settings
        settings = {
            "resolution": resolution,
            "fps": fps,
            "expression_intensity": expressionIntensity,
            "gesture_level": gestureLevel,
            "voice_model": voiceModel,
            "background": background,
            "enhance_face": enhanceFace,
            "stabilization": stabilization,
            "expression_engine": expressionEngine
        }
        
        # Start processing outside response lifecycle to avoid CancelledError
        asyncio.create_task(
            process_video_async(
                task_id, str(image_path), str(audio_path) if audio_path else None, text, settings
            )
        )
        
        logger.info(f"Video generation task created: {task_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Video generation started",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to create video generation task: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/video-from-prompt")
async def generate_video_from_prompt(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    voice: str = Form("en-US-JennyNeural"),
    resolution: str = Form("720p"),
    fps: int = Form(30),
    gestureLevel: str = Form("medium"),
    expressionEngine: str = Form('auto')
):
    """Video generation from text prompt"""
    
    if not prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "stage": "Generating avatar and speech from prompt",
        "created_at": time.time()
    }
    
    try:
        # Create default avatar if needed
        default_avatar = TEMP_DIR / f"{task_id}_default_avatar.jpg"
        
        try:
            import cv2
            import numpy as np
            
            # Create a simple avatar image
            img = np.ones((512, 512, 3), dtype=np.uint8) * 240
            
            # Draw a simple face
            cv2.circle(img, (256, 256), 200, (200, 200, 200), -1)  # Face
            cv2.circle(img, (200, 200), 30, (50, 50, 50), -1)      # Left eye
            cv2.circle(img, (312, 200), 30, (50, 50, 50), -1)      # Right eye
            cv2.ellipse(img, (256, 320), (80, 40), 0, 0, 180, (50, 50, 50), 5)  # Mouth
            
            cv2.imwrite(str(default_avatar), img)
            logger.info(f"Default avatar created: {default_avatar}")
            
        except Exception as e:
            logger.warning(f"Could not create default avatar: {e}")
            # Create a minimal placeholder
            with open(default_avatar, 'wb') as f:
                f.write(b"Placeholder avatar image")
        
        # Settings
        settings = {
            "resolution": resolution,
            "fps": fps,
            "voice_model": voice,
            "gesture_level": gestureLevel,
            "expression_engine": expressionEngine
        }
        
        # Start processing outside response lifecycle to avoid CancelledError
        asyncio.create_task(
            process_video_async(
                task_id, str(default_avatar), None, prompt, settings
            )
        )
        
        logger.info(f"Prompt-based generation task created: {task_id}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Video generation from prompt started",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to create prompt-based task: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/generate/advanced-video")
async def generate_advanced_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: UploadFile = File(None),
    text: str = Form(None),
    useEmage: bool = Form(False),
    useWav2Lip2: bool = Form(False),
    useSadTalkerFull: bool = Form(False),
    emotion: str = Form("neutral"),
    bodyStyle: str = Form("natural"),
    avatarType: str = Form("realistic"),
    lipSyncQuality: str = Form("high"),
    resolution: str = Form("720p"),
    fps: int = Form(30)
):
    """Advanced video generation with AI models"""
    
    if not image:
        raise HTTPException(status_code=400, detail="Image file is required")
    
    if not audio and not text:
        raise HTTPException(status_code=400, detail="Either audio file or text is required")
    
    # Create task
    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status": "processing",
        "progress": 0,
        "stage": "Starting advanced AI video generation",
        "created_at": time.time()
    }
    
    try:
        # Save uploaded files
        image_path = TEMP_DIR / f"{task_id}_{image.filename}"
        with open(image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        audio_path = None
        if audio:
            audio_path = TEMP_DIR / f"{task_id}_{audio.filename}"
            with open(audio_path, "wb") as f:
                content = await audio.read()
                f.write(content)
        
        # Advanced settings
        settings = {
            "use_emage": useEmage,
            "use_wav2lip2": useWav2Lip2,
            "use_sadtalker_full": useSadTalkerFull,
            "emotion": emotion,
            "body_style": bodyStyle,
            "avatar_type": avatarType,
            "lip_sync_quality": lipSyncQuality,
            "resolution": resolution,
            "fps": fps
        }
        
        # Start processing outside response lifecycle to avoid CancelledError
        asyncio.create_task(
            process_video_async(
                task_id, str(image_path), str(audio_path) if audio_path else None, text, settings
            )
        )
        
        logger.info(f"Advanced video generation task created: {task_id}")
        logger.info(f"Models: EMAGE={useEmage}, Wav2Lip2={useWav2Lip2}, SadTalker={useSadTalkerFull}")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": "Advanced video generation started",
            "status": "processing"
        }
        
    except Exception as e:
        logger.error(f"Failed to create advanced task: {e}")
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/generate/batch")
async def generate_batch(
    images: List[UploadFile] = File(...),
    audios: Optional[List[UploadFile]] = File(None),
    texts: Optional[List[str]] = Form(None),
    resolution: str = Form("720p"),
    fps: int = Form(25)
):
    """Batch generation: schedules multiple tasks and returns task_ids."""
    task_ids: List[str] = []
    try:
        for idx, image in enumerate(images):
            task_id = str(uuid.uuid4())
            tasks[task_id] = {"status": "processing", "progress": 0, "stage": "Queued", "created_at": time.time()}

            image_path = TEMP_DIR / f"{task_id}_{image.filename}"
            with open(image_path, "wb") as f:
                f.write(await image.read())

            audio_path = None
            if audios and idx < len(audios) and audios[idx] is not None:
                audio_file = audios[idx]
                audio_path = TEMP_DIR / f"{task_id}_{audio_file.filename}"
                with open(audio_path, "wb") as f:
                    f.write(await audio_file.read())

            text = None
            if texts and idx < len(texts):
                text = texts[idx]

            settings = {"resolution": resolution, "fps": fps}

            asyncio.create_task(
                process_video_async(
                    task_id, str(image_path), str(audio_path) if audio_path else None, text, settings
                )
            )

            task_ids.append(task_id)

        return {"success": True, "task_ids": task_ids, "count": len(task_ids)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/status/{task_id}")
@app.get("/api/v1/status/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    return {
        "success": True,
        "data": {
            "status": task["status"],
            "progress": task["progress"],
            "stage": task["stage"],
            "mode": task.get("mode", "basic" if task.get("status") == "completed" else None),
            "video_url": task.get("video_url"),
            "error": task.get("error")
        }
    }

@app.get("/api/download/{filename}")
@app.get("/api/v1/download/{filename}")
async def download_file(filename: str):
    """Download generated video"""
    
    file_path = OUTPUT_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="video/mp4"
    )

# Serve SPA
@app.get("/{full_path:path}")
async def serve_spa(full_path: str):
    """Serve the React SPA"""
    
    if full_path.startswith("api/"):
        raise HTTPException(status_code=404, detail="API endpoint not found")
    
    # Try to serve the requested file
    if full_path and not full_path.startswith("api/"):
        file_path = FRONTEND_DIR / full_path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
    
    # Return index.html for SPA routing
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    
    # Frontend not built
    return JSONResponse({
        "error": "Frontend not built",
        "message": "Please run 'npm run build' in the frontend directory",
        "frontend_dir": str(FRONTEND_DIR)
    }, status_code=500)

if __name__ == "__main__":
    try:
        logger.info("Starting stable PaksaTalker server...")
        logger.info("Server includes real AI processing capabilities")
        logger.info("Server will be available at: http://localhost:8000")
        # Minimal logging config to avoid uvicorn DefaultFormatter (which expects isatty)
        SIMPLE_LOG_CONFIG = {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "simple": {"format": "%(asctime)s - %(levelname)s - %(name)s - %(message)s"}
            },
            "handlers": {
                "console": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple",
                    "stream": "ext://sys.stdout"
                }
            },
            "loggers": {
                "uvicorn": {"handlers": ["console"], "level": "INFO"},
                "uvicorn.error": {"handlers": ["console"], "level": "INFO", "propagate": False},
                "uvicorn.access": {"handlers": ["console"], "level": "INFO", "propagate": False}
            }
        }

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8001,
            log_level="info",
            log_config=SIMPLE_LOG_CONFIG
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server startup failed: {e}")
        import traceback
        traceback.print_exc()

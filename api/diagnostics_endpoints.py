"""
Diagnostics endpoints to verify runtime versions and model availability in the running server.
"""
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter(prefix="/diagnostics", tags=["diagnostics"])


@router.get("/versions")
async def get_versions() -> Dict[str, Any]:
    versions: Dict[str, Any] = {}
    def _put(name: str, mod: str, attr: str = "__version__"):
        try:
            m = __import__(mod, fromlist=["*"])
            versions[name] = getattr(m, attr, "unknown")
        except Exception as e:
            versions[name] = f"not_installed ({e})"

    _put("python", "sys", "version")
    _put("protobuf", "google.protobuf")
    _put("mediapipe", "mediapipe")
    _put("onnxruntime", "onnxruntime")
    _put("opencv", "cv2")
    _put("tensorflow", "tensorflow")
    _put("torch", "torch")
    return {"success": True, "versions": versions}


@router.get("/models")
async def get_model_status() -> Dict[str, Any]:
    status: Dict[str, Any] = {}
    # SadTalker weights
    try:
        import os
        from pathlib import Path
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

    # Wav2Lip2 weights
    try:
        from pathlib import Path
        status['wav2lip2'] = (Path('wav2lip2-aoti')/ 'checkpoints' / 'wav2lip2_fp8.pt').exists()
    except Exception as e:
        status['wav2lip2'] = f"error ({e})"

    # EMAGE
    try:
        from pathlib import Path
        repo = Path('EMAGE')
        status['emage_repo'] = repo.exists()
        status['emage_weights'] = (repo / 'checkpoints' / 'emage_best.pth').exists()
    except Exception as e:
        status['emage'] = f"error ({e})"

    return {"success": True, "models": status}


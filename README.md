# PaksaTalker

PaksaTalker is an AI-powered video generation stack for creating hyper‑realistic talking avatars with synchronized lip‑sync, expressive facial animation, and natural body gestures. It combines multiple models (SadTalker, Wav2Lip2 AOTI, EMAGE, OpenSeeFace) behind a FastAPI backend and a React + TypeScript frontend.

## Highlights

- End‑to‑end pipeline: text or audio → speech → face + body animation → post‑effects
- Fusion mode: composits an animated face over an EMAGE body track (with fallbacks)
- Background customization: blur/portrait/cinematic, and green‑screen replace (color/image)
- Live progress: optional WebSocket updates with steps, percent, elapsed, ETA
- Asset prefetch: downloads common model assets in the background to avoid first‑run errors

## Quick Start

Prereqs: Python 3.9+, Node 16+, ffmpeg installed and on PATH. NVIDIA GPU recommended.

1) Install backend deps

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

2) Build frontend

```bash
cd frontend
npm install
npm run build
cd ..
```

3) Run the stable backend server

```bash
python stable_server.py
# API at http://localhost:8000, static frontend (if built) at /assets
# Swagger docs: http://localhost:8000/api/docs
```

On first run, the server and frontend automatically trigger background model downloads where possible. You can also call:

```bash
curl -X POST http://localhost:8000/api/v1/assets/ensure
```

to force asset checks and prefetches.

## Using Fusion Generation

Fusion takes an image + audio (or a prompt that will be synthesized) and returns a video. Example (multipart form):

POST /api/v1/generate/fusion-video

Form fields:

- image: file (face photo)
- audio: file (wav/mp3) or use `prompt` to synthesize TTS
- resolution: e.g. 480p, 720p
- fps: e.g. 25, 30
- emotion, style: strings (optional)
- preferWav2Lip2: true/false
- Background options (optional):
  - backgroundMode: none|blur|portrait|cinematic|color|image|greenscreen
  - backgroundColor: e.g. #000000 (for color/greenscreen)
  - backgroundImage: file (for image/greenscreen)
  - chromaColor, similarity, blend: fine‑tuning for green‑screen chroma key

The server will: generate face and body tracks (with fallbacks), composite, then apply optional background effects.

## AI Style Suggestions (MVP)

Get preset suggestions based on basic hints:

```bash
curl -X POST http://localhost:8000/api/v1/style-presets/suggest \
  -F prompt="energetic keynote" -F cultural_context=GLOBAL -F formality=0.7
```

Returns top 3 matching presets from the built‑in set.

## Model Assets & Paths

Many assets auto‑download to local cache. You can override repo roots via env vars:

- EMAGE: `PAKSA_EMAGE_ROOT` or `EMAGE_ROOT` → path to the EMAGE Python repo (not web assets)
- OpenSeeFace: `PAKSA_OSF_ROOT` or `OPENSEEFACE_ROOT` → path to OSF folder with `models/`

Ensure endpoint (non‑blocking downloads): `POST /api/v1/assets/ensure`.

## Troubleshooting

- EMAGE fallback: If you see “No module named EMAGE.models”, EMAGE Python repo isn’t present. Set `PAKSA_EMAGE_ROOT` to the repo path containing `models/gesture_decoder.py` and place `checkpoints/emage_best.pth` (auto‑download supported).
- OpenSeeFace: If face tracking falls back, ensure `OpenSeeFace/models` has required `.onnx` files or set `PAKSA_OSF_ROOT`.
- ffmpeg not found: Install ffmpeg and put it on PATH. The server detects common Windows paths.

## Developer Docs

See docs/DEVELOPER_GUIDE.md for architecture, endpoints, asset prefetch, environment variables, testing, and deployment.


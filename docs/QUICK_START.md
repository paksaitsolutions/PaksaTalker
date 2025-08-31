# Quick Start

This guide shows the fastest way to get PaksaTalker running locally.

## Requirements

- Python 3.9+
- Node.js 16+ and npm 8+
- ffmpeg installed and on PATH
- NVIDIA GPU recommended

## Steps

1) Backend dependencies

```
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/macOS
# source venv/bin/activate
pip install -r requirements.txt
```

2) Frontend build

```
cd frontend
npm install
npm run build
cd ..
```

3) Run stable server

```
python stable_server.py
# API: http://localhost:8000
# Swagger: http://localhost:8000/api/docs
```

4) Prefetch model assets (optional)

```
curl -X POST http://localhost:8000/api/v1/assets/ensure
```

## Fusion Generation

`POST /api/v1/generate/fusion-video` accepts `image`, `audio` or `prompt`, and supports optional background parameters:

- `backgroundMode`: `none|blur|portrait|cinematic|color|image|greenscreen`
- `backgroundColor` (hex), `backgroundImage` (file), `chromaColor`, `similarity`, `blend`

## AI Style Suggestions

`POST /api/v1/style-presets/suggest` → `{ suggestions: [...] }`

Pass `prompt`, `emotion`, `cultural_context`, and/or `formality`.


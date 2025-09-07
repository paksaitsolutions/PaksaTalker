# API Documentation (PaksaTalker)

This document describes the primary API endpoints for PaksaTalker, including the latest parameters and behavior optimized for CPU‑only environments. All endpoints are served under the base path `/api/v1` unless otherwise noted.

NOTE: Errors are returned as JSON with fields `success: false` and `detail` (string message). For long‑running operations, endpoints return a `task_id` and you poll task status until completion.

## Health and Capabilities

- GET `/capabilities`
  - Returns model/tool availability flags used by the frontend to adapt UI.
  - Response: `{ success, data: { models: { sadtalker, wav2lip2, emage, mediapipe, threeddfa, openseeface, mini_xception, sadtalker_weights } , ffmpeg: { available, filters: {...} } } }`

- GET `/expressions/capabilities`
  - Returns availability for expression engines only.

## Voices and Languages (sorted)

- GET `/voices`
  - Returns an alphabetically sorted list of supported TTS voices.
  - Response: `{ success, voices: [{ voice_id, language, language_code, flag, name, gender }, ...] }`

- GET `/languages`
  - Returns an alphabetically sorted list of supported languages.
  - Response: `{ success, languages: [{ code, name, flag, voice_count }, ...] }`

These endpoints are available in the standard API router; when the API router is disabled, the Stable Server exposes fallback routes at the same paths.

## Expression Estimation

- POST `/expressions/estimate`
  - Form fields: `image` (file, required), `engine` (string; one of `auto|mediapipe|threeddfa|openseeface|mini_xception`, default `auto`).
  - Response: `{ success, engine, result: { blendshapes: {...}, emotion_probs: {...} } }`

## Video Generation (Fusion)

- POST `/generate/fusion-video`
  - Purpose: Main entry to generate video from image + audio (or prompt → TTS) using SadTalker where available; falls back to a still+audio ffmpeg video and a lightweight Wav2Lip stub when SadTalker is not available.
  - Form fields:
    - `image` (file): source face image (required for media flow; if missing but `text` present, a default avatar is synthesized).
    - `audio` (file): optional audio; if not present and `text` is provided, the server will synthesize speech.
    - `text` (string): optional text for TTS.
    - `resolution` (string): e.g. `480p | 720p | 1080p` (default: 720p; CPU‑only use 480p for speed).
    - `fps` (int): frames per second (default: 25–30; CPU‑only use 25).
    - `emotion` (string): hint for facial animation (default: `neutral`).
    - `style` (string): animation style (default: `natural`).
    - `expressionEngine` (string): expression estimator; `auto` by default.
    - `preferWav2Lip2` (bool): optional.
    - `preprocess` (string): `full|crop|resize|extcrop|extfull` (default: `full`; CPU‑only recommended: `crop`).
    - Background (optional): `backgroundMode`, `backgroundColor`, `backgroundImage`, chroma key params.
  - Response: `{ success: true, task_id, status: 'processing' }`

## Prompt → Video (Text‑to‑Video)

- POST `/generate/video-from-prompt`
  - Purpose: Generate a video from a prompt using free/offline models by default.
  - Behavior: Uses free TTS provider (`gTTS`) for audio; uses the same underlying video generation path as Fusion (SadTalker if available; fallback otherwise).
  - Form fields:
    - `prompt` (string, required)
    - `voice` (string, optional): a voice id from `/voices` (defaults to `en-US-JennyNeural`).
    - `resolution`, `fps`, `expressionEngine`, `preprocess` as described for Fusion.
  - Response: `{ success: true, task_id, status: 'processing' }`

## Task Status / Polling

- GET `/status/{task_id}`
  - Returns progress for long‑running operations started by generation endpoints.
  - Response (example):
    ```json
    {
      "success": true,
      "data": {
        "status": "processing|completed|failed",
        "progress": 0,
        "stage": "Starting fusion engine",
        "video_url": "/api/v1/download/<file>",
        "error": "" 
      }
    }
    ```

## Download

- GET `/download/{filename}`
  - Returns the generated MP4 file for the given filename.

## Error Model

Errors return HTTP status codes (e.g., 400, 500) and JSON bodies:

```json
{ "success": false, "detail": "Human readable error" }
```

## Environment Variables (Performance)

- `PAKSA_PREPROCESS`: prefer `crop` to speed CPU‑only runs.
- `PAKSA_STRICT_PREPROCESS`: `1|true` to disable full→crop retries and stick to the selected mode.
- `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `NUMEXPR_MAX_THREADS`: set to `4` (or a small value) to avoid CPU thrash.

## Notes on CPU‑only Operation

- SadTalker on CPU is slow (minutes per 200 frames). Use `preprocess=crop`, `size=256`, `fps=25`, and `resolution=480p` for a practical experience.
- If SadTalker fails (e.g., checkpoint mismatch), the system falls back to a still+audio video and applies a light lip‑sync stub when possible.


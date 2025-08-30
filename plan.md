# Project Plan

## Status Updates

- [x] Optimize performance
  - Switched long-running jobs to background `asyncio.create_task` to avoid cancellation and keep tasks alive.
  - Prevented network installs/downloads during generation (offline-first pipeline).
  - Added robust FFmpeg discovery and safer fallbacks; tuned encoding path.
  - Avoided AI re-encoding on success; only re-encode once for effects (veryfast+faststart).
  - Fixed uninitialized `output_path` bug that could force still-image fallback.
  - Gated Wav2Lip2 to run only with local weights (no remote fetching).
  - Recommended protobuf pin (3.20.3) and `soundfile` for faster audio IO.

- [x] Implement advanced rendering effects
  - Added optional lightweight post-processing profiles applied via FFmpeg:
    - `cinematic` (default): sharpen + contrast + vignette
    - `portrait`: subtle contrast/saturation + sharpen
    - `sharpen`: unsharp mask only
    - `none`: disable effects
  - Effects run after AI/fallback video generation and replace the final file in-place.

## Notes
- Core talking-head generation uses SadTalker; Wav2Lip2 runs only if local weights are present.
- EMAGE is skipped unless the EMAGE repo and checkpoints are available locally.
- Preview endpoint implemented for quick inline checks.

## Next
- (Optional) Add `/api/v1/capabilities` and auto-toggle advanced features in the UI.
- (Optional) Wire UI selector for effects profile (cinematic/portrait/sharpen/none).

## Advanced Prompt Engineering

- [x] Enhanced prompting
  - [x] System prompts for consistent persona
    - Implemented in `models/prompt_engine.py` via `SystemPrompt` per `PersonaType`.
    - Exposed at `POST /api/v1/prompt/generate` and `GET /api/v1/prompt/personas`.
  - [x] Few-shot learning templates
    - Templates defined in `models/prompt_engine.py` and exposed via `GET /api/v1/prompt/examples/{category}` and `GET /api/v1/prompt/example-categories`.
  - [x] Dynamic prompt construction
    - `AdvancedPromptEngine.construct_dynamic_prompt(...)` builds context-aware prompts with duration, emotion, persona, and examples.
  - [x] Safety and moderation filters
    - Safety levels (strict/moderate/relaxed) enforced; moderation and validation via `/api/v1/prompt/validate` and applied within generation.

Usage
- Generate persona-aligned script: `POST /api/v1/prompt/generate` (fields: topic, persona, duration, emotion, context, safety_level, include_examples)
- Validate/enhance content: `POST /api/v1/prompt/validate`, `POST /api/v1/prompt/enhance`

## Conversational Abilities

- [x] Multi-turn conversation
  - Endpoints at `/api/v1/conversation/start`, `/api/v1/conversation/message`, `/api/v1/conversation/{session_id}`, `/api/v1/conversation/reset`, `/api/v1/conversation/config`.
  - Stores session messages and metadata in memory (replaceable with DB/Redis).
- [x] Context window management
  - Message history trimmed by `max_messages` and approximate token budget `max_context_tokens`.
  - Adds a lightweight system instruction for persona + topic.
- [x] Memory and state tracking
  - Per-session persona, safety level, topic seed, and messages retained across turns.
- [x] Topic coherence
  - Topic seed derived from first user message; included in system context to bias coherence.
- [x] Follow-up question handling
  - If user asks a question (e.g., ends with `?`), assistant offers an optional follow-up prompt.

## Style and Emotion

- [x] Style adaptation
  - [x] Emotion embedding
    - `POST /api/v1/style/emotion-embedding` returns a 4-d embedding for a named emotion.
  - [x] Formality levels
    - `POST /api/v1/style/adapt-text` with `formality=casual|neutral|formal` rewrites text accordingly.
  - [x] Domain-specific terminology
    - `adapt-text` accepts `domain` (medical|finance|tech|education) and injects key terminology notes.
  - [x] Personality traits
    - `adapt-text` accepts `personality` (friendly|authoritative|enthusiastic|concise) and adds style cues.

Usage
- Adapt text: `POST /api/v1/style/adapt-text` (text, formality, domain, personality, emotion)
- Get emotion embedding: `POST /api/v1/style/emotion-embedding` (emotion)

## Multilingual Support

- [x] Language capabilities
  - [x] Code-switching detection
    - `POST /api/v1/language/code-switch` analyzes script shares and flags multi-script text.
  - [x] Language identification
    - `POST /api/v1/language/detect` returns a heuristic language guess with confidence.
  - [x] Translation integration
    - `POST /api/v1/language/translate` uses Qwen backend when available; otherwise returns a safe fallback with a note.
  - [x] Cultural adaptation
    - `POST /api/v1/language/culture-hints` returns tone/style hints per language; `GET /api/v1/language/voices` lists supported voices.

Usage
- Detect language: `POST /api/v1/language/detect` (text)
- Code-switch check: `POST /api/v1/language/code-switch` (text[, threshold])
- Translate: `POST /api/v1/language/translate` (text, target_lang[, source_lang])
- Culture hints: `POST /api/v1/language/culture-hints` (lang)

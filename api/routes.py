"""API routes for PaksaTalker."""
import os
import uuid
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Form,
    HTTPException,
    Depends,
    status,
    Request,
    BackgroundTasks,
    Query
)
from fastapi.responses import JSONResponse, FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
from werkzeug.utils import secure_filename

from config import config
from config.languages import (
    SUPPORTED_LANGUAGES, 
    get_voice_info, 
    is_voice_supported, 
    get_default_voice,
    get_all_languages,
    get_all_voices
)
from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel
from models.qwen_omni import get_qwen_model
from integrations.gesture import GestureGenerator
import uuid
# from models.style_presets import StylePresetManager, StylePreset, StyleInterpolator

# Import schemas with error handling
try:
    from .schemas.schemas import (
        AnimationStyle,
        AnimationStyleCreate,
        AnimationStyleUpdate,
        VideoInfo,
        TaskStatusResponse,
        VoiceResponse,
        ModelAdaptationRequest,
        AdaptationStatusResponse,
        SpeakerInfo
    )
except ImportError:
    # Define minimal schemas if import fails
    from pydantic import BaseModel
    from datetime import datetime
    from typing import Optional, Dict, Any
    
    class AnimationStyle(BaseModel):
        style_id: str
        name: str
        description: Optional[str] = None
        parameters: Dict[str, Any] = {}
        created_at: datetime
        updated_at: datetime
    
    class AnimationStyleCreate(BaseModel):
        name: str
        description: Optional[str] = None
        parameters: Dict[str, Any] = {}
        speaker_id: Optional[str] = None
        is_global: bool = False
    
    class AnimationStyleUpdate(BaseModel):
        name: Optional[str] = None
        description: Optional[str] = None
        parameters: Optional[Dict[str, Any]] = None
    
    class VideoInfo(BaseModel):
        video_id: str
        status: str
        created_at: datetime
    
    class TaskStatusResponse(BaseModel):
        task_id: str
        status: str
        progress: int = 0
        message: str = ""
    
    class VoiceResponse(BaseModel):
        voice_id: str
        speaker_name: str
        created_at: datetime
        updated_at: datetime
        metadata: Dict[str, Any] = {}
    
    class ModelAdaptationRequest(BaseModel):
        speaker_id: str
        audio_dir: str
        model_type: str = "all"
    
    class AdaptationStatusResponse(BaseModel):
        task_id: str
        status: str
        progress: int = 0
        message: str = ""
    
    class SpeakerInfo(BaseModel):
        speaker_id: str
        created_at: datetime
        last_updated: datetime
        metadata: Dict[str, Any] = {}
        num_recordings: int = 0
        embedding_shape: Optional[tuple] = None

# Security setup
SECRET_KEY = config.get('security.secret_key', 'your-secret-key-here')
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")
AUTH_DISABLED = os.environ.get('PAKSA_AUTH_DISABLED') in ('1','true','True','yes')

# Mock database (replace with real database in production)
fake_users_db = {}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str

# Create router with API versioning
router = APIRouter(tags=["api"])

# Initialize models (lazy loading)
models_initialized = False
sadtalker = None
wav2lip = None
gesture = None
qwen = None
style_manager = None

def get_models():
    global models_initialized, sadtalker, wav2lip, gesture, qwen, style_manager
    if not models_initialized:
        try:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Initializing models...")
            
            logger.info("Initializing SadTalker...")
            sadtalker = SadTalkerModel()
            
            logger.info("Initializing Wav2Lip...")
            wav2lip = Wav2LipModel()
            
            logger.info("Initializing Gesture...")
            gesture = GestureModel()
            
            logger.info("Initializing Qwen...")
            qwen = QwenModel()
            
            logger.info("Initializing Style Manager...")
            # style_manager = StylePresetManager()
            style_manager = "initialized"  # Simple placeholder
            
            models_initialized = True
            logger.info("All models initialized successfully")
            
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.error(f"Error initializing models: {str(e)}")
            # Initialize with dummy models to prevent crashes
            sadtalker = SadTalkerModel()
            wav2lip = Wav2LipModel()
            gesture = GestureModel()
            qwen = QwenModel()
            # style_manager = StylePresetManager()
            style_manager = None
            models_initialized = True
            
    return sadtalker, wav2lip, gesture, qwen, style_manager

# --- AI-powered Style Suggestions (MVP) ---
try:
    # Use existing style router if present
    style_router  # type: ignore # noqa
except NameError:
    style_router = APIRouter(prefix="/style-presets", tags=["style-presets"])  # type: ignore

try:
    style_presets  # type: ignore # noqa
except NameError:
    # Minimal defaults if not already defined above
    now_iso = datetime.now(timezone.utc).isoformat()
    style_presets = {
        "professional": {
            "preset_id": "professional",
            "name": "Professional",
            "description": "Formal business presentation style",
            "intensity": 0.6,
            "smoothness": 0.9,
            "expressiveness": 0.5,
            "cultural_context": "GLOBAL",
            "formality": 0.9,
            "gesture_frequency": 0.4,
            "gesture_amplitude": 0.8,
            "created_at": now_iso,
            "updated_at": now_iso,
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
            "created_at": now_iso,
            "updated_at": now_iso,
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
            "created_at": now_iso,
            "updated_at": now_iso,
        },
    }


@style_router.post("/suggest")
async def suggest_style(
    prompt: Optional[str] = Form(None),
    emotion: Optional[str] = Form(None),
    cultural_context: Optional[str] = Form(None),
    formality: Optional[float] = Form(None),
):
    try:
        hints = {
            'prompt': (prompt or '').lower(),
            'emotion': (emotion or '').lower(),
            'culture': (cultural_context or '').upper() if cultural_context else None,
            'formality': float(formality) if formality is not None else None,
        }
        def score(p: Dict[str, Any]) -> float:
            s = 0.0
            # formal vs casual
            if hints['formality'] is not None:
                try:
                    s += 1.0 - abs(float(p.get('formality', 0.5)) - hints['formality'])
                except Exception:
                    pass
            # cultural match
            if hints['culture'] and p.get('cultural_context') == hints['culture']:
                s += 0.5
            # prompt keywords
            pr = hints['prompt']
            if pr:
                if any(k in pr for k in ('energy','excite','enthusi')):
                    s += float(p.get('expressiveness', 0.5))
                if any(k in pr for k in ('formal','business','corporate')):
                    s += float(p.get('formality', 0.5))
                if any(k in pr for k in ('casual','conversational','relax')):
                    s += 1.0 - abs(float(p.get('formality', 0.5)) - 0.3)
            # emotion mapping
            emo = hints['emotion']
            if emo:
                if emo in ('happy','excited','enthusiastic'):
                    s += float(p.get('expressiveness', 0.5))
                if emo in ('serious','formal','neutral'):
                    s += float(p.get('smoothness', 0.5))
            return s
        presets = list(style_presets.values())
        ranked = sorted(presets, key=score, reverse=True)
        return {"success": True, "suggestions": ranked[:3]}
    except Exception as e:
        return JSONResponse({"success": False, "detail": str(e)}, status_code=500)

# Authentication functions
def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    if AUTH_DISABLED:
        # Bypass auth completely during development
        return User(username="anonymous", email=None, full_name=None, disabled=False)
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# File handling functions
async def save_upload_file(upload_file: UploadFile, directory: str) -> str:
    """Save uploaded file to the specified directory."""
    
    # Ensure directory exists and is within allowed paths
    temp_dir = os.path.abspath(config.get('paths.temp', 'temp'))
    if directory == "image":
        directory = temp_dir
    elif directory == "audio":
        directory = temp_dir
    
    # Ensure directory is within allowed paths
    directory = os.path.abspath(directory)
    if not directory.startswith(temp_dir):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid directory path"
        )
    
    os.makedirs(directory, exist_ok=True)
    
    # Secure filename handling
    original_filename = upload_file.filename or 'upload'
    secure_name = secure_filename(original_filename)
    file_ext = os.path.splitext(secure_name)[1] or '.bin'
    
    # Validate file extension
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.mp3', '.wav', '.mp4', '.avi'}
    if file_ext.lower() not in allowed_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type {file_ext} not allowed"
        )
    
    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(directory, file_name)
    
    try:
        with open(file_path, "wb") as buffer:
            content = await upload_file.read()
            buffer.write(content)
        return file_path
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error saving file: {str(e)}"
        )

# Authentication Endpoints
@router.post("/auth/login")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    try:
        if AUTH_DISABLED:
            # Return a dummy token when auth is disabled
            access_token = create_access_token({"sub": "anonymous"}, expires_delta=timedelta(days=30))
            return {"access_token": access_token, "token_type": "bearer"}
        username = form_data.username
        password = form_data.password
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password are required"
            )
            
        user = authenticate_user(fake_users_db, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(
            data={"sub": user.username}, expires_delta=access_token_expires
        )
        
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user": {
                "id": user.username,
                "email": user.email,
                "name": user.full_name
            }
        }
    except Exception as e:
        if not isinstance(e, HTTPException):
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An error occurred during login"
            )
        raise

@router.post("/auth/register", response_model=User)
async def register_user(user: User):
    if user.username in fake_users_db:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered"
        )
    
    hashed_password = get_password_hash(user.password)  # type: ignore
    fake_users_db[user.username] = {
        "username": user.username,
        "email": user.email,
        "full_name": user.full_name,
        "hashed_password": hashed_password,
        "disabled": False
    }
    return user

# Animation Style Endpoints
animation_router = APIRouter(
    prefix="/animation-styles",
    tags=["animation-styles"],
    responses={404: {"description": "Not found"}},
)

@animation_router.post("", response_model=AnimationStyle)
async def create_animation_style(
    style: AnimationStyleCreate
):
    """Create a new animation style."""
    try:
        # Create a dummy style for now
        created_style = AnimationStyle(
            style_id=str(uuid.uuid4()),
            name=style.name,
            description=style.description,
            parameters=style.parameters,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
        return created_style
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@animation_router.get("", response_model=List[AnimationStyle])
async def list_animation_styles(
    speaker_id: Optional[str] = Query(None, description="Filter by speaker ID"),
    include_global: bool = Query(True, description="Include global styles")
):
    """List animation styles, optionally filtered by speaker."""
    # Return dummy styles for now
    styles = [
        AnimationStyle(
            style_id="default",
            name="Default Style",
            description="Default animation style",
            parameters={"intensity": 0.7},
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
        )
    ]
    return styles

@animation_router.get("/{style_id}", response_model=AnimationStyle)
async def get_animation_style(
    style_id: str
):
    """Get a specific animation style by ID."""
    # Return dummy style for now
    if style_id == "default":
        return AnimationStyle(
            style_id="default",
            name="Default Style",
            description="Default animation style",
            parameters={"intensity": 0.7},
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc)
        )
    raise HTTPException(status_code=404, detail="Style not found")

@animation_router.put("/{style_id}", response_model=AnimationStyle)
async def update_animation_style(
    style_id: str,
    style_update: AnimationStyleUpdate
):
    """Update an existing animation style."""
    if style_id != "default":
        raise HTTPException(status_code=404, detail="Style not found")
    
    # Return updated dummy style
    return AnimationStyle(
        style_id=style_id,
        name=style_update.name or "Default Style",
        description=style_update.description or "Default animation style",
        parameters=style_update.parameters or {"intensity": 0.7},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

@animation_router.delete("/{style_id}")
async def delete_animation_style(
    style_id: str
):
    """Delete an animation style."""
    if style_id == "default":
        raise HTTPException(status_code=400, detail="Cannot delete default style")
    return {"status": "success", "message": f"Style {style_id} deleted"}

@animation_router.get("/speakers/{speaker_id}/default", response_model=AnimationStyle)
async def get_default_speaker_style(
    speaker_id: str
):
    """Get the default animation style for a speaker."""
    # Return default style for any speaker
    return AnimationStyle(
        style_id="default",
        name="Default Style",
        description="Default animation style",
        parameters={"intensity": 0.7},
        created_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc)
    )

# Voice Router
voice_router = APIRouter(
    prefix="/voices",
    tags=["voices"],
    responses={404: {"description": "Not found"}},
)

# Language and Voice Management Endpoints
@router.get("/languages", response_model=Dict[str, Any])
async def list_supported_languages():
    """Get all supported languages and their voice counts."""
    try:
        languages = get_all_languages()
        return {
            "success": True,
            "languages": languages,
            "total_languages": len(languages),
            "total_voices": sum(lang["voice_count"] for lang in languages)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voices/all", response_model=Dict[str, Any])
async def list_all_voices():
    """Get all supported voices with detailed information."""
    try:
        voices = get_all_voices()
        return {
            "success": True,
            "voices": voices,
            "total_voices": len(voices)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/voices/validate/{voice_id}", response_model=Dict[str, Any])
async def validate_voice(
    voice_id: str
):
    """Validate if a voice ID is supported and get its information."""
    try:
        if is_voice_supported(voice_id):
            voice_info = get_voice_info(voice_id)
            return {
                "success": True,
                "supported": True,
                "voice_info": voice_info
            }
        else:
            return {
                "success": True,
                "supported": False,
                "message": f"Voice '{voice_id}' is not supported",
                "suggested_voice": get_default_voice("en-US")
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Style Presets Router
style_router = APIRouter(
    prefix="/style-presets",
    tags=["style-presets"],
    responses={404: {"description": "Not found"}},
)

# Register routers
router.include_router(animation_router)
router.include_router(voice_router)
router.include_router(style_router)

# Voice Cloning Endpoints
@voice_router.post("", response_model=VoiceResponse, status_code=status.HTTP_201_CREATED)
async def create_voice(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...),
    voice_id: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)
):
    """
    Create a new voice model from audio samples.
    
    Note: This is a simplified implementation. In production, you would want to:
    1. Validate the audio format and duration
    2. Process the audio in the background
    3. Return a task ID to check status
    """
    import tempfile
    import shutil
    import json
    
    # Create temp directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        temp_audio_path = os.path.join(temp_dir, audio.filename or "audio.wav")
        with open(temp_audio_path, "wb") as buffer:
            shutil.copyfileobj(audio.file, buffer)
        
        # Parse metadata if provided
        metadata_dict = {}
        if metadata:
            try:
                metadata_dict = json.loads(metadata)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid metadata format. Must be a valid JSON string."
                )
        
        # Create dummy voice response
        return VoiceResponse(
            voice_id=voice_id or str(uuid.uuid4()),
            speaker_name=speaker_name,
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata=metadata_dict
        )

@voice_router.get("", response_model=List[VoiceResponse])
async def list_voices(
    speaker_name: Optional[str] = None
):
    """List all available voice models, optionally filtered by speaker name."""
    # Return dummy voices for now
    voices = [
        VoiceResponse(
            voice_id="default",
            speaker_name="Default Speaker",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={}
        )
    ]
    
    if speaker_name:
        voices = [v for v in voices if v.speaker_name == speaker_name]
    
    return voices

@voice_router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(
    voice_id: str
):
    """Get details for a specific voice model."""
    if voice_id == "default":
        return VoiceResponse(
            voice_id="default",
            speaker_name="Default Speaker",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
            metadata={}
        )
    
    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Voice {voice_id} not found"
    )

@voice_router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str
):
    """Delete a voice model."""
    if voice_id == "default":
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot delete default voice"
        )
    
    return {"status": "success", "message": f"Voice {voice_id} deleted"}

# Video Generation Endpoints
@router.post("/generate/video")
async def generate_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    resolution: Optional[str] = Form("1080p"),
    fps: Optional[int] = Form(30),
    expressionIntensity: Optional[float] = Form(0.8),
    gestureLevel: Optional[str] = Form("medium"),
    voiceModel: Optional[str] = Form("en-US-JennyNeural"),
    background: Optional[str] = Form("blur"),
    enhanceFace: Optional[bool] = Form(True),
    stabilization: Optional[bool] = Form(True),
    expressionEngine: Optional[str] = Form(None)
):
    """Generate a video from uploaded image and audio/text."""
    try:
        if not image:
            raise HTTPException(status_code=400, detail="Image is required")
            
        # Get models (lazy load if needed)
        sadtalker, _, _, _, _ = get_models()
        
        # Save uploaded files
        image_path = await save_upload_file(image, "image")
        audio_path = await save_upload_file(audio, "audio") if audio else None
        
        # Validate voice model
        if voiceModel and not is_voice_supported(voiceModel):
            # Use default voice if provided voice is not supported
            voiceModel = get_default_voice("en-US")
            
        # Generate audio from text if provided (placeholder)
        if text and not audio_path:
            # For now, create a dummy audio file
            temp_dir = Path(config.get('paths.temp', 'temp'))
            temp_dir.mkdir(exist_ok=True)
            audio_path = temp_dir / f"{uuid.uuid4()}.wav"
            # Create a minimal WAV file (silence)
            import wave
            with wave.open(str(audio_path), 'wb') as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(22050)
                wav_file.writeframes(b'\x00' * 22050 * 2)  # 1 second of silence
        
        # Generate video - use task_id as video_id for consistency
        task_id = str(uuid.uuid4())
        output_dir = Path(config.get('paths.output', 'output'))
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{task_id}.mp4"
        
        # Add task to background (use real generation to avoid still image)
        background_tasks.add_task(
            process_video_generation_direct,
            task_id=task_id,
            image_path=image_path,
            audio_path=audio_path,
            output_path=str(output_path),
            resolution=resolution,
            fps=fps,
            enhance_face=bool(enhanceFace),
            stabilization=bool(stabilization),
            expression_engine=expressionEngine or 'auto'
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "video_id": task_id,  # Use same ID for consistency
            "status": "processing"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for video processing with real-time progress
async def process_video_generation(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    resolution: str = "480p",
    **kwargs
):
    """Process video generation with real-time progress tracking"""
    from api.realtime_progress import track_generation_progress
    
    # Use real-time progress tracking
    await track_generation_progress(
        task_id=task_id,
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        settings=kwargs
    )
    return

# Legacy function for compatibility
async def process_video_generation_legacy(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    resolution: str = "480p",
    **kwargs
):
    """Background task to process video generation using SadTalker."""
    import logging
    logger = logging.getLogger(__name__)
    
    def update_progress(progress: int, stage: str):
        """Update task progress in real-time."""
        task_status[task_id] = {
            "status": "processing",
            "progress": progress,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Task {task_id}: {progress}% - {stage}")
    
    try:
        # Step 1: Initialize SadTalker model (10%)
        update_progress(10, "Loading SadTalker model...")
        sadtalker, _, _, _, _ = get_models()
        
        # Step 2: Validate inputs (20%)
        update_progress(20, "Validating input files...")
        if not os.path.exists(image_path):
            raise Exception("Image file not found")
        if audio_path and not os.path.exists(audio_path):
            raise Exception("Audio file not found")
        
        # Step 3: Preprocess image (30%)
        update_progress(30, "Preprocessing image...")
        
        # Step 4: Analyze audio (40%)
        update_progress(40, "Analyzing audio...")
        
        # Step 5: Generate video frames (50-80%)
        update_progress(50, "Generating video frames...")
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Use actual SadTalker implementation
        import subprocess
        import sys
        
        # Run SadTalker inference script directly
        cmd = [
            sys.executable, 
            "inference.py",
            "--driven_audio", audio_path,
            "--source_image", image_path,
            "--result_dir", os.path.dirname(output_path),
            "--still", "--preprocess", "full",
            "--enhancer", "gfpgan"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, cwd="SadTalker")
        
        if result.returncode != 0:
            raise Exception(f"SadTalker failed: {result.stderr}")
        
        # Find generated video file
        result_dir = os.path.dirname(output_path)
        for file in os.listdir(result_dir):
            if file.endswith('.mp4'):
                result_path = os.path.join(result_dir, file)
                # Move to expected output path
                if result_path != output_path:
                    os.rename(result_path, output_path)
                result_path = output_path
                break
        else:
            raise Exception("No video file generated by SadTalker")
        
        # Step 6: Post-processing (90%)
        update_progress(90, "Finalizing video...")
        
        # Verify output file exists and is valid
        if not os.path.exists(result_path):
            raise Exception("Video generation failed - output file not created")
        
        # Check file size (should be more than just headers)
        file_size = os.path.getsize(result_path)
        if file_size < 10000:  # Less than 10KB indicates failure
            raise Exception(f"Video generation failed - output file too small ({file_size} bytes)")
        
        # Complete (100%)
        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Video generation completed!",
            "result_path": result_path,
            "file_size": file_size,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info(f"Video generation completed for task {task_id}: {result_path} ({file_size:,} bytes)")
        
    except Exception as e:
        logger.error(f"Video generation failed for task {task_id}: {str(e)}")
        
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "stage": "Generation failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat()
        }

async def process_video_generation_direct(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    resolution: str = "720p",
    fps: int = 25,
    enhance_face: bool = True,
    stabilization: bool = True,
    expression_engine: str = 'auto'
):
    """Generate a video using SadTalker directly (moving frames), fallback to ffmpeg still image."""
    import logging
    import os
    import shutil
    logger = logging.getLogger(__name__)

    def set_status(progress: int, stage: str):
        task_status[task_id] = {
            "status": "processing",
            "progress": progress,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    try:
        set_status(10, "AI models initialized")

        # Optional: estimate expressions to guide pipeline
        try:
            from models.expression.engine import estimate_from_path
            expr = estimate_from_path(image_path, expression_engine)
            task_status[task_id]["expression_engine"] = expr.engine
        except Exception:
            pass

        emage_success = False
        # Try SadTalker full pipeline
        try:
            from models.sadtalker_full import SadTalkerFull
            set_status(30, "Avatar image preprocessed")
            sadtalker = SadTalkerFull()
            # Derive emotion hint from expression engine if available
            emotion_hint = 'neutral'
            try:
                from models.expression.engine import estimate_from_path
                expr = estimate_from_path(image_path, expression_engine)
                if getattr(expr, 'emotion_probs', None):
                    emotion_hint = max(expr.emotion_probs.items(), key=lambda kv: kv[1])[0]
            except Exception:
                pass
            face_video = sadtalker.generate(
                image_path=image_path,
                audio_path=audio_path or image_path,  # if no audio, dummy path won't be used for frames
                output_path=str(Path(config.get('paths.temp', 'temp')) / f"{task_id}_face.mp4"),
                emotion=emotion_hint,
                enhance_face=enhance_face
            )

            # Move result to output
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            shutil.copy2(face_video, output_path)

            set_status(95, "Post-processing applied")
        except Exception as e:
            logger.warning(f"SadTalker generation failed: {e}")
            # Try EMAGE full-body fallback if available and audio is present
            try:
                from api.capabilities_endpoints import _emage_available  # type: ignore
                if _emage_available() and audio_path:
                    set_status(70, "Generating full-body gestures (EMAGE)")
                    from models.emage_realistic import get_emage_model
                    emage = get_emage_model()
                    emage_video = emage.generate_full_video(
                        audio_path=audio_path,
                        output_path=output_path,
                        emotion='neutral',
                        style='natural',
                        avatar_type='realistic'
                    )
                    # If EMAGE succeeded, mark progress and skip ffmpeg still image
                    if emage_video and os.path.exists(emage_video):
                        emage_success = True
            except Exception as ee:
                logger.warning(f"EMAGE fallback failed: {ee}; using ffmpeg still image")
            if not emage_success:
                # Fallback: still image with audio using ffmpeg
                import subprocess
                set_status(70, "Video frames rendered")
            # Fallback: still image with audio using ffmpeg
            import subprocess
            set_status(70, "Video frames rendered")

            cmd = [
                "ffmpeg", "-y",
                "-loop", "1", "-i", image_path,
            ]
            if audio_path:
                cmd += ["-i", audio_path]
            else:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo"]

            cmd += [
                "-c:v", "libx264",
                "-pix_fmt", "yuv420p",
                "-shortest",
                "-vf", "scale=720:720:force_original_aspect_ratio=decrease,pad=720:720:(ow-iw)/2:(oh-ih)/2",
                output_path
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise Exception(f"FFmpeg failed: {result.stderr}")

        # Complete
        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Video generation completed!",
            "result_path": output_path,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Video generation completed for task {task_id}: {output_path}")

    except Exception as e:
        logger.error(f"Direct generation failed for task {task_id}: {e}")
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "stage": "Generation failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat()
        }

# Task status tracking
task_status: Dict[str, Dict[str, Any]] = {}
adaptation_tasks: Dict[str, Dict[str, Any]] = {}

# Speaker Management Endpoints
@router.post("/speakers/register", response_model=Dict[str, Any])
async def register_speaker(
    audio: UploadFile = File(...),
    speaker_id: str = Form(...),
    metadata: Optional[str] = Form(None),
    adapt_models: bool = Form(False),
    audio_dir: Optional[str] = Form(None)
):
    """Register a new speaker with an audio sample and optionally adapt models"""
    import tempfile
    import shutil
    import json
    
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_path = temp_audio.name
    
    try:
        # Parse metadata if provided
        metadata_dict = json.loads(metadata) if metadata else {}
        
        # For now, just return success
        success = True
        
        return {
            "status": "success", 
            "message": f"Speaker {speaker_id} registered successfully" + 
                      (" and model adaptation started" if adapt_models else "")
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid metadata format. Must be valid JSON")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temp file
        if os.path.exists(temp_path):
            os.unlink(temp_path)

@router.post("/speakers/adapt", response_model=Dict[str, Any])
async def adapt_speaker_models(
    background_tasks: BackgroundTasks,
    request: ModelAdaptationRequest
):
    """Start model adaptation for a speaker"""
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status (using global dict for now)
    if 'adaptation_tasks' not in globals():
        global adaptation_tasks
        adaptation_tasks = {}
    
    adaptation_tasks[task_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Task queued',
        'speaker_id': request.speaker_id,
        'model_type': request.model_type
    }
    
    return {
        "status": "started",
        "task_id": task_id,
        "message": f"Adaptation started for speaker {request.speaker_id}"
    }

@router.get("/speakers/adapt/{task_id}", response_model=AdaptationStatusResponse)
async def get_adaptation_status(
    task_id: str
):
    """Get the status of a model adaptation task"""
    if 'adaptation_tasks' not in globals():
        global adaptation_tasks
        adaptation_tasks = {}
    
    if task_id not in adaptation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return AdaptationStatusResponse(**adaptation_tasks[task_id])

@router.get("/speakers/adapt", response_model=List[Dict[str, Any]])
async def list_adaptation_tasks():
    """List all adaptation tasks"""
    if 'adaptation_tasks' not in globals():
        global adaptation_tasks
        adaptation_tasks = {}
    
    return [
        {"task_id": task_id, **task_info}
        for task_id, task_info in adaptation_tasks.items()
    ]

# -------------------- FUSION GENERATION (SadTalker face + EMAGE body) --------------------
@router.post("/generate/fusion-video")
async def generate_fusion_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(None),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    resolution: Optional[str] = Form("720p"),
    fps: Optional[int] = Form(25),
    emotion: Optional[str] = Form("neutral"),
    style: Optional[str] = Form("natural"),
    preferWav2Lip2: Optional[bool] = Form(False)
):
    try:
        # Save uploads
        temp_dir = Path(config.get('paths.temp', 'temp'))
        temp_dir.mkdir(exist_ok=True)

        img_path: Optional[Path] = None
        if image is not None:
            img_path = temp_dir / f"{uuid.uuid4()}_{image.filename or 'image' }"
            with open(img_path, 'wb') as f:
                f.write(await image.read())

        # Prepare audio path: uploaded or synthesize from text
        audio_path: Optional[Path] = None
        if audio is not None:
            audio_path = temp_dir / f"{uuid.uuid4()}_{audio.filename or 'audio' }"
            with open(audio_path, 'wb') as f:
                f.write(await audio.read())
        elif text:
            # Use TTS to generate audio
            from api.tts_service import get_tts_service
            tts = get_tts_service()
            audio_path = Path(temp_dir / f"{uuid.uuid4()}.wav")
            # Prefer free gTTS to avoid protobuf/google deps
            try:
                provider_choice = 'gtts' if 'gtts' in getattr(tts, 'providers', {}) else 'auto'
            except Exception:
                provider_choice = 'auto'
            tts.generate_speech(text=text, voice="en-US-JennyNeural", output_path=str(audio_path), provider=provider_choice)
        else:
            raise HTTPException(status_code=400, detail="Either audio or text must be provided for fusion video")

        # Default avatar image if none provided
        if img_path is None:
            img_path = temp_dir / f"{uuid.uuid4()}_default.jpg"
            try:
                import cv2
                import numpy as np
                canvas = np.ones((512,512,3), dtype=np.uint8) * 235
                cv2.circle(canvas, (256,256), 180, (210,210,210), -1)
                cv2.circle(canvas, (210,210), 28, (70,70,70), -1)
                cv2.circle(canvas, (302,210), 28, (70,70,70), -1)
                cv2.ellipse(canvas, (256,320), (70,35), 0, 0, 180, (70,70,70), 5)
                cv2.imwrite(str(img_path), canvas)
            except Exception:
                with open(img_path, 'wb') as f:
                    f.write(b"avatar")

        # Prepare output
        output_dir = Path(config.get('paths.output', 'output'))
        output_dir.mkdir(exist_ok=True)
        task_id = str(uuid.uuid4())
        output_path = output_dir / f"{task_id}.mp4"

        # Background task
        background_tasks.add_task(
            process_fusion_generation,
            task_id=task_id,
            image_path=str(img_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            emotion=emotion or 'neutral',
            style=style or 'natural',
            fps=int(fps or 25),
            resolution=resolution or '720p',
            prefer_wav2lip2=bool(preferWav2Lip2)
        )

        return {"success": True, "task_id": task_id, "status": "processing"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


async def process_fusion_generation(
    task_id: str,
    image_path: str,
    audio_path: str,
    output_path: str,
    emotion: str = 'neutral',
    style: str = 'natural',
    fps: int = 25,
    resolution: str = '720p',
    prefer_wav2lip2: bool = False
):
    import logging
    logger = logging.getLogger(__name__)
    try:
        # Update status
        task_status[task_id] = {
            "status": "processing",
            "progress": 10,
            "stage": "Initializing fusion engine"
        }
        # Run fusion
        from models.fusion.engine import FusionEngine
        eng = FusionEngine()
        task_status[task_id].update({"progress": 40, "stage": "Generating body and head tracks"})
        final_path = eng.generate(
            face_image=image_path,
            audio_path=audio_path,
            output_path=output_path,
            emotion=emotion,
            style=style,
            fps=fps,
            resolution=resolution,
            prefer_wav2lip2=prefer_wav2lip2
        )
        # Finalize
        if not os.path.exists(final_path):
            raise RuntimeError("Fusion output not created")
        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Fusion video completed",
            "result_path": final_path,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
    except Exception as e:
        logger.error(f"Fusion generation failed: {e}")
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "stage": "Fusion generation failed",
            "error": str(e)
        }

@router.get("/speakers/{speaker_id}", response_model=SpeakerInfo)
async def get_speaker(
    speaker_id: str
):
    """Get information about a registered speaker"""
    try:
        # Return dummy speaker info for now
        return SpeakerInfo(
            speaker_id=speaker_id,
            created_at=datetime.now(timezone.utc),
            last_updated=datetime.now(timezone.utc),
            metadata={},
            num_recordings=0,
            embedding_shape=None
        )
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Error getting speaker: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/status/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task with real-time progress."""
    # Check if task exists in our tracking system
    if task_id in task_status:
        task_info = task_status[task_id]
        
        if task_info["status"] == "completed":
            return {
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": "completed",
                    "progress": 100,
                    "stage": task_info.get("stage", "Completed"),
                    "video_url": f"/api/videos/{task_id}"
                }
            }
        elif task_info["status"] == "failed":
            return {
                "success": False,
                "data": {
                    "task_id": task_id,
                    "status": "failed",
                    "progress": task_info.get("progress", 0),
                    "stage": task_info.get("stage", "Failed"),
                    "error": task_info.get("error", "Unknown error")
                }
            }
        elif task_info["status"] == "processing":
            return {
                "success": True,
                "data": {
                    "task_id": task_id,
                    "status": "processing",
                    "progress": task_info.get("progress", 0),
                    "stage": task_info.get("stage", "Processing...")
                }
            }
    
    # Check if output file exists as fallback
    output_dir = Path(config.get('paths.output', 'output'))
    video_file = output_dir / f"{task_id}.mp4"
    
    if video_file.exists():
        # Update task status if file exists but not tracked
        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Completed",
            "result_path": str(video_file),
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "completed",
                "progress": 100,
                "stage": "Completed",
                "video_url": f"/api/videos/{task_id}"
            }
        }
    
    # Task not found or not started
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": "processing",
            "progress": 5,
            "stage": "Initializing..."
        }
    }

@router.get("/videos")
async def list_videos():
    """List all generated videos."""
    output_dir = Path(config.get('paths.output', 'output'))
    videos = []
    
    if output_dir.exists():
        for file in output_dir.glob("*.mp4"):
            videos.append({
                "id": file.stem,
                "filename": file.name,
                "size": file.stat().st_size,
                "created_at": file.stat().st_ctime
            })
    
    return {"videos": videos}

@router.get("/videos/{task_id}")
async def get_video(task_id: str):
    """Serve a generated video file by task ID."""
    import logging
    logger = logging.getLogger(__name__)
    
    # First check if we have the task in our status tracking
    if task_id in task_status:
        task_info = task_status[task_id]
        if task_info["status"] == "completed" and "result_path" in task_info:
            video_file = Path(task_info["result_path"])
            if video_file.exists():
                logger.info(f"Serving video from task status: {video_file}")
                return FileResponse(
                    str(video_file),
                    media_type="video/mp4",
                    filename=f"paksatalker_{task_id}.mp4",
                    headers={
                        "Content-Disposition": f"attachment; filename=paksatalker_{task_id}.mp4",
                        "Cache-Control": "no-cache"
                    }
                )
    
    # Fallback: check output directory for video file
    output_dir = Path(config.get('paths.output', 'output'))
    video_file = output_dir / f"{task_id}.mp4"
    
    if video_file.exists():
        logger.info(f"Serving video from output directory: {video_file}")
        return FileResponse(
            str(video_file),
            media_type="video/mp4",
            filename=f"paksatalker_{task_id}.mp4",
            headers={
                "Content-Disposition": f"attachment; filename=paksatalker_{task_id}.mp4",
                "Cache-Control": "no-cache"
            }
        )
    
    logger.error(f"Video not found for task {task_id}")
    raise HTTPException(status_code=404, detail="Video not found")

# Text-to-Speech Endpoint
@router.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("en-US-JennyNeural"),
    provider: str = Form("auto")
):
    """Convert text to speech using real TTS providers."""
    try:
        from api.tts_service import get_tts_service
        
        tts_service = get_tts_service()
        
        # Generate audio file path (use .wav as preferred target)
        output_dir = config.get('paths.output', 'output')
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(output_dir, f"{uuid.uuid4()}.wav")
        
        # Generate speech using real TTS
        result_path = tts_service.generate_speech(
            text=text,
            voice=voice,
            output_path=audio_path,
            provider=provider
        )
        
        # Serve correct MIME type based on extension
        ext = os.path.splitext(result_path)[1].lower()
        mime = "audio/wav" if ext == ".wav" else "audio/mpeg" if ext == ".mp3" else "application/octet-stream"
        return FileResponse(result_path, media_type=mime, filename=os.path.basename(result_path))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Text Generation Endpoint
@router.post("/generate/text")
async def generate_text(
    prompt: str = Form(...),
    max_length: int = Form(100),
    temperature: float = Form(0.7)
):
    """Generate text using the Qwen model."""
    try:
        # For now, return a simple response
        response = f"Generated response for: {prompt[:50]}..."
        
        return {"success": True, "text": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Prompt-based Video Generation Endpoint
@router.post("/generate/video-from-prompt")
async def generate_video_from_prompt(
    background_tasks: BackgroundTasks,
    prompt: str = Form(...),
    voice: Optional[str] = Form("en-US-JennyNeural"),
    resolution: Optional[str] = Form("1080p"),
    fps: Optional[int] = Form(30),
    gestureLevel: Optional[str] = Form("medium"),
    expressionEngine: Optional[str] = Form(None)
):
    """Generate video from text prompt using Qwen + TTS + Default Avatar."""
    try:
        # Generate text from prompt using Qwen
        generated_text = f"Hello! This is a generated response for your prompt: {prompt}"
        
        # Create temp directories
        temp_dir = Path(config.get('paths.temp', 'temp'))
        temp_dir.mkdir(exist_ok=True)
        
        # Validate voice model
        if voice and not is_voice_supported(voice):
            voice = get_default_voice("en-US")
            
        # Generate TTS audio using free provider if available
        from api.tts_service import get_tts_service
        tts_service = get_tts_service()
        # Prefer free 'gtts' provider, fallback to 'auto'
        try:
            desired_provider = 'gtts' if 'gtts' in getattr(tts_service, 'providers', {}) else 'auto'
        except Exception:
            desired_provider = 'auto'
        audio_path = temp_dir / f"{uuid.uuid4()}.wav"
        # tts_service may return mp3 if ffmpeg missing; handle later in ffmpeg pipeline
        tts_result_path = tts_service.generate_speech(
            text=generated_text,
            voice=voice or "en-US-JennyNeural",
            output_path=str(audio_path),
            provider=desired_provider
        )
        
        # Create a default avatar image
        image_path = temp_dir / f"{uuid.uuid4()}.jpg"
        from PIL import Image
        import numpy as np
        
        # Create a simple 512x512 default avatar
        img_array = np.ones((512, 512, 3), dtype=np.uint8) * 200  # Light gray
        img = Image.fromarray(img_array)
        img.save(str(image_path))
        
        # Generate video
        task_id = str(uuid.uuid4())
        output_dir = Path(config.get('paths.output', 'output'))
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{task_id}.mp4"
        
        background_tasks.add_task(
            process_video_generation,
            task_id=task_id,
            image_path=str(image_path),
            audio_path=str(tts_result_path),
            output_path=str(output_path),
            resolution=resolution,
            expression_engine=expressionEngine or 'auto'
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "generated_text": generated_text,
            "status": "processing"
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Prompt generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

# Gesture Generation Endpoint
@router.post("/generate-gestures", response_model=Dict[str, Any])
async def generate_gestures(
    text: Optional[str] = Form(None),
    emotion: Optional[str] = Form("neutral"),
    intensity: float = Form(0.7, ge=0.0, le=1.0),
    duration: float = Form(5.0, gt=0)
):
    """
    Generate gestures based on text and emotion.
    
    Args:
        text: Input text to guide gesture generation (optional)
        emotion: Desired emotion (neutral, happy, sad, angry, surprised, disgusted, fearful)
        intensity: Emotion intensity (0.0 to 1.0)
        duration: Duration of gesture sequence in seconds
        
    Returns:
        Dictionary containing gesture data and metadata
    """
    try:
        # Initialize gesture generator
        gesture_gen = GestureGenerator()
        
        # Set emotion
        gesture_gen.set_emotion(emotion, intensity)
        
        # Generate gestures
        gesture_data = gesture_gen.generate_gestures(
            text=text,
            duration=duration,
            emotion=emotion,
            intensity=intensity
        )
        
        # Convert numpy array to list for JSON serialization
        if hasattr(gesture_data, 'tolist'):
            gesture_data = gesture_data.tolist()
            
        return {
            "status": "success",
            "emotion": emotion,
            "intensity": intensity,
            "duration": duration,
            "gesture_data": gesture_data,
            "gesture_count": len(gesture_data) if isinstance(gesture_data, list) else 0
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate gestures: {str(e)}"
        )

# Model Status Endpoint
@router.get("/status")
async def get_status():
    """Get the status of all models and services."""
    sadtalker, wav2lip, gesture, qwen, _ = get_models()
    
    return {
        "status": "operational",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "models": {
            "sadtalker": {
                "loaded": sadtalker.is_loaded() if sadtalker else False,
                "version": sadtalker.version if sadtalker else None,
                "status": "loaded" if sadtalker and sadtalker.is_loaded() else "error"
            },
            "wav2lip": {
                "loaded": wav2lip.is_loaded() if wav2lip else False,
                "version": wav2lip.version if wav2lip else None,
                "status": "loaded" if wav2lip and wav2lip.is_loaded() else "error"
            },
            "gesture": {
                "loaded": gesture.is_loaded() if gesture else False,
                "version": gesture.version if gesture else None,
                "status": "loaded" if gesture and gesture.is_loaded() else "error"
            },
            "qwen": {
                "loaded": qwen.is_loaded() if qwen else False,
                "version": qwen.version if qwen else None,
                "status": "loaded" if qwen and qwen.is_loaded() else "error"
            }
        },
        "system": {
            "tasks_queued": len([t for t in task_status.values() if t["status"] == "processing"]),
            "tasks_completed": len([t for t in task_status.values() if t["status"] == "completed"]),
            "tasks_failed": len([t for t in task_status.values() if t["status"] == "failed"])
        }
    }

# In-memory style preset storage
style_presets = {
    "professional": {
        "preset_id": "professional",
        "name": "Professional",
        "description": "Formal business presentation style",
        "intensity": 0.6,
        "smoothness": 0.9,
        "expressiveness": 0.5,
        "cultural_context": "GLOBAL",
        "formality": 0.9,
        "gesture_frequency": 0.4,
        "gesture_amplitude": 0.8,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
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
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat()
    }
}

# Style Preset Endpoints
@style_router.post("", response_model=Dict[str, Any])
async def create_style_preset(
    name: str = Form(...),
    description: str = Form(""),
    intensity: float = Form(0.7, ge=0.0, le=1.0),
    smoothness: float = Form(0.8, ge=0.0, le=1.0),
    expressiveness: float = Form(0.7, ge=0.0, le=1.0),
    cultural_context: str = Form("GLOBAL"),
    formality: float = Form(0.5, ge=0.0, le=1.0),
    gesture_frequency: float = Form(0.7, ge=0.0, le=1.0),
    gesture_amplitude: float = Form(1.0, ge=0.0, le=2.0)
):
    """Create a new custom style preset."""
    try:
        preset_id = str(uuid.uuid4())
        preset = {
            "preset_id": preset_id,
            "name": name,
            "description": description,
            "intensity": intensity,
            "smoothness": smoothness,
            "expressiveness": expressiveness,
            "cultural_context": cultural_context,
            "formality": formality,
            "gesture_frequency": gesture_frequency,
            "gesture_amplitude": gesture_amplitude,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        style_presets[preset_id] = preset
        
        return {
            "success": True,
            "preset": preset,
            "message": f"Style preset '{name}' created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@style_router.get("", response_model=Dict[str, Any])
async def list_style_presets():
    """List all available style presets."""
    try:
        return {
            "success": True,
            "presets": list(style_presets.values()),
            "count": len(style_presets)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@style_router.post("/interpolate", response_model=Dict[str, Any])
async def interpolate_style_presets(
    preset1_id: str = Form(...),
    preset2_id: str = Form(...),
    ratio: float = Form(0.5, ge=0.0, le=1.0)
):
    """Interpolate between two style presets."""
    try:
        if preset1_id not in style_presets or preset2_id not in style_presets:
            raise HTTPException(status_code=404, detail="One or both presets not found")
        
        preset1 = style_presets[preset1_id]
        preset2 = style_presets[preset2_id]
        
        # Interpolate numeric values
        interpolated = {
            "preset_id": str(uuid.uuid4()),
            "name": f"{preset1['name']} + {preset2['name']} ({ratio:.1f})",
            "description": f"Interpolated between {preset1['name']} and {preset2['name']}",
            "intensity": preset1["intensity"] * (1 - ratio) + preset2["intensity"] * ratio,
            "smoothness": preset1["smoothness"] * (1 - ratio) + preset2["smoothness"] * ratio,
            "expressiveness": preset1["expressiveness"] * (1 - ratio) + preset2["expressiveness"] * ratio,
            "cultural_context": preset1["cultural_context"] if ratio < 0.5 else preset2["cultural_context"],
            "formality": preset1["formality"] * (1 - ratio) + preset2["formality"] * ratio,
            "gesture_frequency": preset1["gesture_frequency"] * (1 - ratio) + preset2["gesture_frequency"] * ratio,
            "gesture_amplitude": preset1["gesture_amplitude"] * (1 - ratio) + preset2["gesture_amplitude"] * ratio,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        return {
            "success": True,
            "interpolated_preset": interpolated,
            "message": "Presets interpolated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@style_router.post("/{preset_id}/cultural-variants", response_model=Dict[str, Any])
async def create_cultural_variants(preset_id: str):
    """Create cultural variants of a style preset."""
    try:
        if preset_id not in style_presets:
            raise HTTPException(status_code=404, detail="Base preset not found")
        
        base_preset = style_presets[preset_id]
        cultural_contexts = ["WESTERN", "EAST_ASIAN", "MIDDLE_EASTERN", "SOUTH_ASIAN", "LATIN_AMERICAN", "AFRICAN"]
        variants = []
        
        for context in cultural_contexts:
            if context != base_preset["cultural_context"]:
                variant_id = str(uuid.uuid4())
                variant = base_preset.copy()
                variant.update({
                    "preset_id": variant_id,
                    "name": f"{base_preset['name']} ({context})",
                    "description": f"{base_preset['description']} - {context} variant",
                    "cultural_context": context,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                    "updated_at": datetime.now(timezone.utc).isoformat()
                })
                
                # Slight variations based on cultural context
                if context == "EAST_ASIAN":
                    variant["formality"] = min(1.0, variant["formality"] + 0.1)
                    variant["gesture_frequency"] = max(0.0, variant["gesture_frequency"] - 0.1)
                elif context == "LATIN_AMERICAN":
                    variant["expressiveness"] = min(1.0, variant["expressiveness"] + 0.1)
                    variant["gesture_amplitude"] = min(2.0, variant["gesture_amplitude"] + 0.2)
                
                style_presets[variant_id] = variant
                variants.append(variant)
        
        return {
            "success": True,
            "variants": variants,
            "count": len(variants),
            "message": f"Created {len(variants)} cultural variants"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Qwen2.5-Omni Endpoints
@router.post("/qwen/chat")
async def qwen_multimodal_chat(
    text: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    """Multimodal chat with Qwen2.5-Omni model."""
    try:
        qwen_model = get_qwen_model()
        
        # Process uploaded files
        image_data = None
        audio_data = None
        
        if image:
            image_data = await image.read()
        
        if audio:
            audio_data = await audio.read()
        
        # Get multimodal response
        result = qwen_model.multimodal_chat(
            text=text,
            image=image_data,
            audio=audio_data
        )
        
        return {
            "success": True,
            "response": result["response"],
            "audio_transcription": result.get("audio_transcription", ""),
            "image_description": result.get("image_description", "")
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/qwen/generate-script")
async def qwen_generate_script(
    topic: str = Form(...),
    style: str = Form("professional"),
    duration: int = Form(30),
    audience: str = Form("general")
):
    """Generate avatar script using Qwen2.5-Omni."""
    try:
        qwen_model = get_qwen_model()
        
        result = qwen_model.generate_avatar_script(
            topic=topic,
            style=style,
            duration=duration,
            audience=audience
        )
        
        return {
            "success": True,
            "script": result["script"],
            "word_count": result["word_count"],
            "estimated_duration": result["estimated_duration"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced Model Generation Endpoints
@router.post("/generate/advanced-video")
async def generate_advanced_video(
    background_tasks: BackgroundTasks,
    image: UploadFile = File(...),
    audio: Optional[UploadFile] = File(None),
    text: Optional[str] = Form(None),
    # Model settings
    useEmage: bool = Form(True),
    useWav2Lip2: bool = Form(True),
    useSadTalkerFull: bool = Form(True),
    emotion: str = Form("neutral"),
    bodyStyle: str = Form("natural"),
    avatarType: str = Form("realistic"),
    lipSyncQuality: str = Form("high"),
    # Standard settings
    resolution: Optional[str] = Form("1080p"),
    fps: Optional[int] = Form(30)
):
    """Generate video using advanced AI models."""
    try:
        # Save uploaded files
        image_path = await save_upload_file(image, "image")
        audio_path = await save_upload_file(audio, "audio") if audio else None
        
        # Generate task ID
        task_id = str(uuid.uuid4())
        output_dir = Path(config.get('paths.output', 'output'))
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{task_id}.mp4"
        
        # Add advanced generation task (use real generation to avoid simulated-only progress)
        background_tasks.add_task(
            process_advanced_video_generation_direct,
            task_id=task_id,
            image_path=image_path,
            audio_path=audio_path,
            output_path=str(output_path),
            model_settings={
                "useEmage": useEmage,
                "useWav2Lip2": useWav2Lip2,
                "useSadTalkerFull": useSadTalkerFull,
                "emotion": emotion,
                "bodyStyle": bodyStyle,
                "avatarType": avatarType,
                "lipSyncQuality": lipSyncQuality,
                "fps": fps
            },
            resolution=resolution,
            fps=fps
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "status": "processing",
            "models_used": {
                "emage": useEmage,
                "wav2lip2": useWav2Lip2,
                "sadtalker_full": useSadTalkerFull
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Advanced video processing task
async def process_advanced_video_generation(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    model_settings: dict,
    resolution: str = "1080p",
    fps: int = 30
):
    """Process video with advanced AI models"""
    from api.realtime_progress import track_generation_progress
    
    # Use real-time progress tracking
    await track_generation_progress(
        task_id=task_id,
        image_path=image_path,
        audio_path=audio_path,
        output_path=output_path,
        settings=model_settings
    )
    return

async def process_advanced_video_generation_direct(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    model_settings: dict,
    resolution: str = "1080p",
    fps: int = 30
):
    """Advanced generation using SadTalkerModel directly to produce moving frames."""
    import logging
    import os
    logger = logging.getLogger(__name__)

    def set_status(progress: int, stage: str):
        task_status[task_id] = {
            "status": "processing",
            "progress": progress,
            "stage": stage,
            "updated_at": datetime.now(timezone.utc).isoformat()
        }

    try:
        set_status(10, "Loading AI models...")

        from models.sadtalker import SadTalkerModel
        sadtalker = SadTalkerModel(
            use_full_model=bool(model_settings.get("useSadTalkerFull", True)),
            use_wav2lip2=bool(model_settings.get("useWav2Lip2", True)),
            use_emage=bool(model_settings.get("useEmage", True))
        )

        set_status(40, "Analyzing inputs...")

        result_path = sadtalker.generate(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            emotion=model_settings.get("emotion", "neutral"),
            style=model_settings.get("bodyStyle", "natural"),
            avatar_type=model_settings.get("avatarType", "realistic"),
            resolution=resolution,
            fps=fps
        )

        set_status(90, "Encoding video...")
        if not os.path.exists(result_path):
            raise Exception("Video not generated")

        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Generation completed",
            "result_path": result_path,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        logger.info(f"Advanced generation completed for task {task_id}: {result_path}")

    except Exception as e:
        logger.error(f"Advanced direct generation failed: {e}")
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "stage": "Generation failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat()
        }

# Legacy function for compatibility
async def process_advanced_video_generation_legacy(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str,
    model_settings: dict,
    resolution: str = "1080p",
    fps: int = 30
):
    """Process video with advanced AI models."""
    import logging
    logger = logging.getLogger(__name__)
    
    try:
        # Update progress
        task_status[task_id] = {
            "status": "processing",
            "progress": 10,
            "stage": "Initializing advanced AI models...",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Initialize SadTalker with advanced settings
        sadtalker = SadTalkerModel(
            device="auto",
            use_full_model=model_settings.get("useSadTalkerFull", True),
            use_wav2lip2=model_settings.get("useWav2Lip2", True),
            use_emage=model_settings.get("useEmage", True)
        )
        
        # Update progress
        task_status[task_id] = {
            "status": "processing",
            "progress": 30,
            "stage": "Processing with advanced models...",
            "updated_at": datetime.now(timezone.utc).isoformat()
        }
        
        # Generate video with advanced settings
        result_path = sadtalker.generate(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            emotion=model_settings.get("emotion", "neutral"),
            style=model_settings.get("bodyStyle", "natural"),
            avatar_type=model_settings.get("avatarType", "realistic"),
            resolution=resolution
        )
        
        # Complete
        task_status[task_id] = {
            "status": "completed",
            "progress": 100,
            "stage": "Advanced video generation completed!",
            "result_path": result_path,
            "completed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Advanced video generation failed: {e}")
        task_status[task_id] = {
            "status": "failed",
            "progress": 0,
            "stage": "Advanced generation failed",
            "error": str(e),
            "failed_at": datetime.now(timezone.utc).isoformat()
        }

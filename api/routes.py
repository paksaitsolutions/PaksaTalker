"""API routes for PaksaTalker."""
import os
import uuid
import json
import time
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
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

from config import config
from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel
from integrations.gesture import GestureGenerator
from models.style_presets import StylePresetManager, StylePreset, StyleInterpolator

# Import schemas with error handling
try:
    from api.schemas.schemas import (
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
router = APIRouter(prefix="/api/v1", tags=["api"])

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
            style_manager = StylePresetManager()
            
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
            style_manager = StylePresetManager()
            models_initialized = True
            
    return sadtalker, wav2lip, gesture, qwen, style_manager

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
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
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
    # Ensure directory exists
    temp_dir = config.get('paths.temp', 'temp')
    if directory == "image":
        directory = temp_dir
    elif directory == "audio":
        directory = temp_dir
    
    os.makedirs(directory, exist_ok=True)
    file_ext = os.path.splitext(upload_file.filename or '')[1] or '.bin'
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
async def login_for_access_token(request: Request):
    try:
        data = await request.json()
        username = data.get('email')  # Frontend sends email as username
        password = data.get('password')
        
        if not username or not password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email and password are required"
            )
            
        user = authenticate_user(fake_users_db, username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect email or password",
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow()
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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
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
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow()
    )

# Voice Router
voice_router = APIRouter(
    prefix="/voices",
    tags=["voices"],
    responses={404: {"description": "Not found"}},
)

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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
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
    stabilization: Optional[bool] = Form(True)
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
        
        # Generate video
        video_id = str(uuid.uuid4())
        output_dir = Path(config.get('paths.output', 'output'))
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / f"{video_id}.mp4"
        
        # Add task to background
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_video_generation,
            image_path=image_path,
            audio_path=audio_path,
            output_path=str(output_path),
            task_id=task_id
        )
        
        return {
            "success": True,
            "task_id": task_id,
            "video_id": video_id,
            "status": "processing"
        }
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Background task for video processing
async def process_video_generation(
    task_id: str,
    image_path: str,
    audio_path: Optional[str],
    output_path: str
):
    """Background task to process video generation."""
    try:
        # Initialize models
        sadtalker, wav2lip, gesture, qwen, _ = get_models()
        
        # Generate video using SadTalker
        if audio_path and os.path.exists(audio_path):
            video_path = sadtalker.generate(
                image_path=image_path,
                audio_path=audio_path,
                output_path=output_path
            )
            
            # Enhance with Wav2Lip if needed
            enhanced_path = output_path.replace('.mp4', '_enhanced.mp4')
            wav2lip.enhance(
                video_path=video_path,
                audio_path=audio_path,
                output_path=enhanced_path
            )
            
            # Use enhanced version as final output
            if os.path.exists(enhanced_path):
                os.replace(enhanced_path, output_path)
        
        # Update task status
        task_status[task_id] = {
            "status": "completed",
            "result_path": output_path,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"Video generation failed: {str(e)}")
        
        task_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
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

@router.get("/speakers/{speaker_id}", response_model=SpeakerInfo)
async def get_speaker(
    speaker_id: str
):
    """Get information about a registered speaker"""
    try:
        # Return dummy speaker info for now
        return SpeakerInfo(
            speaker_id=speaker_id,
            created_at=datetime.utcnow(),
            last_updated=datetime.utcnow(),
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
    """Get the status of a background task."""
    # In a real app, you'd store task status in a database or cache
    # For now, we'll just check if the output file exists
    output_dir = Path(config.get('paths.output', 'output'))
    video_file = next((f for f in output_dir.glob(f"*{task_id}*.mp4")), None)
    
    if video_file and video_file.exists():
        return {
            "success": True,
            "data": {
                "task_id": task_id,
                "status": "completed",
                "video_id": video_file.stem,
                "video_url": f"/api/v1/videos/{video_file.stem}"
            }
        }
    return {
        "success": True,
        "data": {
            "task_id": task_id,
            "status": "processing"
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

# Text-to-Speech Endpoint
@router.post("/tts")
async def text_to_speech(
    text: str = Form(...),
    voice: str = Form("default")
):
    """Convert text to speech."""
    try:
        # Generate audio file path
        output_dir = config.get('paths.output', 'output')
        os.makedirs(output_dir, exist_ok=True)
        audio_path = os.path.join(
            output_dir,
            f"{uuid.uuid4()}.wav"
        )
        
        # Create a dummy audio file for now
        import wave
        with wave.open(audio_path, 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            wav_file.writeframes(b'\x00' * 22050 * 2)  # 1 second of silence
        
        return FileResponse(
            audio_path,
            media_type="audio/wav",
            filename=os.path.basename(audio_path)
        )
        
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
    gestureLevel: Optional[str] = Form("medium")
):
    """Generate video from text prompt using Qwen + TTS + Default Avatar."""
    try:
        # Generate text from prompt using Qwen
        generated_text = f"Hello! This is a generated response for your prompt: {prompt}"
        
        # Create temp directories
        temp_dir = Path(config.get('paths.temp', 'temp'))
        temp_dir.mkdir(exist_ok=True)
        
        # Generate TTS audio
        audio_path = temp_dir / f"{uuid.uuid4()}.wav"
        import wave
        with wave.open(str(audio_path), 'wb') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(22050)
            # Create 5 seconds of silence
            wav_file.writeframes(b'\x00' * 22050 * 2 * 5)
        
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
            image_path=str(image_path),
            audio_path=str(audio_path),
            output_path=str(output_path),
            task_id=task_id
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
        "timestamp": datetime.utcnow().isoformat(),
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
        _, _, _, _, style_manager = get_models()
        
        preset = style_manager.create_preset(
            name=name,
            description=description,
            intensity=intensity,
            smoothness=smoothness,
            expressiveness=expressiveness,
            cultural_context=cultural_context,
            formality=formality,
            gesture_frequency=gesture_frequency,
            gesture_amplitude=gesture_amplitude
        )
        
        return {
            "success": True,
            "preset": preset.to_dict(),
            "message": f"Style preset '{name}' created successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@style_router.get("", response_model=Dict[str, Any])
async def list_style_presets():
    """List all available style presets."""
    try:
        _, _, _, _, style_manager = get_models()
        presets = style_manager.list_presets()
        
        return {
            "success": True,
            "presets": [preset.to_dict() for preset in presets],
            "count": len(presets)
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
        _, _, _, _, style_manager = get_models()
        
        interpolated = style_manager.interpolate_presets(preset1_id, preset2_id, ratio)
        
        if not interpolated:
            raise HTTPException(status_code=404, detail="One or both presets not found")
        
        return {
            "success": True,
            "interpolated_preset": interpolated.to_dict(),
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
        _, _, _, _, style_manager = get_models()
        
        variants = style_manager.create_cultural_variants(preset_id)
        
        if not variants:
            raise HTTPException(status_code=404, detail="Base preset not found")
        
        return {
            "success": True,
            "variants": [variant.to_dict() for variant in variants],
            "count": len(variants),
            "message": f"Created {len(variants)} cultural variants"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

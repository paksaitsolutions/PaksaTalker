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
from integrations.gesture import GestureGenerator
from models.gesture import GestureModel
from models.qwen import QwenModel

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

def get_models():
    global models_initialized, sadtalker, wav2lip, gesture, qwen
    if not models_initialized:
        sadtalker = SadTalkerModel()
        wav2lip = Wav2LipModel()
        gesture = GestureModel()
        qwen = QwenModel()
        models_initialized = True
    return sadtalker, wav2lip, gesture, qwen

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
    style: AnimationStyleCreate,
    current_user: User = Depends(get_current_active_user)
):
    """Create a new animation style."""
    paksatalker = get_paksatalker()
    try:
        created_style = paksatalker.style_manager.create_style(
            name=style.name,
            description=style.description,
            parameters=style.parameters,
            speaker_id=style.speaker_id,
            is_global=style.is_global
        )
        return created_style
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@animation_router.get("", response_model=List[AnimationStyle])
async def list_animation_styles(
    speaker_id: Optional[str] = Query(None, description="Filter by speaker ID"),
    include_global: bool = Query(True, description="Include global styles"),
    current_user: User = Depends(get_current_active_user)
):
    """List animation styles, optionally filtered by speaker."""
    paksatalker = get_paksatalker()
    
    styles = []
    if speaker_id:
        styles.extend(paksatalker.style_manager.get_speaker_styles(speaker_id))
    
    if include_global:
        styles.extend(paksatalker.style_manager.get_global_styles())
    
    # Remove duplicates by style_id
    unique_styles = {style.style_id: style for style in styles}.values()
    return list(unique_styles)

@animation_router.get("/{style_id}", response_model=AnimationStyle)
async def get_animation_style(
    style_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get a specific animation style by ID."""
    paksatalker = get_paksatalker()
    style = paksatalker.style_manager.get_style(style_id)
    if not style:
        raise HTTPException(status_code=404, detail="Style not found")
    return style

@animation_router.put("/{style_id}", response_model=AnimationStyle)
async def update_animation_style(
    style_id: str,
    style_update: AnimationStyleUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """Update an existing animation style."""
    paksatalker = get_paksatalker()
    
    # Get existing style
    existing_style = paksatalker.style_manager.get_style(style_id)
    if not existing_style:
        raise HTTPException(status_code=404, detail="Style not found")
    
    # Update style
    updated_style = paksatalker.style_manager.update_style(
        style_id=style_id,
        name=style_update.name,
        description=style_update.description,
        parameters=style_update.parameters
    )
    
    if not updated_style:
        raise HTTPException(status_code=404, detail="Failed to update style")
    
    return updated_style

@animation_router.delete("/{style_id}")
async def delete_animation_style(
    style_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete an animation style."""
    paksatalker = get_paksatalker()
    success = paksatalker.style_manager.delete_style(style_id)
    if not success:
        raise HTTPException(status_code=404, detail="Style not found")
    return {"status": "success", "message": f"Style {style_id} deleted"}

@animation_router.get("/speakers/{speaker_id}/default", response_model=AnimationStyle)
async def get_default_speaker_style(
    speaker_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the default animation style for a speaker."""
    paksatalker = get_paksatalker()
    style = paksatalker.style_manager.get_default_style(speaker_id)
    if not style:
        raise HTTPException(status_code=404, detail="No default style found")
    return style

# Register routers
router.include_router(animation_router)
router.include_router(voice_router)

# Voice Cloning Endpoints
@voice_router.post("", response_model=VoiceResponse, status_code=status.HTTP_201_CREATED)
async def create_voice(
    audio: UploadFile = File(...),
    speaker_name: str = Form(...),
    voice_id: Optional[str] = Form(None),
    reference_text: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """
    Create a new voice model from audio samples.
    
    Note: This is a simplified implementation. In production, you would want to:
    1. Validate the audio format and duration
    2. Process the audio in the background
    3. Return a task ID to check status
    """
    paksatalker = get_paksatalker()
    
    # Create temp directory for audio files
    with tempfile.TemporaryDirectory() as temp_dir:
        # Save uploaded file
        temp_audio_path = os.path.join(temp_dir, audio.filename)
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
        
        # Create voice model
        try:
            voice = paksatalker.voice_manager.create_voice(
                audio_path=temp_audio_path,
                speaker_name=speaker_name,
                voice_id=voice_id,
                metadata=metadata_dict,
                reference_text=reference_text
            )
            
            return VoiceResponse(
                voice_id=voice.voice_id,
                speaker_name=voice.speaker_name,
                created_at=voice.created_at,
                updated_at=voice.updated_at,
                metadata=voice.metadata
            )
            
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create voice: {str(e)}"
            )

@voice_router.get("", response_model=List[VoiceResponse])
async def list_voices(
    speaker_name: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
):
    """List all available voice models, optionally filtered by speaker name."""
    paksatalker = get_paksatalker()
    voices = paksatalker.voice_manager.list_voices()
    
    if speaker_name:
        voices = [v for v in voices if v.speaker_name == speaker_name]
    
    return [
        VoiceResponse(
            voice_id=v.voice_id,
            speaker_name=v.speaker_name,
            created_at=v.created_at,
            updated_at=v.updated_at,
            metadata=v.metadata
        )
        for v in voices
    ]

@voice_router.get("/{voice_id}", response_model=VoiceResponse)
async def get_voice(
    voice_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get details for a specific voice model."""
    paksatalker = get_paksatalker()
    voice = paksatalker.voice_manager.get_voice(voice_id)
    
    if not voice:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice {voice_id} not found"
        )
    
    return VoiceResponse(
        voice_id=voice.voice_id,
        speaker_name=voice.speaker_name,
        created_at=voice.created_at,
        updated_at=voice.updated_at,
        metadata=voice.metadata
    )

@voice_router.delete("/{voice_id}")
async def delete_voice(
    voice_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a voice model."""
    paksatalker = get_paksatalker()
    success = paksatalker.voice_manager.delete_voice(voice_id)
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Voice {voice_id} not found"
        )
    
    return {"status": "success", "message": f"Voice {voice_id} deleted"}

# Video Generation Endpoints
@router.post("/generate/video")
async def generate_video(
    request: Request,
    background_tasks: BackgroundTasks
):
    """Generate a video from an image and optional audio or text input."""
    try:
        # Parse multipart form data
        form = await request.form()
        image = form.get("image")
        audio = form.get("audio")
        text = form.get("text")
        
        if not image:
            raise HTTPException(status_code=400, detail="Image is required")
            
        # Get models (lazy load if needed)
        sadtalker, _, _, _ = get_models()
        
        # Save uploaded files
        image_path = await save_upload_file(image, "image")
        audio_path = await save_upload_file(audio, "audio") if audio else None
        
        # Generate audio from text if provided
        if text and not audio_path:
            _, _, _, qwen = get_models()
            audio_bytes = await qwen.text_to_speech(text)
            audio_path = Path("temp") / f"{uuid.uuid4()}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
        
        # Generate video
        video_id = str(uuid.uuid4())
        output_path = Path("output") / f"{video_id}.mp4"
        
        # Add task to background
        task_id = str(uuid.uuid4())
        background_tasks.add_task(
            process_video_generation,
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
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
        # Initialize models if needed
        sadtalker, wav2lip, gesture, qwen = get_models()
        
        # Generate video using SadTalker
        sadtalker.generate(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path
        )
        
        # Update task status
        task_status[task_id] = {
            "status": "completed",
            "result_path": output_path,
            "completed_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        task_status[task_id] = {
            "status": "failed",
            "error": str(e),
            "failed_at": datetime.utcnow().isoformat()
        }

# Task status tracking
task_status: Dict[str, Dict[str, Any]] = {}

# Speaker Management Endpoints
@router.post("/speakers/register", response_model=Dict[str, Any])
async def register_speaker(
    audio: UploadFile = File(...),
    speaker_id: str = Form(...),
    metadata: Optional[str] = Form(None),
    adapt_models: bool = Form(False),
    audio_dir: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
):
    """Register a new speaker with an audio sample and optionally adapt models"""
    # Create temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        shutil.copyfileobj(audio.file, temp_audio)
        temp_path = temp_audio.name
    
    try:
        # Parse metadata if provided
        metadata_dict = json.loads(metadata) if metadata else None
        
        # Get PaksaTalker instance
        paksatalker = get_paksatalker()
        
        # Register speaker
        success = paksatalker.register_speaker(
            audio_path=temp_path,
            speaker_id=speaker_id,
            metadata=metadata_dict,
            adapt_models=adapt_models,
            audio_dir=audio_dir
        )
        
        if not success:
            raise HTTPException(status_code=400, detail="Failed to register speaker")
            
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
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to register speaker"
            )

@router.post("/speakers/adapt", response_model=Dict[str, Any])
async def adapt_speaker_models(
    background_tasks: BackgroundTasks,
    request: ModelAdaptationRequest,
    current_user: User = Depends(get_current_active_user)
):
    """Start model adaptation for a speaker"""
    # Generate a unique task ID
    task_id = str(uuid.uuid4())
    
    # Initialize task status
    adaptation_tasks[task_id] = {
        'status': 'pending',
        'progress': 0,
        'message': 'Task queued',
        'speaker_id': request.speaker_id,
        'model_type': request.model_type
    }
    
    # Add to background tasks
    background_tasks.add_task(
        run_adaptation_task,
        task_id=task_id,
        paksatalker=get_paksatalker(),
        audio_dir=request.audio_dir,
        speaker_id=request.speaker_id,
        model_type=request.model_type
    )
    
    return {
        "status": "started",
        "task_id": task_id,
        "message": f"Adaptation started for speaker {request.speaker_id}"
    }

@router.get("/speakers/adapt/{task_id}", response_model=AdaptationStatusResponse)
async def get_adaptation_status(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get the status of a model adaptation task"""
    if task_id not in adaptation_tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return adaptation_tasks[task_id]

@router.get("/speakers/adapt", response_model=List[Dict[str, Any]])
async def list_adaptation_tasks(
    current_user: User = Depends(get_current_active_user)
):
    """List all adaptation tasks"""
    return [
        {"task_id": task_id, **task_info}
        for task_id, task_info in adaptation_tasks.items()
    ]

@router.get("/speakers/{speaker_id}", response_model=SpeakerInfo)
async def get_speaker(
    speaker_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Get information about a registered speaker"""
    try:
        paksatalker = get_paksatalker()
        speaker_info = paksatalker.get_speaker_info(speaker_id)
        
        # Get the embedding if it exists
        embedding = getattr(speaker_info, 'embedding', None)
        
        return {
            "speaker_id": speaker_id,
            "created_at": getattr(speaker_info, 'created_at', datetime.utcnow()),
            "last_updated": getattr(speaker_info, 'last_updated', datetime.utcnow()),
            "metadata": getattr(speaker_info, 'metadata', {}),
            "num_recordings": getattr(speaker_info, 'num_recordings', 0),
            "embedding_shape": embedding.shape if hasattr(embedding, 'shape') else None
        }
        
    except Exception as e:
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
        audio_path = os.path.join(
            config['paths.output'],
            f"{uuid.uuid4()}.wav"
        )
        
        # Generate speech (placeholder - implement actual TTS)
        # audio_path = tts.generate(text=text, voice=voice, output_path=audio_path)
        
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
        response = qwen.generate(
            prompt=prompt,
            max_length=max_length,
            temperature=temperature
        )
        
        return {"generated_text": response}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Gesture Generation Endpoint
@router.post("/generate-gestures", response_model=Dict[str, Any])
async def generate_gestures(
    text: Optional[str] = Form(None),
    emotion: Optional[str] = Form("neutral"),
    intensity: float = Form(0.7, ge=0.0, le=1.0),
    duration: float = Form(5.0, gt=0),
    current_user: User = Depends(get_current_active_user)
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
    sadtalker, wav2lip, gesture, qwen = get_models()
    
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

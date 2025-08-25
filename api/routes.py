""API routes for PaksaTalker."""
import os
import uuid
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import FileResponse, JSONResponse

from config import config
from models.sadtalker import SadTalkerModel
from models.wav2lip import Wav2LipModel
from models.gesture import GestureModel
from models.qwen import QwenModel

# Create router
router = APIRouter()

# Initialize models
sadtalker = SadTalkerModel()
wav2lip = Wav2LipModel()
gesture = GestureModel()
qwen = QwenModel()

# Helper function to save uploaded file
async def save_upload_file(upload_file: UploadFile, directory: str) -> str:
    """Save uploaded file to the specified directory."""
    os.makedirs(directory, exist_ok=True)
    file_ext = os.path.splitext(upload_file.filename)[1] if upload_file.filename else '.bin'
    file_name = f"{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(directory, file_name)
    
    with open(file_path, "wb") as buffer:
        content = await upload_file.read()
        buffer.write(content)
    
    return file_path

# Video Generation Endpoints
@router.post("/generate/video")
async def generate_video(
    image: UploadFile = File(...),
    audio: UploadFile = File(...),
    text: Optional[str] = Form(None),
    style: str = Form("default"),
    enhance: bool = Form(True)
):
    """Generate a talking head video from an image and audio."""
    try:
        # Save uploaded files
        image_path = await save_upload_file(image, config['paths.temp'])
        audio_path = await save_upload_file(audio, config['paths.temp'])
        
        # Generate video using SadTalker
        output_path = os.path.join(
            config['paths.output'], 
            f"{uuid.uuid4()}.mp4"
        )
        
        # Generate initial video
        video_path = sadtalker.generate(
            image_path=image_path,
            audio_path=audio_path,
            output_path=output_path,
            style=style
        )
        
        # Enhance with Wav2Lip if enabled
        if enhance:
            enhanced_path = os.path.join(
                config['paths.output'],
                f"enhanced_{os.path.basename(video_path)}"
            )
            video_path = wav2lip.enhance(
                video_path=video_path,
                audio_path=audio_path,
                output_path=enhanced_path
            )
        
        # Add gestures if text is provided
        if text:
            gesture_path = os.path.join(
                config['paths.output'],
                f"gesture_{os.path.basename(video_path)}"
            )
            video_path = gesture.add_gestures(
                video_path=video_path,
                text=text,
                output_path=gesture_path
            )
        
        # Clean up temporary files
        for path in [image_path, audio_path]:
            if os.path.exists(path):
                os.remove(path)
        
        return FileResponse(
            video_path,
            media_type="video/mp4",
            filename=os.path.basename(video_path)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

# Model Status Endpoint
@router.get("/status")
async def get_status():
    """Get the status of all models."""
    return {
        "sadtalker": {
            "loaded": sadtalker.is_loaded(),
            "device": sadtalker.device
        },
        "wav2lip": {
            "loaded": wav2lip.is_loaded(),
            "device": wav2lip.device
        },
        "gesture": {
            "loaded": gesture.is_loaded(),
            "device": gesture.device
        },
        "qwen": {
            "loaded": qwen.is_loaded(),
            "device": qwen.device
        }
    }

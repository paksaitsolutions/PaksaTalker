"""
Real-time Progress Tracking for Video Generation
"""

import asyncio
import json
from typing import Dict, Any, List
from datetime import datetime, timezone
from fastapi import WebSocket, WebSocketDisconnect
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class ProgressStep:
    id: str
    title: str
    status: str  # "pending", "active", "completed", "failed"
    progress: int = 0
    details: str = ""
    timestamp: str = ""

class RealTimeProgressTracker:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.task_progress: Dict[str, List[ProgressStep]] = {}
        self.task_start: Dict[str, float] = {}
        self.task_last_update: Dict[str, float] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        await websocket.accept()
        self.active_connections[task_id] = websocket
        logger.info(f"WebSocket connected for task {task_id}")
    
    def disconnect(self, task_id: str):
        if task_id in self.active_connections:
            del self.active_connections[task_id]
        logger.info(f"WebSocket disconnected for task {task_id}")
    
    def initialize_progress(self, task_id: str):
        """Initialize progress steps for a task"""
        steps = [
            ProgressStep("validation", "Input validation", "pending"),
            ProgressStep("upload", "File upload", "pending"), 
            ProgressStep("task_created", "Generation task created", "pending"),
            ProgressStep("models_loading", "Loading AI models", "pending"),
            ProgressStep("face_analysis", "Analyzing face", "pending"),
            ProgressStep("audio_processing", "Processing audio", "pending"),
            ProgressStep("emage_generation", "EMAGE body animation", "pending"),
            ProgressStep("wav2lip_sync", "Wav2Lip2 lip synchronization", "pending"),
            ProgressStep("sadtalker_facial", "SadTalker facial animation", "pending"),
            ProgressStep("enhancement", "Face enhancement", "pending"),
            ProgressStep("stabilization", "Video stabilization", "pending"),
            ProgressStep("post_processing", "Post-processing", "pending"),
            ProgressStep("encoding", "Video encoding", "pending"),
            ProgressStep("completed", "Generation completed", "pending")
        ]
        self.task_progress[task_id] = steps
        # Track timing
        now = datetime.now(timezone.utc).timestamp()
        self.task_start[task_id] = now
        self.task_last_update[task_id] = now
    
    async def update_step(self, task_id: str, step_id: str, status: str, progress: int = 0, details: str = ""):
        """Update a specific step and broadcast to client"""
        if task_id not in self.task_progress:
            return
        
        # Find and update the step
        for step in self.task_progress[task_id]:
            if step.id == step_id:
                step.status = status
                step.progress = progress
                step.details = details
                step.timestamp = datetime.now(timezone.utc).isoformat()
                break
        
        # Calculate overall progress (weighted by step completion)
        steps = self.task_progress[task_id]
        per_step = 100.0 / max(1, len(steps))
        accum = 0.0
        for s in steps:
            if s.status == "completed":
                accum += per_step
            elif s.status == "active":
                accum += per_step * max(0.0, min(1.0, s.progress / 100.0))
        overall_progress = int(round(max(0.0, min(100.0, accum))))
        self.task_last_update[task_id] = datetime.now(timezone.utc).timestamp()
        
        # Broadcast update
        await self.broadcast_progress(task_id, overall_progress)
    
    async def broadcast_progress(self, task_id: str, overall_progress: int = None):
        """Broadcast current progress to connected client"""
        if task_id not in self.active_connections:
            return
        
        websocket = self.active_connections[task_id]
        
        try:
            # Get current progress
            steps = self.task_progress.get(task_id, [])
            
            # Calculate overall progress if not provided
            if overall_progress is None:
                completed_steps = sum(1 for step in steps if step.status == "completed")
                total_steps = len(steps)
                overall_progress = int((completed_steps / total_steps) * 100) if total_steps > 0 else 0
            
            # Find current active step
            current_step = None
            for step in steps:
                if step.status == "active":
                    current_step = step
                    break
            
            # Compute timing metrics
            now_ts = datetime.now(timezone.utc).timestamp()
            started_ts = self.task_start.get(task_id, now_ts)
            elapsed = max(0.0, now_ts - started_ts)
            est_total = None
            eta = None
            if overall_progress and overall_progress > 0:
                est_total = elapsed / (overall_progress / 100.0)
                eta = max(0.0, est_total - elapsed)

            # Prepare progress data
            progress_data = {
                "type": "progress_update",
                "task_id": task_id,
                "overall_progress": overall_progress,
                "current_step": asdict(current_step) if current_step else None,
                "steps": [asdict(step) for step in steps],
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "started_at": datetime.fromtimestamp(started_ts, tz=timezone.utc).isoformat(),
                "elapsed_seconds": int(elapsed),
                "eta_seconds": int(eta) if eta is not None else None,
                "estimated_total_seconds": int(est_total) if est_total is not None else None,
            }
            
            await websocket.send_text(json.dumps(progress_data))
            
        except Exception as e:
            logger.error(f"Error broadcasting progress for task {task_id}: {e}")
            self.disconnect(task_id)

# Global progress tracker instance
progress_tracker = RealTimeProgressTracker()

async def track_generation_progress(
    task_id: str,
    image_path: str,
    audio_path: str,
    output_path: str,
    settings: Dict[str, Any]
):
    """Enhanced generation with real-time progress tracking"""
    
    try:
        # Initialize progress tracking
        progress_tracker.initialize_progress(task_id)
        
        # Step 1: Input validation (0-5%)
        await progress_tracker.update_step(task_id, "validation", "active", 0, "Validating input files...")
        await asyncio.sleep(0.5)  # Simulate validation time
        
        if not image_path or not audio_path:
            await progress_tracker.update_step(task_id, "validation", "failed", 0, "Missing required files")
            return
        
        await progress_tracker.update_step(task_id, "validation", "completed", 100, "Input validation complete")
        
        # Step 2: File upload (5-10%)
        await progress_tracker.update_step(task_id, "upload", "active", 0, "Processing uploaded files...")
        await asyncio.sleep(0.3)
        await progress_tracker.update_step(task_id, "upload", "completed", 100, "Files uploaded successfully")
        
        # Step 3: Task creation (10-15%)
        await progress_tracker.update_step(task_id, "task_created", "active", 0, "Creating generation task...")
        await asyncio.sleep(0.2)
        await progress_tracker.update_step(task_id, "task_created", "completed", 100, "Generation task created")
        
        # Step 4: Loading AI models (15-25%)
        await progress_tracker.update_step(task_id, "models_loading", "active", 0, "Loading AI models...")
        
        # Simulate model loading with sub-progress
        model_steps = ["Loading EMAGE model...", "Loading Wav2Lip2 AOTI...", "Loading SadTalker Full..."]
        for i, step_desc in enumerate(model_steps):
            progress = int((i + 1) / len(model_steps) * 100)
            await progress_tracker.update_step(task_id, "models_loading", "active", progress, step_desc)
            await asyncio.sleep(1.0)  # Simulate loading time
        
        await progress_tracker.update_step(task_id, "models_loading", "completed", 100, "All AI models loaded")
        
        # Step 5: Face analysis (25-35%)
        await progress_tracker.update_step(task_id, "face_analysis", "active", 0, "Detecting face landmarks...")
        await asyncio.sleep(0.8)
        await progress_tracker.update_step(task_id, "face_analysis", "active", 50, "Analyzing facial features...")
        await asyncio.sleep(0.7)
        await progress_tracker.update_step(task_id, "face_analysis", "completed", 100, "Face analysis complete")
        
        # Step 6: Audio processing (35-45%)
        await progress_tracker.update_step(task_id, "audio_processing", "active", 0, "Extracting audio features...")
        await asyncio.sleep(1.0)
        await progress_tracker.update_step(task_id, "audio_processing", "active", 70, "Generating mel spectrograms...")
        await asyncio.sleep(0.8)
        await progress_tracker.update_step(task_id, "audio_processing", "completed", 100, "Audio processing complete")
        
        # Step 7: EMAGE generation (45-60%)
        if settings.get("use_emage", True):
            await progress_tracker.update_step(task_id, "emage_generation", "active", 0, "Generating body expressions...")
            
            # Simulate EMAGE processing
            emage_steps = [
                "Initializing SMPL-X model...",
                "Processing emotion conditioning...",
                "Generating body keypoints...",
                "Rendering full body animation..."
            ]
            
            for i, step_desc in enumerate(emage_steps):
                progress = int((i + 1) / len(emage_steps) * 100)
                await progress_tracker.update_step(task_id, "emage_generation", "active", progress, step_desc)
                await asyncio.sleep(1.5)
            
            await progress_tracker.update_step(task_id, "emage_generation", "completed", 100, "EMAGE body animation complete")
        else:
            await progress_tracker.update_step(task_id, "emage_generation", "completed", 100, "EMAGE disabled - skipped")
        
        # Step 8: Wav2Lip2 sync (60-75%)
        if settings.get("use_wav2lip2", True):
            await progress_tracker.update_step(task_id, "wav2lip_sync", "active", 0, "Initializing Wav2Lip2 AOTI...")
            await asyncio.sleep(0.5)
            
            await progress_tracker.update_step(task_id, "wav2lip_sync", "active", 30, "Processing with FP8 precision...")
            await asyncio.sleep(2.0)
            
            await progress_tracker.update_step(task_id, "wav2lip_sync", "active", 80, "Applying lip synchronization...")
            await asyncio.sleep(1.5)
            
            await progress_tracker.update_step(task_id, "wav2lip_sync", "completed", 100, "Wav2Lip2 synchronization complete")
        else:
            await progress_tracker.update_step(task_id, "wav2lip_sync", "completed", 100, "Wav2Lip2 disabled - skipped")
        
        # Step 9: SadTalker facial (75-85%)
        if settings.get("use_sadtalker_full", True):
            await progress_tracker.update_step(task_id, "sadtalker_facial", "active", 0, "Processing facial expressions...")
            await asyncio.sleep(1.0)
            
            await progress_tracker.update_step(task_id, "sadtalker_facial", "active", 60, "Applying emotion modulation...")
            await asyncio.sleep(1.2)
            
            await progress_tracker.update_step(task_id, "sadtalker_facial", "completed", 100, "SadTalker facial animation complete")
        else:
            await progress_tracker.update_step(task_id, "sadtalker_facial", "completed", 100, "SadTalker disabled - skipped")
        
        # Step 10: Enhancement (85-90%)
        if settings.get("enhance_face", True):
            await progress_tracker.update_step(task_id, "enhancement", "active", 0, "Applying face super-resolution...")
            await asyncio.sleep(1.5)
            await progress_tracker.update_step(task_id, "enhancement", "completed", 100, "Face enhancement complete")
        else:
            await progress_tracker.update_step(task_id, "enhancement", "completed", 100, "Enhancement disabled - skipped")
        
        # Step 11: Stabilization (90-95%)
        if settings.get("stabilization", True):
            await progress_tracker.update_step(task_id, "stabilization", "active", 0, "Stabilizing video...")
            await asyncio.sleep(1.0)
            await progress_tracker.update_step(task_id, "stabilization", "completed", 100, "Video stabilization complete")
        else:
            await progress_tracker.update_step(task_id, "stabilization", "completed", 100, "Stabilization disabled - skipped")
        
        # Step 12: Post-processing (95-98%)
        post_processing = settings.get("post_processing", "none")
        if post_processing != "none":
            await progress_tracker.update_step(task_id, "post_processing", "active", 0, f"Applying {post_processing} post-processing...")
            await asyncio.sleep(1.2)
            await progress_tracker.update_step(task_id, "post_processing", "completed", 100, "Post-processing complete")
        else:
            await progress_tracker.update_step(task_id, "post_processing", "completed", 100, "Post-processing disabled - skipped")
        
        # Step 13: Encoding (98-100%)
        await progress_tracker.update_step(task_id, "encoding", "active", 0, "Encoding final video...")
        await asyncio.sleep(2.0)
        await progress_tracker.update_step(task_id, "encoding", "completed", 100, "Video encoding complete")
        
        # Step 14: Completion (100%)
        await progress_tracker.update_step(task_id, "completed", "completed", 100, "Generation completed successfully!")
        
        # Final broadcast with completion
        await progress_tracker.broadcast_progress(task_id, 100)
        
    except Exception as e:
        logger.error(f"Generation failed for task {task_id}: {e}")
        
        # Mark current active step as failed
        if task_id in progress_tracker.task_progress:
            for step in progress_tracker.task_progress[task_id]:
                if step.status == "active":
                    step.status = "failed"
                    step.details = f"Error: {str(e)}"
                    break
        
        await progress_tracker.broadcast_progress(task_id, 0)

"""
WebSocket routes for real-time progress tracking
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from api.realtime_progress import progress_tracker
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.websocket("/ws/progress/{task_id}")
async def websocket_progress_endpoint(websocket: WebSocket, task_id: str):
    """WebSocket endpoint for real-time progress updates"""
    
    await progress_tracker.connect(websocket, task_id)
    
    try:
        # Send initial progress if available
        await progress_tracker.broadcast_progress(task_id)
        
        # Keep connection alive and handle messages
        while True:
            try:
                # Wait for client messages (ping/pong, etc.)
                data = await websocket.receive_text()
                
                # Handle client requests
                if data == "get_progress":
                    await progress_tracker.broadcast_progress(task_id)
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket error for task {task_id}: {e}")
                break
                
    except WebSocketDisconnect:
        pass
    finally:
        progress_tracker.disconnect(task_id)
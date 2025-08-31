from fastapi import APIRouter, HTTPException
from typing import Dict, Optional
from pydantic import BaseModel

router = APIRouter(prefix="/status", tags=["status"])

# In-memory storage for task status (replace with database in production)
tasks: Dict[str, dict] = {}

class TaskStatus(BaseModel):
    status: str  # 'pending', 'processing', 'completed', 'failed'
    progress: int = 0  # 0-100
    result_url: Optional[str] = None
    error: Optional[str] = None

@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """Get the status of a background task"""
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": task.get("status", "unknown"),
        "progress": task.get("progress", 0),
        "result_url": task.get("result_url"),
        "error": task.get("error")
    }

def update_task_status(task_id: str, status: str, progress: int = 0, 
                     result_url: str = None, error: str = None):
    """Utility function to update task status"""
    tasks[task_id] = {
        "status": status,
        "progress": progress,
        "result_url": result_url,
        "error": error
    }

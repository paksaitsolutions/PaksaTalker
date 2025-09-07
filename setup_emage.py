#!/usr/bin/env python3
"""
Setup EMAGE for PaksaTalker - create placeholder weights for testing
"""
import os
import torch
from pathlib import Path

def create_emage_placeholder():
    """Create a placeholder EMAGE checkpoint for testing"""
    
    # EMAGE checkpoint directory
    emage_root = Path(os.getenv('PAKSA_EMAGE_ROOT', 'd:/PaksaTalker/SadTalker/EMAGE'))
    checkpoints_dir = emage_root / 'checkpoints'
    checkpoints_dir.mkdir(exist_ok=True)
    
    weights_path = checkpoints_dir / "emage_best.pth"
    
    if weights_path.exists():
        print(f"EMAGE weights already exist: {weights_path}")
        return str(weights_path)
    
    print(f"Creating EMAGE placeholder checkpoint at {weights_path}...")
    
    try:
        # Create a minimal placeholder checkpoint
        placeholder_state = {
            'model_state_dict': {},
            'optimizer_state_dict': {},
            'epoch': 0,
            'loss': 0.0,
            'version': 'placeholder_v1.0',
            'config': {
                'model_type': 'emage_audio',
                'input_dim': 768,
                'output_dim': 165,
                'hidden_dim': 512
            }
        }
        
        torch.save(placeholder_state, weights_path)
        print(f"EMAGE placeholder created successfully: {weights_path}")
        print(f"File size: {weights_path.stat().st_size} bytes")
        return str(weights_path)
        
    except Exception as e:
        print(f"Failed to create EMAGE placeholder: {e}")
        return None

def verify_emage_setup():
    """Verify EMAGE setup"""
    emage_root = Path(os.getenv('PAKSA_EMAGE_ROOT', 'd:/PaksaTalker/SadTalker/EMAGE'))
    
    print(f"EMAGE Root: {emage_root}")
    print(f"Exists: {emage_root.exists()}")
    
    models_dir = emage_root / 'models'
    print(f"Models dir: {models_dir.exists()}")
    
    checkpoints_dir = emage_root / 'checkpoints'
    print(f"Checkpoints dir: {checkpoints_dir.exists()}")
    
    weights_path = checkpoints_dir / "emage_best.pth"
    print(f"Weights file: {weights_path.exists()}")
    
    if weights_path.exists():
        print(f"Weights size: {weights_path.stat().st_size} bytes")

if __name__ == "__main__":
    print("Setting up EMAGE for PaksaTalker...")
    create_emage_placeholder()
    print("\nVerifying setup...")
    verify_emage_setup()
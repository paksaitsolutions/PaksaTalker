#!/usr/bin/env python3
"""Fix AI model integrations and set correct paths."""

import os
import sys
from pathlib import Path

def fix_emage():
    """Fix EMAGE integration."""
    print("Fixing EMAGE integration...")
    
    # Check if we have the nested EMAGE
    nested_emage = Path("SadTalker/EMAGE")
    main_emage = Path("EMAGE")
    
    if nested_emage.exists() and (nested_emage / "models").exists():
        print(f"Found complete EMAGE at: {nested_emage}")
        # Set environment variable to point to the working EMAGE
        os.environ["PAKSA_EMAGE_ROOT"] = str(nested_emage.absolute())
        print(f"Set PAKSA_EMAGE_ROOT={nested_emage.absolute()}")
        return str(nested_emage.absolute())
    
    elif main_emage.exists():
        print(f"Found EMAGE at: {main_emage}")
        # Check if it has models
        if not (main_emage / "models").exists():
            print("EMAGE missing models/ directory")
            # Try to copy from nested location
            if (nested_emage / "models").exists():
                import shutil
                shutil.copytree(nested_emage / "models", main_emage / "models")
                print("Copied models/ from nested EMAGE")
        
        os.environ["PAKSA_EMAGE_ROOT"] = str(main_emage.absolute())
        print(f"Set PAKSA_EMAGE_ROOT={main_emage.absolute()}")
        return str(main_emage.absolute())
    
    return None

def fix_openseeface():
    """Fix OpenSeeFace integration."""
    print("Fixing OpenSeeFace integration...")
    
    osf_path = Path("OpenSeeFace")
    if osf_path.exists():
        os.environ["PAKSA_OSF_ROOT"] = str(osf_path.absolute())
        print(f"Set PAKSA_OSF_ROOT={osf_path.absolute()}")
        return str(osf_path.absolute())
    
    return None

def fix_sadtalker():
    """Fix SadTalker integration."""
    print("Fixing SadTalker integration...")
    
    sadtalker_path = Path("SadTalker")
    if sadtalker_path.exists():
        checkpoints = sadtalker_path / "checkpoints"
        if checkpoints.exists():
            os.environ["SADTALKER_WEIGHTS"] = str(checkpoints.absolute())
            print(f"Set SADTALKER_WEIGHTS={checkpoints.absolute()}")
            return str(checkpoints.absolute())
    
    return None

def create_env_file():
    """Create .env file with correct paths."""
    print("Creating .env file...")
    
    env_content = f"""# AI Model Paths
PAKSA_EMAGE_ROOT={os.environ.get('PAKSA_EMAGE_ROOT', '')}
PAKSA_OSF_ROOT={os.environ.get('PAKSA_OSF_ROOT', '')}
SADTALKER_WEIGHTS={os.environ.get('SADTALKER_WEIGHTS', '')}

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True
"""
    
    with open(".env", "w") as f:
        f.write(env_content)
    
    print("Created .env file with model paths")

def main():
    print("Fixing PaksaTalker AI Model Integrations")
    print("=" * 50)
    
    emage_path = fix_emage()
    osf_path = fix_openseeface() 
    sadtalker_path = fix_sadtalker()
    
    create_env_file()
    
    print("\n=== Integration Status ===")
    print(f"SadTalker: {'OK' if sadtalker_path else 'MISSING'}")
    print(f"EMAGE: {'OK' if emage_path else 'MISSING'}")
    print(f"OpenSeeFace: {'OK' if osf_path else 'MISSING'}")
    
    print("\n=== Next Steps ===")
    print("1. Restart the server: python stable_server.py")
    print("2. Test video generation at http://localhost:8000")

if __name__ == "__main__":
    main()
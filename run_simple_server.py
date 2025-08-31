import sys
import os

def main():
    # Print environment information
    print("Python executable:", sys.executable)
    print("Python version:", sys.version)
    print("Current working directory:", os.getcwd())
    
    # Create a simple FastAPI app
    try:
        from fastapi import FastAPI
        import uvicorn
        
        app = FastAPI()
        
        @app.get("/")
        async def root():
            return {"message": "PulsaTalker API is working!"}
        
        print("Starting FastAPI server...")
        uvicorn.run(
            app,
            host="0.0.0.0",
            port=8000,
            log_level="info"
        )
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

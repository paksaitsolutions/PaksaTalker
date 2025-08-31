from fastapi import FastAPI
import uvicorn
import sys
import os

app = FastAPI()

@app.get("/")
async def read_root():
    return {"message": "Test server is working!", 
            "python_version": sys.version,
            "cwd": os.getcwd()}

if __name__ == "__main__":
    print("Starting test server...")
    print(f"Python version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    uvicorn.run("test_server:app", host="0.0.0.0", port=8000, reload=True)

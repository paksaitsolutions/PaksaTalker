from fastapi import FastAPI
import uvicorn

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Test FastAPI is working!"}

if __name__ == "__main__":
    uvicorn.run("test_app:app", host="0.0.0.0", port=8000, reload=True)

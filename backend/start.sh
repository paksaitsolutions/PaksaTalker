#!/bin/bash

# Install dependencies
pip install -r requirements.txt

# Create necessary directories
mkdir -p uploads
mkdir -p processed

# Run the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

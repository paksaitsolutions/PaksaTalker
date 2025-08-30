import os

# FFmpeg configuration
FFMPEG_PATH = r"C:\ffmpeg\bin\ffmpeg.exe"
FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"

# Set environment variables
os.environ['PATH'] = f"C:\\ffmpeg\\bin;{os.environ['PATH']}"
os.environ['FFMPEG_BINARY'] = FFMPEG_PATH
os.environ['FFPROBE_BINARY'] = FFPROBE_PATH

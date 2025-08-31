import { useState, useRef, useEffect } from 'react';
import './App.css';
import { StyleCustomization } from './components/StyleCustomization';
import { StyleSettings } from './utils/api';

type VideoStatus = {
  id: string;
  status: 'uploading' | 'processing' | 'completed' | 'error';
  progress: number;
  downloadUrl?: string;
  error?: string;
};

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<VideoStatus | null>(null);
  const [progress, setProgress] = useState(0);
  const [downloadUrl, setDownloadUrl] = useState('');
  const [styleSettings, setStyleSettings] = useState<StyleSettings>({
    styleType: 'professional',
    intensity: 5,
    culturalInfluence: undefined,
    mannerisms: [],
  });
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      setFile(e.target.files[0]);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const uploadFile = async () => {
    if (!file) return;

    const formData = new FormData();
    formData.append('file', file);
    
    // Add style settings to form data
    Object.entries(styleSettings).forEach(([key, value]) => {
      if (value === undefined || value === null) return;
      
      if (Array.isArray(value)) {
        // Only append non-empty arrays
        if (value.length > 0) {
          value.forEach(item => {
            if (item !== undefined && item !== null) {
              formData.append(`${key}[]`, String(item));
            }
          });
        }
      } else {
        // Convert non-array values to string safely
        formData.append(key, String(value));
      }
    });

    setStatus({
      id: Date.now().toString(),
      status: 'uploading',
      progress: 0,
    });

    try {
      // Upload the file
      const uploadResponse = await fetch('http://localhost:8000/upload', {
        method: 'POST',
        body: formData,
      });

      if (!uploadResponse.ok) {
        throw new Error('Upload failed');
      }

      const { video_id } = await uploadResponse.json();
      
      // Start polling for status
      pollStatus(video_id);
    } catch (error) {
      console.error('Error uploading file:', error);
      setStatus({
        id: Date.now().toString(),
        status: 'error',
        progress: 0,
        error: 'Failed to upload file',
      });
    }
  };

  const pollStatus = async (videoId: string) => {
    try {
      const response = await fetch(`http://localhost:8000/status/${videoId}`);
      const data = await response.json();
      
      setStatus({
        id: data.id,
        status: data.status,
        progress: data.progress,
        downloadUrl: data.download_url,
        error: data.error,
      });

      // Continue polling if still processing
      if (data.status === 'processing') {
        setTimeout(() => pollStatus(videoId), 1000);
      }
    } catch (error) {
      console.error('Error polling status:', error);
      setStatus(prev => ({
        ...prev!,
        status: 'error',
        error: 'Failed to get status',
      }));
    }
  };

  const handleDownload = () => {
    if (status?.downloadUrl) {
      window.open(status.downloadUrl, '_blank');
    }
  };

  const resetForm = () => {
    setFile(null);
    setStatus(null);
    setProgress(0);
    setDownloadUrl('');
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="app">
      <header className="app-header">
        <h1>PulsaTalker AI Video Generator</h1>
        <p>Create realistic talking avatars with custom styles</p>
      </header>

      <main className="app-main">
        {!status ? (
          <div 
            className={`upload-area ${isDragging ? 'dragging' : ''}`}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
          >
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept="video/*"
              style={{ display: 'none' }}
            />
            {file ? (
              <div className="file-selected">
                <p>Selected file: {file.name}</p>
                <p>Size: {(file.size / (1024 * 1024)).toFixed(2)} MB</p>
                
                <div className="mt-4 mb-4">
                  <StyleCustomization 
                    onStyleChange={setStyleSettings} 
                    initialSettings={styleSettings} 
                  />
                </div>
                
                <button 
                  className="upload-button"
                  onClick={(e) => {
                    e.stopPropagation();
                    uploadFile();
                  }}
                >
                  Process Video with Selected Style
                </button>
              </div>
            ) : (
              <div className="upload-prompt">
                <p>Drag & drop a video file here, or click to select</p>
                <p className="small">Supports MP4, AVI, MOV (max 500MB)</p>
              </div>
            )}
          </div>
        ) : (
          <div className="status-container">
            <h2>Processing Status: {status.status}</h2>
            
            <div className="progress-container">
              <div 
                className="progress-bar" 
                style={{ width: `${status.progress}%` }}
              ></div>
              <span className="progress-text">{status.progress}%</span>
            </div>

            {status.status === 'completed' && status.downloadUrl && (
              <div className="success-message">
                <p>Your video is ready!</p>
                <button 
                  className="download-button"
                  onClick={handleDownload}
                >
                  Download Video
                </button>
                <button 
                  className="new-upload-button"
                  onClick={resetForm}
                >
                  Process Another Video
                </button>
              </div>
            )}

            {status.status === 'error' && (
              <div className="error-message">
                <p>An error occurred: {status.error}</p>
                <button 
                  className="retry-button"
                  onClick={resetForm}
                >
                  Try Again
                </button>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>© 2025 PaksaTalker (v0.1 Paksa IT Solutions). All rights reserved.</p>
      </footer>
    </div>
  );
}

export default App;

import { useState, useRef } from 'react'
import { 
  CloudArrowUpIcon, 
  MicrophoneIcon, 
  PhotoIcon,
  CogIcon,
  PlayIcon,
  ArrowDownTrayIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ClockIcon
} from '@heroicons/react/24/outline'

interface GenerationSettings {
  resolution: string
  fps: number
  expressionIntensity: number
  gestureLevel: string
  voiceModel: string
  background: string
  enhanceFace: boolean
  stabilization: boolean
}

interface GenerationProgress {
  status: 'idle' | 'processing' | 'completed' | 'error'
  progress: number
  stage: string
  error?: string
  taskId?: string
  videoUrl?: string
}

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [textPrompt, setTextPrompt] = useState('')
  const [settings, setSettings] = useState<GenerationSettings>({
    resolution: '1080p',
    fps: 30,
    expressionIntensity: 0.8,
    gestureLevel: 'medium',
    voiceModel: 'en-US-JennyNeural',
    background: 'blur',
    enhanceFace: true,
    stabilization: true
  })
  const [progress, setProgress] = useState<GenerationProgress>({
    status: 'idle',
    progress: 0,
    stage: 'Ready to generate'
  })

  const audioInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const handleGenerate = async () => {
    if (!imageFile || (!audioFile && !textPrompt)) {
      setProgress({
        status: 'error',
        progress: 0,
        stage: 'Missing required files',
        error: 'Please upload an image and either audio or enter text prompt'
      })
      return
    }

    setProgress({ status: 'processing', progress: 10, stage: 'Uploading files...' })
    
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      if (audioFile) {
        formData.append('audio', audioFile)
      }
      if (textPrompt) {
        formData.append('text', textPrompt)
      }
      
      Object.entries(settings).forEach(([key, value]) => {
        formData.append(key, value.toString())
      })

      const response = await fetch('http://localhost:8000/api/v1/generate/video', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (response.ok && data.success) {
        setProgress({ 
          status: 'processing', 
          progress: 30, 
          stage: 'Video generation started...',
          taskId: data.task_id
        })
        
        pollTaskStatus(data.task_id)
      } else {
        throw new Error(data.detail || 'Generation failed')
      }
    } catch (error) {
      console.error('Generation error:', error)
      setProgress({
        status: 'error',
        progress: 0,
        stage: 'Generation failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const pollTaskStatus = async (taskId: string) => {
    const maxAttempts = 60
    let attempts = 0
    
    const poll = async () => {
      try {
        const response = await fetch(`http://localhost:8000/api/v1/status/${taskId}`)
        const data = await response.json()
        
        if (response.ok && data.success && data.data) {
          const { status } = data.data
          
          if (status === 'completed') {
            setProgress({
              status: 'completed',
              progress: 100,
              stage: 'Video generated successfully!',
              videoUrl: data.data.video_url
            })
            return
          } else if (status === 'failed') {
            setProgress({
              status: 'error',
              progress: 0,
              stage: 'Generation failed',
              error: 'Video generation failed'
            })
            return
          }
          
          const progressValue = Math.min(30 + (attempts * 2), 95)
          setProgress({
            status: 'processing',
            progress: progressValue,
            stage: 'Generating video...',
            taskId
          })
        }
        
        attempts++
        if (attempts < maxAttempts) {
          setTimeout(poll, 5000)
        } else {
          setProgress({
            status: 'error',
            progress: 0,
            stage: 'Generation timeout',
            error: 'Generation took too long'
          })
        }
      } catch (error) {
        console.error('Status check error:', error)
        setProgress({
          status: 'error',
          progress: 0,
          stage: 'Status check failed',
          error: error instanceof Error ? error.message : 'Unknown error'
        })
      }
    }
    
    poll()
  }

  return (
    <div className="min-h-screen bg-gray-50 p-6">
      <div className="max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            PaksaTalker AI Video Studio
          </h1>
          <p className="text-lg text-gray-600">
            Create hyper-realistic talking avatars with perfect lip-sync and natural gestures
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Section 1: File Upload */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <CloudArrowUpIcon className="h-6 w-6 text-blue-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">Media Upload</h2>
            </div>
            
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Avatar Image *
              </label>
              <div 
                onClick={() => imageInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
              >
                {imageFile ? (
                  <div className="flex items-center justify-center">
                    <PhotoIcon className="h-8 w-8 text-green-500 mr-2" />
                    <span className="text-sm text-gray-600">{imageFile.name}</span>
                  </div>
                ) : (
                  <div>
                    <PhotoIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">Click to upload image</p>
                    <p className="text-xs text-gray-400">JPG, PNG up to 10MB</p>
                  </div>
                )}
              </div>
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => setImageFile(e.target.files?.[0] || null)}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Audio File (Optional)
              </label>
              <div 
                onClick={() => audioInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 transition-colors"
              >
                {audioFile ? (
                  <div className="flex items-center justify-center">
                    <MicrophoneIcon className="h-8 w-8 text-green-500 mr-2" />
                    <span className="text-sm text-gray-600">{audioFile.name}</span>
                  </div>
                ) : (
                  <div>
                    <MicrophoneIcon className="h-12 w-12 text-gray-400 mx-auto mb-2" />
                    <p className="text-sm text-gray-600">Click to upload audio</p>
                    <p className="text-xs text-gray-400">MP3, WAV up to 50MB</p>
                  </div>
                )}
              </div>
              <input
                ref={audioInputRef}
                type="file"
                accept="audio/*"
                className="hidden"
                onChange={(e) => setAudioFile(e.target.files?.[0] || null)}
              />
            </div>
          </div>

          {/* Section 2: Text Prompt */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <CogIcon className="h-6 w-6 text-purple-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">AI Text Generation</h2>
            </div>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Text Prompt (Qwen AI)
              </label>
              <textarea
                value={textPrompt}
                onChange={(e) => setTextPrompt(e.target.value)}
                placeholder="Enter your text here or describe what you want the avatar to say..."
                className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none"
              />
            </div>

            <div className="grid grid-cols-2 gap-4 mb-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Voice Model
                </label>
                <select 
                  value={settings.voiceModel}
                  onChange={(e) => setSettings({...settings, voiceModel: e.target.value})}
                  className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                >
                  <option value="en-US-JennyNeural">Jenny (Female)</option>
                  <option value="en-US-ChristopherNeural">Christopher (Male)</option>
                  <option value="en-US-AriaNeural">Aria (Female)</option>
                  <option value="en-US-GuyNeural">Guy (Male)</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Language
                </label>
                <select className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500">
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
            </div>

            <button 
              onClick={() => setTextPrompt('Hello! Welcome to PaksaTalker. I am your AI-powered avatar ready to deliver your message with perfect lip-sync and natural expressions.')}
              className="w-full bg-purple-100 text-purple-700 py-2 px-4 rounded-lg hover:bg-purple-200 transition-colors text-sm"
            >
              Use Sample Text
            </button>
          </div>

          {/* Section 3: Generation Settings */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <CogIcon className="h-6 w-6 text-green-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">Generation Settings</h2>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Resolution
                  </label>
                  <select 
                    value={settings.resolution}
                    onChange={(e) => setSettings({...settings, resolution: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="720p">720p HD</option>
                    <option value="1080p">1080p Full HD</option>
                    <option value="4k">4K Ultra HD</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Frame Rate
                  </label>
                  <select 
                    value={settings.fps}
                    onChange={(e) => setSettings({...settings, fps: parseInt(e.target.value)})}
                    className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value={24}>24 FPS</option>
                    <option value={30}>30 FPS</option>
                    <option value={60}>60 FPS</option>
                  </select>
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Expression Intensity: {settings.expressionIntensity}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.expressionIntensity}
                  onChange={(e) => setSettings({...settings, expressionIntensity: parseFloat(e.target.value)})}
                  className="w-full"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Gesture Level
                  </label>
                  <select 
                    value={settings.gestureLevel}
                    onChange={(e) => setSettings({...settings, gestureLevel: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="low">Subtle</option>
                    <option value="medium">Natural</option>
                    <option value="high">Expressive</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Background
                  </label>
                  <select 
                    value={settings.background}
                    onChange={(e) => setSettings({...settings, background: e.target.value})}
                    className="w-full p-2 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="original">Original</option>
                    <option value="blur">Blur</option>
                    <option value="office">Office</option>
                    <option value="studio">Studio</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center space-x-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.enhanceFace}
                    onChange={(e) => setSettings({...settings, enhanceFace: e.target.checked})}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">Enhance Face</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.stabilization}
                    onChange={(e) => setSettings({...settings, stabilization: e.target.checked})}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">Stabilization</span>
                </label>
              </div>
            </div>
          </div>

          {/* Section 4: Output & Progress */}
          <div className="bg-white rounded-lg shadow-lg p-6">
            <div className="flex items-center mb-4">
              <PlayIcon className="h-6 w-6 text-red-600 mr-2" />
              <h2 className="text-xl font-semibold text-gray-900">Generation Output</h2>
            </div>
            
            <button
              onClick={handleGenerate}
              disabled={progress.status === 'processing'}
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors mb-6 font-medium"
            >
              {progress.status === 'processing' ? 'Generating...' : 'Generate Video'}
            </button>

            <div className="space-y-4">
              <div className="flex items-center">
                {progress.status === 'idle' && (
                  <ClockIcon className="h-5 w-5 text-gray-400 mr-2" />
                )}
                {progress.status === 'processing' && (
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-2"></div>
                )}
                {progress.status === 'completed' && (
                  <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2" />
                )}
                {progress.status === 'error' && (
                  <ExclamationTriangleIcon className="h-5 w-5 text-red-500 mr-2" />
                )}
                <span className="text-sm font-medium text-gray-700">
                  {progress.stage}
                </span>
              </div>

              {progress.status === 'processing' && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress.progress}%` }}
                  ></div>
                </div>
              )}

              {progress.status === 'error' && progress.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-700">{progress.error}</p>
                </div>
              )}

              {progress.status === 'completed' && progress.videoUrl && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-green-800">Video Ready!</p>
                      <p className="text-xs text-green-600">Your talking avatar video has been generated</p>
                    </div>
                    <a 
                      href={progress.videoUrl}
                      download
                      className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center"
                    >
                      <ArrowDownTrayIcon className="h-4 w-4 mr-1" />
                      Download
                    </a>
                  </div>
                </div>
              )}

              <div className="bg-gray-50 rounded-lg p-3">
                <h4 className="text-sm font-medium text-gray-700 mb-2">Generation Info</h4>
                <div className="text-xs text-gray-600 space-y-1">
                  <div>Resolution: {settings.resolution}</div>
                  <div>Frame Rate: {settings.fps} FPS</div>
                  <div>Voice: {settings.voiceModel}</div>
                  <div>Gesture Level: {settings.gestureLevel}</div>
                  {progress.taskId && (
                    <div>Task ID: {progress.taskId}</div>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
import { useState, useRef } from 'react'
import { generateVideo, generateVideoFromPrompt } from './utils/api'
import StyleCustomization from './components/StyleCustomization'

interface GenerationSettings {
  resolution: string
  fps: number
  expressionIntensity: number
  gestureLevel: string
  voiceModel: string
  background: string
  enhanceFace: boolean
  stabilization: boolean
  stylePreset?: any
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
    stabilization: true,
    stylePreset: null
  })
  const [progress, setProgress] = useState<GenerationProgress>({
    status: 'idle',
    progress: 0,
    stage: 'Ready to generate'
  })

  const audioInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  const handleGenerate = async () => {
    const isPromptBased = textPrompt && !imageFile && !audioFile
    const isFileBased = imageFile && (audioFile || textPrompt)
    
    if (!isPromptBased && !isFileBased) {
      setProgress({
        status: 'error',
        progress: 0,
        stage: 'Missing required input',
        error: 'Either upload image + audio/text OR enter text prompt only'
      })
      return
    }

    setProgress({ status: 'processing', progress: 10, stage: 'Processing...' })
    
    try {
      const formData = new FormData()
      let data
      
      if (isPromptBased) {
        formData.append('prompt', textPrompt)
        formData.append('voice', settings.voiceModel)
        formData.append('resolution', settings.resolution)
        formData.append('fps', settings.fps.toString())
        formData.append('gestureLevel', settings.gestureLevel)
        
        data = await generateVideoFromPrompt(formData)
      } else {
        formData.append('image', imageFile!)
        if (audioFile) {
          formData.append('audio', audioFile)
        }
        if (textPrompt) {
          formData.append('text', textPrompt)
        }
        
        Object.entries(settings).forEach(([key, value]) => {
          if (value !== null && value !== undefined) {
            formData.append(key, value.toString())
          }
        })
        
        data = await generateVideo(formData)
      }

      if (data.ok && data.success) {
        setProgress({ 
          status: 'processing', 
          progress: 30, 
          stage: 'Video generation started...',
          taskId: data.task_id
        })
        
        pollTaskStatus(data.task_id)
      } else {
        throw new Error(data.error || 'Generation failed')
      }
    } catch (error) {
      setProgress({
        status: 'error',
        progress: 0,
        stage: 'Generation failed',
        error: error instanceof Error ? error.message : 'Unknown error'
      })
    }
  }

  const pollTaskStatus = async (taskId: string) => {
    const maxAttempts = 600  // 50 minutes timeout (600 * 5 seconds)
    let attempts = 0
    
    const poll = async () => {
      try {
        const response = await fetch(`/api/v1/status/${taskId}`)
        const data = await response.json()
        
        if (data.success && data.data) {
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
          
          const progressValue = Math.min(30 + (attempts * 0.1), 95)  // Slower progress increase
          const timeElapsed = Math.floor((attempts * 5) / 60)  // Minutes elapsed
          setProgress({
            status: 'processing',
            progress: progressValue,
            stage: `Generating video... (${timeElapsed}m elapsed)`,
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
            error: 'Generation took longer than 50 minutes. Please try with shorter audio or lower resolution.'
          })
        }
      } catch (error) {
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
        {/* Header */}
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            PaksaTalker AI Video Studio
          </h1>
          <p className="text-lg text-gray-600">
            Create hyper-realistic talking avatars with perfect lip-sync and natural gestures
          </p>
        </div>

        {/* 2x2 Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          
          {/* Section 1: File Upload (Top Left) */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Media Upload</h2>
            </div>
            
            {/* Image Upload */}
            <div className="mb-6">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Avatar Image *
              </label>
              <div 
                onClick={() => imageInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
              >
                {imageFile ? (
                  <div className="flex items-center justify-center">
                    <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center mr-2">
                      <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <span className="text-sm text-gray-600 font-medium">{imageFile.name}</span>
                  </div>
                ) : (
                  <div>
                    <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                      <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                    <p className="text-sm text-gray-600 font-medium">Click to upload image</p>
                    <p className="text-xs text-gray-400 mt-1">JPG, PNG up to 10MB</p>
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

            {/* Audio Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Audio File (Optional)
              </label>
              <div 
                onClick={() => audioInputRef.current?.click()}
                className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-blue-400 hover:bg-blue-50 transition-all duration-200"
              >
                {audioFile ? (
                  <div className="flex items-center justify-center">
                    <div className="w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center mr-2">
                      <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                      </svg>
                    </div>
                    <span className="text-sm text-gray-600 font-medium">{audioFile.name}</span>
                  </div>
                ) : (
                  <div>
                    <div className="w-12 h-12 bg-gray-100 rounded-lg flex items-center justify-center mx-auto mb-3">
                      <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                      </svg>
                    </div>
                    <p className="text-sm text-gray-600 font-medium">Click to upload audio</p>
                    <p className="text-xs text-gray-400 mt-1">MP3, WAV up to 50MB</p>
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

          {/* Section 2: Text Prompt (Top Right) */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-pink-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
                </svg>
              </div>
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
                className="w-full h-32 p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 resize-none text-sm"
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
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
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
                <select className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm">
                  <option value="en">English</option>
                  <option value="es">Spanish</option>
                  <option value="fr">French</option>
                  <option value="de">German</option>
                </select>
              </div>
            </div>

            <button 
              onClick={() => setTextPrompt('Hello! Welcome to PaksaTalker. I am your AI-powered avatar ready to deliver your message with perfect lip-sync and natural expressions.')}
              className="w-full bg-purple-50 text-purple-700 py-2 px-4 rounded-lg hover:bg-purple-100 transition-colors text-sm font-medium"
            >
              Use Sample Text
            </button>
          </div>

          {/* Section 3: Style Customization (Bottom Left) */}
          <StyleCustomization 
            onStyleChange={(preset) => setSettings(prev => ({ ...prev, stylePreset: preset }))}
          />

          {/* Section 4: Generation Settings (Bottom Right) - moved from bottom left */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-green-500 to-emerald-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                </svg>
              </div>
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
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
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
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
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
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
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
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
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
                    className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                  >
                    <option value="original">Original</option>
                    <option value="blur">Blur</option>
                    <option value="office">Office</option>
                    <option value="studio">Studio</option>
                  </select>
                </div>
              </div>

              <div className="flex items-center space-x-6">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.enhanceFace}
                    onChange={(e) => setSettings({...settings, enhanceFace: e.target.checked})}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  <span className="ml-2 text-sm text-gray-700">Enhance Face</span>
                </label>
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={settings.stabilization}
                    onChange={(e) => setSettings({...settings, stabilization: e.target.checked})}
                    className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500 focus:ring-2"
                  />
                  <span className="ml-2 text-sm text-gray-700">Stabilization</span>
                </label>
              </div>
            </div>
          </div>

        </div>
        
        {/* Second Row - Full Width */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
          
          {/* Output & Progress (Left) */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M14.828 14.828a4 4 0 01-5.656 0M9 10h1.01M15 10h1.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Generation Output</h2>
            </div>
            
            {/* Generate Button */}
            <button
              onClick={handleGenerate}
              disabled={progress.status === 'processing'}
              className="w-full bg-blue-600 text-white py-3 px-6 rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition-colors mb-6 font-medium text-sm"
            >
              {progress.status === 'processing' ? 'Generating...' : 'Generate Video'}
            </button>

            {/* Progress Section */}
            <div className="space-y-4">
              {/* Status Indicator */}
              <div className="flex items-center">
                {progress.status === 'idle' && (
                  <div className="w-5 h-5 bg-gray-100 rounded-full flex items-center justify-center mr-3">
                    <svg className="w-3 h-3 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                  </div>
                )}
                {progress.status === 'processing' && (
                  <div className="w-5 h-5 mr-3">
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600"></div>
                  </div>
                )}
                {progress.status === 'completed' && (
                  <div className="w-5 h-5 bg-green-100 rounded-full flex items-center justify-center mr-3">
                    <svg className="w-3 h-3 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                )}
                {progress.status === 'error' && (
                  <div className="w-5 h-5 bg-red-100 rounded-full flex items-center justify-center mr-3">
                    <svg className="w-3 h-3 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
                )}
                <span className="text-sm font-medium text-gray-700">
                  {progress.stage}
                </span>
              </div>

              {/* Progress Bar */}
              {progress.status === 'processing' && (
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${progress.progress}%` }}
                  ></div>
                </div>
              )}

              {/* Error Message */}
              {progress.status === 'error' && progress.error && (
                <div className="bg-red-50 border border-red-200 rounded-lg p-3">
                  <p className="text-sm text-red-700">{progress.error}</p>
                </div>
              )}

              {/* Success & Download */}
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
                      className="bg-green-600 text-white px-4 py-2 rounded-lg hover:bg-green-700 transition-colors flex items-center text-sm font-medium"
                    >
                      <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                      </svg>
                      Download
                    </a>
                  </div>
                </div>
              )}

              {/* Generation Info */}
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
          
          {/* Advanced Settings Panel (Right) */}
          <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
            <div className="flex items-center mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-teal-500 to-cyan-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
                <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 100 4m0-4v2m0-6V4" />
                </svg>
              </div>
              <h2 className="text-xl font-semibold text-gray-900">Advanced Controls</h2>
            </div>
            
            {/* Current Style Display */}
            {settings.stylePreset && (
              <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
                <h3 className="font-medium text-blue-900 mb-2">Active Style</h3>
                <div className="text-sm text-blue-800">
                  <div className="font-medium">{settings.stylePreset.name}</div>
                  <div className="text-xs mt-1">{settings.stylePreset.description}</div>
                  <div className="mt-2 text-xs">
                    Cultural Context: {settings.stylePreset.cultural_context}
                  </div>
                </div>
              </div>
            )}
            
            {/* Mannerism Controls */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900">Fine-Tune Mannerisms</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Micro-Expression Rate: {settings.stylePreset?.micro_expression_rate?.toFixed(2) || '0.50'}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.micro_expression_rate || 0.5}
                  onChange={(e) => {
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, micro_expression_rate: parseFloat(e.target.value) }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Breathing Intensity: {settings.stylePreset?.breathing_intensity?.toFixed(2) || '0.30'}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.breathing_intensity || 0.3}
                  onChange={(e) => {
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, breathing_intensity: parseFloat(e.target.value) }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Posture Variation: {settings.stylePreset?.posture_variation?.toFixed(2) || '0.40'}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.posture_variation || 0.4}
                  onChange={(e) => {
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, posture_variation: parseFloat(e.target.value) }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              {/* Cultural Context Override */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cultural Context Override
                </label>
                <select 
                  value={settings.stylePreset?.cultural_context || 'GLOBAL'}
                  onChange={(e) => {
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, cultural_context: e.target.value }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    }
                  }}
                  className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                >
                  <option value="GLOBAL">Global</option>
                  <option value="WESTERN">Western</option>
                  <option value="EAST_ASIAN">East Asian</option>
                  <option value="MIDDLE_EASTERN">Middle Eastern</option>
                  <option value="SOUTH_ASIAN">South Asian</option>
                  <option value="LATIN_AMERICAN">Latin American</option>
                  <option value="AFRICAN">African</option>
                </select>
              </div>
              
              {/* Reset to Default */}
              <button
                onClick={() => setSettings(prev => ({ ...prev, stylePreset: null }))}
                className="w-full bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors text-sm font-medium"
              >
                Reset to Default Style
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default App
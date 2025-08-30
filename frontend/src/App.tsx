import { useState, useRef, useEffect } from 'react'
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
  completedTasks?: string[]
}

function App() {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [textPrompt, setTextPrompt] = useState('')
  const [settings, setSettings] = useState<GenerationSettings>({
    resolution: '480p',
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
    stage: 'Ready to generate',
    completedTasks: []
  })

  const audioInputRef = useRef<HTMLInputElement>(null)
  const imageInputRef = useRef<HTMLInputElement>(null)

  // Backend capabilities for auto-toggle of advanced features
  const [caps, setCaps] = useState<{models:{sadtalker:boolean;wav2lip2:boolean;emage:boolean;qwen:boolean}}|null>(null)
  useEffect(() => {
    fetch('/api/v1/capabilities')
      .then(r => r.ok ? r.json() : null)
      .then(j => {
        if (j && j.success && j.data && j.data.models) setCaps({models:j.data.models})
      })
      .catch(() => {})
  }, [])

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

    setProgress({ status: 'processing', progress: 10, stage: 'Processing...', completedTasks: ['✓ Input validation complete'] })
    
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

        // Basic settings
        formData.append('resolution', settings.resolution)
        formData.append('fps', settings.fps.toString())

        const canEmage = caps?.models?.emage
        const canW2L = caps?.models?.wav2lip2

        if (canEmage || canW2L) {
          formData.append('useSadTalkerFull', 'true')
          formData.append('useEmage', canEmage ? 'true' : 'false')
          formData.append('useWav2Lip2', canW2L ? 'true' : 'false')

          const resp = await fetch('/api/generate/advanced-video', { method: 'POST', body: formData })
          data = await resp.json()
        } else {
          // Fallback to standard endpoint
          Object.entries(settings).forEach(([key, value]) => {
            if (value !== null && value !== undefined) {
              formData.append(key, value.toString())
            }
          })
          data = await generateVideo(formData)
        }
      }

      if (data.ok && data.success) {
        setProgress({ 
          status: 'processing', 
          progress: 30, 
          stage: 'Video generation started...',
          taskId: data.task_id,
          completedTasks: ['✓ Input validation complete', '✓ Files uploaded successfully', '✓ Generation task created']
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
              videoUrl: data.data.video_url,
              completedTasks: [
                '✓ Input validation complete',
                '✓ Files uploaded successfully', 
                '✓ Generation task created',
                '✓ Audio preprocessing complete',
                '✓ Face detection complete',
                '✓ 3D face reconstruction complete',
                '✓ Expression coefficients generated',
                '✓ Head pose estimation complete',
                '✓ Video frames rendered',
                '✓ Final video compilation complete'
              ]
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
          
          const progressValue = Math.min(30 + (attempts * 0.05), 90)  // Very slow progress increase
          const timeElapsed = Math.floor((attempts * 5) / 60)  // Minutes elapsed
          
          // Simulate completed tasks based on progress
          const completedTasks = ['✓ Input validation complete', '✓ Files uploaded successfully', '✓ Generation task created']
          if (progressValue > 35) completedTasks.push('✓ Audio preprocessing complete')
          if (progressValue > 45) completedTasks.push('✓ Face detection complete')
          if (progressValue > 55) completedTasks.push('✓ 3D face reconstruction complete')
          if (progressValue > 65) completedTasks.push('✓ Expression coefficients generated')
          if (progressValue > 75) completedTasks.push('✓ Head pose estimation complete')
          if (progressValue > 85) completedTasks.push('✓ Video frames rendered')
          
          setProgress({
            status: 'processing',
            progress: progressValue,
            stage: `Generating video... ${Math.round(progressValue)}% (${timeElapsed}m elapsed)`,
            taskId,
            completedTasks
          })
        }
        
        attempts++
        // No timeout - continue polling until completion or failure
        setTimeout(poll, 5000)
        
      } catch (error) {
        // Only stop on actual network errors, not timeouts
        console.error('Status check error:', error)
        // Retry after a longer delay on network errors
        setTimeout(poll, 10000)
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
          <p className="text-lg text-gray-600 mb-6">
            Create hyper-realistic talking avatars with perfect lip-sync and natural gestures
          </p>
          
          {/* Professional Navigation Links */}
          <div className="flex flex-wrap justify-center gap-4 mb-6">
            <a 
              href="https://github.com/paksaitsolutions/PaksaTalker" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-gray-900 text-white rounded-lg hover:bg-gray-800 transition-colors text-sm font-medium shadow-md hover:shadow-lg"
            >
              <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 24 24">
                <path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/>
              </svg>
              GitHub Repository
            </a>
            
            <a 
              href="/api/docs" 
              target="_blank"
              className="inline-flex items-center px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium shadow-md hover:shadow-lg"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              API Documentation
            </a>
            
            <a 
              href="https://docs.paksatalker.com" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors text-sm font-medium shadow-md hover:shadow-lg"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
              </svg>
              Developer Guide
            </a>
            
            <a 
              href="https://paksa.com.pk" 
              target="_blank" 
              rel="noopener noreferrer"
              className="inline-flex items-center px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium shadow-md hover:shadow-lg"
            >
              <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9v-9m0 9c-1.657 0-3-4.03-3-9s1.343-9 3-9m0 18c1.657 0 3-4.03 3-9s-1.343-9-3-9m-9 9a9 9 0 019-9" />
              </svg>
              Paksa IT Solutions
            </a>
          </div>
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
            
            {/* Generate Button for Media Upload */}
            <div className="mt-6">
              {progress.status !== 'processing' ? (
                <button
                  onClick={handleGenerate}
                  disabled={!imageFile || (!audioFile && !textPrompt)}
                  className={`w-full py-3 px-6 rounded-lg font-medium text-sm transition-colors ${
                    imageFile && (audioFile || textPrompt)
                      ? 'bg-blue-600 text-white hover:bg-blue-700'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Generate Video from Media
                </button>
              ) : (
                <div className="space-y-2">
                  <button
                    disabled
                    className="w-full bg-gray-400 text-white py-3 px-6 rounded-lg cursor-not-allowed font-medium text-sm"
                  >
                    Generating...
                  </button>
                  <button
                    onClick={() => {
                      setProgress({
                        status: 'idle',
                        progress: 0,
                        stage: 'Generation cancelled by user'
                      })
                    }}
                    className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
                  >
                    Cancel Generation
                  </button>
                </div>
              )}
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
                  <optgroup label="🇺🇸 English (US)">
                    <option value="en-US-JennyNeural">Jenny (Female)</option>
                    <option value="en-US-ChristopherNeural">Christopher (Male)</option>
                    <option value="en-US-AriaNeural">Aria (Female)</option>
                    <option value="en-US-GuyNeural">Guy (Male)</option>
                    <option value="en-US-SaraNeural">Sara (Female)</option>
                    <option value="en-US-TonyNeural">Tony (Male)</option>
                  </optgroup>
                  <optgroup label="🇬🇧 English (UK)">
                    <option value="en-GB-SoniaNeural">Sonia (Female)</option>
                    <option value="en-GB-RyanNeural">Ryan (Male)</option>
                    <option value="en-GB-LibbyNeural">Libby (Female)</option>
                  </optgroup>
                  <optgroup label="🇪🇸 Spanish">
                    <option value="es-ES-ElviraNeural">Elvira (Female)</option>
                    <option value="es-ES-AlvaroNeural">Alvaro (Male)</option>
                    <option value="es-MX-DaliaNeural">Dalia (Female, Mexico)</option>
                    <option value="es-MX-JorgeNeural">Jorge (Male, Mexico)</option>
                  </optgroup>
                  <optgroup label="🇫🇷 French">
                    <option value="fr-FR-DeniseNeural">Denise (Female)</option>
                    <option value="fr-FR-HenriNeural">Henri (Male)</option>
                    <option value="fr-CA-SylvieNeural">Sylvie (Female, Canada)</option>
                  </optgroup>
                  <optgroup label="🇩🇪 German">
                    <option value="de-DE-KatjaNeural">Katja (Female)</option>
                    <option value="de-DE-ConradNeural">Conrad (Male)</option>
                    <option value="de-AT-IngridNeural">Ingrid (Female, Austria)</option>
                  </optgroup>
                  <optgroup label="🇮🇹 Italian">
                    <option value="it-IT-ElsaNeural">Elsa (Female)</option>
                    <option value="it-IT-DiegoNeural">Diego (Male)</option>
                    <option value="it-IT-IsabellaNeural">Isabella (Female)</option>
                  </optgroup>
                  <optgroup label="🇵🇹 Portuguese">
                    <option value="pt-BR-FranciscaNeural">Francisca (Female, Brazil)</option>
                    <option value="pt-BR-AntonioNeural">Antonio (Male, Brazil)</option>
                    <option value="pt-PT-RaquelNeural">Raquel (Female, Portugal)</option>
                  </optgroup>
                  <optgroup label="🇷🇺 Russian">
                    <option value="ru-RU-SvetlanaNeural">Svetlana (Female)</option>
                    <option value="ru-RU-DmitryNeural">Dmitry (Male)</option>
                  </optgroup>
                  <optgroup label="🇨🇳 Chinese (Mandarin)">
                    <option value="zh-CN-XiaoxiaoNeural">Xiaoxiao (Female)</option>
                    <option value="zh-CN-YunxiNeural">Yunxi (Male)</option>
                    <option value="zh-CN-XiaohanNeural">Xiaohan (Female)</option>
                  </optgroup>
                  <optgroup label="🇯🇵 Japanese">
                    <option value="ja-JP-NanamiNeural">Nanami (Female)</option>
                    <option value="ja-JP-KeitaNeural">Keita (Male)</option>
                    <option value="ja-JP-AoiNeural">Aoi (Female)</option>
                  </optgroup>
                  <optgroup label="🇰🇷 Korean">
                    <option value="ko-KR-SunHiNeural">SunHi (Female)</option>
                    <option value="ko-KR-InJoonNeural">InJoon (Male)</option>
                  </optgroup>
                  <optgroup label="🇮🇳 Hindi">
                    <option value="hi-IN-SwaraNeural">Swara (Female)</option>
                    <option value="hi-IN-MadhurNeural">Madhur (Male)</option>
                  </optgroup>
                  <optgroup label="🇵🇰 Urdu">
                    <option value="ur-PK-UzmaNeural">Uzma (Female)</option>
                    <option value="ur-PK-AsadNeural">Asad (Male)</option>
                  </optgroup>
                  <optgroup label="🇧🇩 Bengali">
                    <option value="bn-BD-NabanitaNeural">Nabanita (Female)</option>
                    <option value="bn-BD-PradeepNeural">Pradeep (Male)</option>
                  </optgroup>
                  <optgroup label="🇱🇰 Tamil">
                    <option value="ta-IN-PallaviNeural">Pallavi (Female)</option>
                    <option value="ta-IN-ValluvarNeural">Valluvar (Male)</option>
                  </optgroup>
                  <optgroup label="🇵🇰 Pakistani Languages">
                    <option value="ps-AF-LatifaNeural">Latifa (Female, Pashto)</option>
                    <option value="ps-AF-GulNawazNeural">Gul Nawaz (Male, Pashto)</option>
                    <option value="fa-IR-DilaraNeural">Dilara (Female, Persian)</option>
                    <option value="fa-IR-FaridNeural">Farid (Male, Persian)</option>
                    <option value="pa-IN-GaganNeural">Gagan (Male, Punjabi)</option>
                    <option value="pa-IN-HarpreetNeural">Harpreet (Female, Punjabi)</option>
                    <option value="sd-PK-AminaNeural">Amina (Female, Sindhi)</option>
                    <option value="sd-PK-AsharNeural">Ashar (Male, Sindhi)</option>
                    <option value="bal-PK-BibiNeural">Bibi (Female, Balochi)</option>
                    <option value="bal-PK-JamNeural">Jam (Male, Balochi)</option>
                    <option value="gjk-PK-RubinaNeural">Rubina (Female, Gojri)</option>
                    <option value="gjk-PK-RashidNeural">Rashid (Male, Gojri)</option>
                  </optgroup>
                  <optgroup label="🇨🇳 Chinese Variants">
                    <option value="zh-CN-XiaoxiaoNeural">Xiaoxiao (Female, Mandarin)</option>
                    <option value="zh-CN-YunxiNeural">Yunxi (Male, Mandarin)</option>
                    <option value="zh-CN-XiaohanNeural">Xiaohan (Female, Mandarin)</option>
                    <option value="zh-HK-HiuMaanNeural">HiuMaan (Female, Cantonese)</option>
                    <option value="zh-HK-WanLungNeural">WanLung (Male, Cantonese)</option>
                    <option value="zh-TW-HsiaoChenNeural">HsiaoChen (Female, Taiwanese)</option>
                    <option value="zh-TW-YunJheNeural">YunJhe (Male, Taiwanese)</option>
                  </optgroup>
                  <optgroup label="🇦🇸 Other Asian Languages">
                    <option value="my-MM-NilarNeural">Nilar (Female, Myanmar)</option>
                    <option value="my-MM-ThihaNeural">Thiha (Male, Myanmar)</option>
                    <option value="km-KH-PisachNeural">Pisach (Male, Khmer)</option>
                    <option value="km-KH-SreymomNeural">Sreymom (Female, Khmer)</option>
                    <option value="lo-LA-ChanthavongNeural">Chanthavong (Male, Lao)</option>
                    <option value="lo-LA-KeomanyNeural">Keomany (Female, Lao)</option>
                    <option value="si-LK-SameeraNeural">Sameera (Male, Sinhala)</option>
                    <option value="si-LK-ThiliniNeural">Thilini (Female, Sinhala)</option>
                    <option value="ne-NP-HemkalaNeural">Hemkala (Female, Nepali)</option>
                    <option value="ne-NP-SagarNeural">Sagar (Male, Nepali)</option>
                    <option value="mn-MN-BatbayarNeural">Batbayar (Male, Mongolian)</option>
                    <option value="mn-MN-YesuiNeural">Yesui (Female, Mongolian)</option>
                    <option value="uz-UZ-MadinaNeural">Madina (Female, Uzbek)</option>
                    <option value="uz-UZ-SardorNeural">Sardor (Male, Uzbek)</option>
                    <option value="kk-KZ-AigulNeural">Aigul (Female, Kazakh)</option>
                    <option value="kk-KZ-DauletNeural">Daulet (Male, Kazakh)</option>
                    <option value="ky-KG-AidaNeural">Aida (Female, Kyrgyz)</option>
                    <option value="ky-KG-TentekNeural">Tentek (Male, Kyrgyz)</option>
                    <option value="tg-TJ-HulkarNeural">Hulkar (Female, Tajik)</option>
                    <option value="tg-TJ-AbdullohNeural">Abdulloh (Male, Tajik)</option>
                  </optgroup>
                  <optgroup label="🇦🇪 Arabic">
                    <option value="ar-SA-ZariyahNeural">Zariyah (Female, Saudi)</option>
                    <option value="ar-SA-HamedNeural">Hamed (Male, Saudi)</option>
                    <option value="ar-EG-SalmaNeural">Salma (Female, Egypt)</option>
                    <option value="ar-EG-ShakirNeural">Shakir (Male, Egypt)</option>
                    <option value="ar-AE-FatimaNeural">Fatima (Female, UAE)</option>
                    <option value="ar-AE-HamadNeural">Hamad (Male, UAE)</option>
                    <option value="ar-JO-SanaNeural">Sana (Female, Jordan)</option>
                    <option value="ar-LB-LaylaNeural">Layla (Female, Lebanon)</option>
                  </optgroup>
                  <optgroup label="🇹🇷 Turkish">
                    <option value="tr-TR-EmelNeural">Emel (Female)</option>
                    <option value="tr-TR-AhmetNeural">Ahmet (Male)</option>
                  </optgroup>
                  <optgroup label="🇳🇱 Dutch">
                    <option value="nl-NL-ColetteNeural">Colette (Female)</option>
                    <option value="nl-NL-MaartenNeural">Maarten (Male)</option>
                  </optgroup>
                  <optgroup label="🇸🇪 Swedish">
                    <option value="sv-SE-SofieNeural">Sofie (Female)</option>
                    <option value="sv-SE-MattiasNeural">Mattias (Male)</option>
                  </optgroup>
                  <optgroup label="🇳🇴 Norwegian">
                    <option value="nb-NO-PernilleNeural">Pernille (Female)</option>
                    <option value="nb-NO-FinnNeural">Finn (Male)</option>
                  </optgroup>
                  <optgroup label="🇩🇰 Danish">
                    <option value="da-DK-ChristelNeural">Christel (Female)</option>
                    <option value="da-DK-JeppeNeural">Jeppe (Male)</option>
                  </optgroup>
                  <optgroup label="🇫🇮 Finnish">
                    <option value="fi-FI-NooraNeural">Noora (Female)</option>
                    <option value="fi-FI-HarriNeural">Harri (Male)</option>
                  </optgroup>
                  <optgroup label="🇵🇱 Polish">
                    <option value="pl-PL-ZofiaNeural">Zofia (Female)</option>
                    <option value="pl-PL-MarekNeural">Marek (Male)</option>
                  </optgroup>
                  <optgroup label="🇨🇿 Czech">
                    <option value="cs-CZ-VlastaNeural">Vlasta (Female)</option>
                    <option value="cs-CZ-AntoninNeural">Antonin (Male)</option>
                  </optgroup>
                  <optgroup label="🇭🇺 Hungarian">
                    <option value="hu-HU-NoemiNeural">Noemi (Female)</option>
                    <option value="hu-HU-TamasNeural">Tamas (Male)</option>
                  </optgroup>
                  <optgroup label="🇬🇷 Greek">
                    <option value="el-GR-AthinaNeural">Athina (Female)</option>
                    <option value="el-GR-NestorNeural">Nestor (Male)</option>
                  </optgroup>
                  <optgroup label="🇹🇭 Thai">
                    <option value="th-TH-AcharaNeural">Achara (Female)</option>
                    <option value="th-TH-NiwatNeural">Niwat (Male)</option>
                  </optgroup>
                  <optgroup label="🇻🇳 Vietnamese">
                    <option value="vi-VN-HoaiMyNeural">HoaiMy (Female)</option>
                    <option value="vi-VN-NamMinhNeural">NamMinh (Male)</option>
                  </optgroup>
                  <optgroup label="🇮🇩 Indonesian">
                    <option value="id-ID-GadisNeural">Gadis (Female)</option>
                    <option value="id-ID-ArdiNeural">Ardi (Male)</option>
                  </optgroup>
                  <optgroup label="🇲🇾 Malay">
                    <option value="ms-MY-YasminNeural">Yasmin (Female)</option>
                    <option value="ms-MY-OsmanNeural">Osman (Male)</option>
                  </optgroup>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Language
                </label>
                <select className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm">
                  <option value="en">🇺🇸 English</option>
                  <option value="es">🇪🇸 Spanish</option>
                  <option value="fr">🇫🇷 French</option>
                  <option value="de">🇩🇪 German</option>
                  <option value="it">🇮🇹 Italian</option>
                  <option value="pt">🇵🇹 Portuguese</option>
                  <option value="ru">🇷🇺 Russian</option>
                  <option value="zh">🇨🇳 Chinese</option>
                  <option value="ja">🇯🇵 Japanese</option>
                  <option value="ko">🇰🇷 Korean</option>
                  <option value="hi">🇮🇳 Hindi</option>
                  <option value="ur">🇵🇰 Urdu</option>
                  <option value="bn">🇧🇩 Bengali</option>
                  <option value="ta">🇱🇰 Tamil</option>
                  <option value="ps">🇦🇫 Pashto</option>
                  <option value="fa">🇮🇷 Persian</option>
                  <option value="pa">🇮🇳 Punjabi</option>
                  <option value="sd">🇵🇰 Sindhi</option>
                  <option value="bal">🇵🇰 Balochi</option>
                  <option value="gjk">🇵🇰 Gojri</option>
                  <option value="my">🇲🇲 Myanmar</option>
                  <option value="km">🇰🇭 Khmer</option>
                  <option value="lo">🇱🇦 Lao</option>
                  <option value="si">🇱🇰 Sinhala</option>
                  <option value="ne">🇳🇵 Nepali</option>
                  <option value="mn">🇲🇳 Mongolian</option>
                  <option value="uz">🇺🇿 Uzbek</option>
                  <option value="kk">🇰🇿 Kazakh</option>
                  <option value="ky">🇰🇬 Kyrgyz</option>
                  <option value="tg">🇹🇯 Tajik</option>
                  <option value="ar">🇦🇪 Arabic</option>
                  <option value="tr">🇹🇷 Turkish</option>
                  <option value="nl">🇳🇱 Dutch</option>
                  <option value="sv">🇸🇪 Swedish</option>
                  <option value="no">🇳🇴 Norwegian</option>
                  <option value="da">🇩🇰 Danish</option>
                  <option value="fi">🇫🇮 Finnish</option>
                  <option value="pl">🇵🇱 Polish</option>
                  <option value="cs">🇨🇿 Czech</option>
                  <option value="hu">🇭🇺 Hungarian</option>
                  <option value="el">🇬🇷 Greek</option>
                  <option value="th">🇹🇭 Thai</option>
                  <option value="vi">🇻🇳 Vietnamese</option>
                  <option value="id">🇮🇩 Indonesian</option>
                  <option value="ms">🇲🇾 Malay</option>
                </select>
              </div>
            </div>

            <button 
              onClick={() => setTextPrompt('Hello! Welcome to PaksaTalker. I am your AI-powered avatar ready to deliver your message with perfect lip-sync and natural expressions.')}
              className="w-full bg-purple-50 text-purple-700 py-2 px-4 rounded-lg hover:bg-purple-100 transition-colors text-sm font-medium mb-4"
            >
              Use Sample Text
            </button>
            
            {/* Generate Button for Prompt */}
            <div>
              {progress.status !== 'processing' ? (
                <button
                  onClick={handleGenerate}
                  disabled={!textPrompt || !!(imageFile || audioFile)}
                  className={`w-full py-3 px-6 rounded-lg font-medium text-sm transition-colors ${
                    textPrompt && !imageFile && !audioFile
                      ? 'bg-purple-600 text-white hover:bg-purple-700'
                      : 'bg-gray-300 text-gray-500 cursor-not-allowed'
                  }`}
                >
                  Generate Video from Prompt
                </button>
              ) : (
                <div className="space-y-2">
                  <button
                    disabled
                    className="w-full bg-gray-400 text-white py-3 px-6 rounded-lg cursor-not-allowed font-medium text-sm"
                  >
                    Generating...
                  </button>
                  <button
                    onClick={() => {
                      setProgress({
                        status: 'idle',
                        progress: 0,
                        stage: 'Generation cancelled by user'
                      })
                    }}
                    className="w-full bg-red-600 text-white py-2 px-4 rounded-lg hover:bg-red-700 transition-colors text-sm font-medium"
                  >
                    Cancel Generation
                  </button>
                </div>
              )}
            </div>
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
                    <option value="240p">240p Basic</option>
                    <option value="360p">360p Low</option>
                    <option value="480p">480p SD</option>
                    <option value="720p">720p HD</option>
                    <option value="1080p">1080p Full HD</option>
                    <option value="1440p">1440p QHD</option>
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

              {/* Completed Tasks */}
              {progress.completedTasks && progress.completedTasks.length > 0 && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-3">
                  <h4 className="text-sm font-medium text-green-800 mb-2">Completed Tasks</h4>
                  <div className="space-y-1">
                    {progress.completedTasks.map((task, index) => (
                      <div key={index} className="text-xs text-green-700 flex items-center">
                        <span className="mr-2">•</span>
                        {task}
                      </div>
                    ))}
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
            {settings.stylePreset ? (
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
            ) : (
              <div className="mb-6 p-4 bg-gray-50 border border-gray-200 rounded-lg">
                <h3 className="font-medium text-gray-700 mb-2">Default Style Active</h3>
                <div className="text-sm text-gray-600">
                  <div className="text-xs">No custom style selected. Using default animation settings.</div>
                  <div className="mt-2 text-xs">
                    Select a style preset from the Style Customization panel to unlock more options.
                  </div>
                </div>
              </div>
            )}
            
            {/* Mannerism Controls */}
            <div className="space-y-4">
              <h3 className="font-medium text-gray-900">Fine-Tune Mannerisms</h3>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Micro-Expression Rate: {(settings.stylePreset?.micro_expression_rate || 0.5).toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.micro_expression_rate || 0.5}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value)
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, micro_expression_rate: value }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    } else {
                      // Create a default preset with the new value
                      const defaultPreset = {
                        preset_id: 'custom',
                        name: 'Custom Style',
                        description: 'User customized style',
                        intensity: 0.7,
                        smoothness: 0.8,
                        expressiveness: 0.7,
                        cultural_context: 'GLOBAL',
                        formality: 0.5,
                        gesture_frequency: 0.7,
                        gesture_amplitude: 1.0,
                        micro_expression_rate: value,
                        breathing_intensity: 0.3,
                        posture_variation: 0.4,
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      }
                      setSettings(prev => ({ ...prev, stylePreset: defaultPreset }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Breathing Intensity: {(settings.stylePreset?.breathing_intensity || 0.3).toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.breathing_intensity || 0.3}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value)
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, breathing_intensity: value }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    } else {
                      const defaultPreset = {
                        preset_id: 'custom',
                        name: 'Custom Style',
                        description: 'User customized style',
                        intensity: 0.7,
                        smoothness: 0.8,
                        expressiveness: 0.7,
                        cultural_context: 'GLOBAL',
                        formality: 0.5,
                        gesture_frequency: 0.7,
                        gesture_amplitude: 1.0,
                        micro_expression_rate: 0.5,
                        breathing_intensity: value,
                        posture_variation: 0.4,
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      }
                      setSettings(prev => ({ ...prev, stylePreset: defaultPreset }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Posture Variation: {(settings.stylePreset?.posture_variation || 0.4).toFixed(2)}
                </label>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.1"
                  value={settings.stylePreset?.posture_variation || 0.4}
                  onChange={(e) => {
                    const value = parseFloat(e.target.value)
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, posture_variation: value }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    } else {
                      const defaultPreset = {
                        preset_id: 'custom',
                        name: 'Custom Style',
                        description: 'User customized style',
                        intensity: 0.7,
                        smoothness: 0.8,
                        expressiveness: 0.7,
                        cultural_context: 'GLOBAL',
                        formality: 0.5,
                        gesture_frequency: 0.7,
                        gesture_amplitude: 1.0,
                        micro_expression_rate: 0.5,
                        breathing_intensity: 0.3,
                        posture_variation: value,
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      }
                      setSettings(prev => ({ ...prev, stylePreset: defaultPreset }))
                    }
                  }}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
              
              {/* Cultural Context Override */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Cultural Context
                </label>
                <select 
                  value={settings.stylePreset?.cultural_context || 'GLOBAL'}
                  onChange={(e) => {
                    const value = e.target.value
                    if (settings.stylePreset) {
                      const updated = { ...settings.stylePreset, cultural_context: value }
                      setSettings(prev => ({ ...prev, stylePreset: updated }))
                    } else {
                      const defaultPreset = {
                        preset_id: 'custom',
                        name: 'Custom Style',
                        description: 'User customized style',
                        intensity: 0.7,
                        smoothness: 0.8,
                        expressiveness: 0.7,
                        cultural_context: value,
                        formality: 0.5,
                        gesture_frequency: 0.7,
                        gesture_amplitude: 1.0,
                        micro_expression_rate: 0.5,
                        breathing_intensity: 0.3,
                        posture_variation: 0.4,
                        created_at: new Date().toISOString(),
                        updated_at: new Date().toISOString()
                      }
                      setSettings(prev => ({ ...prev, stylePreset: defaultPreset }))
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
        
        {/* Footer with Copyright */}
        <footer className="mt-12 pt-8 border-t border-gray-200">
          <div className="text-center">
            <div className="flex flex-col md:flex-row justify-center items-center gap-4 mb-4">
              <div className="flex items-center">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-purple-600 rounded-lg flex items-center justify-center mr-3">
                  <span className="text-white font-bold text-sm">P</span>
                </div>
                <span className="text-lg font-semibold text-gray-900">Paksa IT Solutions</span>
              </div>
              
              <div className="flex items-center gap-6 text-sm text-gray-600">
                <a href="mailto:info@paksait.com" className="hover:text-blue-600 transition-colors">
                  info@paksait.com
                </a>
                <a href="tel:+923001234567" className="hover:text-blue-600 transition-colors">
                  +92 300 123 4567
                </a>
                <a href="https://paksait.com" target="_blank" rel="noopener noreferrer" className="hover:text-blue-600 transition-colors">
                  www.paksait.com
                </a>
              </div>
            </div>
            
            <div className="text-sm text-gray-500 mb-4">
              <p>© {new Date().getFullYear()} Paksa IT Solutions. All rights reserved.</p>
              <p className="mt-1">Powered by cutting-edge AI technology for next-generation video synthesis</p>
            </div>
            
            <div className="flex justify-center gap-6 text-xs text-gray-400">
              <a href="/privacy" className="hover:text-gray-600 transition-colors">Privacy Policy</a>
              <a href="/terms" className="hover:text-gray-600 transition-colors">Terms of Service</a>
              <a href="/support" className="hover:text-gray-600 transition-colors">Support</a>
              <a href="/license" className="hover:text-gray-600 transition-colors">License</a>
            </div>
          </div>
        </footer>
      </div>
    </div>
  )
}

export default App

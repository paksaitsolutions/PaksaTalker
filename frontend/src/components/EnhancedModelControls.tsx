import { useState } from 'react'

interface EnhancedModelSettings {
  // Core AI Models
  useEmage: boolean
  useWav2Lip2: boolean
  useSadTalkerFull: boolean
  
  // Animation Settings
  emotion: string
  emotionIntensity: number
  bodyStyle: string
  avatarType: string
  
  // Quality & Performance
  resolution: string
  fps: number
  renderQuality: string
  lipSyncQuality: string
  
  // Visual Enhancement
  enhanceFace: boolean
  stabilization: boolean
  backgroundType: string
  lightingStyle: string
  postProcessing: string
  
  // Advanced Animation
  gestureAmplitude: number
  headMovement: string
  eyeTracking: boolean
  breathingEffect: boolean
  microExpressions: boolean
  
  // Cultural & Style
  culturalStyle: string
  voiceSync: string
  
  // Technical
  memoryOptimization: boolean
  gpuAcceleration: boolean
  batchProcessing: boolean
}

interface EnhancedModelControlsProps {
  onSettingsChange: (settings: EnhancedModelSettings) => void
}

export default function EnhancedModelControls({ onSettingsChange }: EnhancedModelControlsProps) {
  const [settings, setSettings] = useState<EnhancedModelSettings>({
    // Core AI Models
    useEmage: true,
    useWav2Lip2: true,
    useSadTalkerFull: true,
    
    // Animation Settings
    emotion: 'neutral',
    emotionIntensity: 0.8,
    bodyStyle: 'natural',
    avatarType: 'realistic',
    
    // Quality & Performance
    resolution: '1080p',
    fps: 30,
    renderQuality: 'high',
    lipSyncQuality: 'ultra',
    
    // Visual Enhancement
    enhanceFace: true,
    stabilization: true,
    backgroundType: 'blur',
    lightingStyle: 'natural',
    postProcessing: 'enhanced',
    
    // Advanced Animation
    gestureAmplitude: 1.0,
    headMovement: 'natural',
    eyeTracking: true,
    breathingEffect: true,
    microExpressions: true,
    
    // Cultural & Style
    culturalStyle: 'global',
    voiceSync: 'precise',
    
    // Technical
    memoryOptimization: true,
    gpuAcceleration: true,
    batchProcessing: false
  })

  const updateSettings = (newSettings: Partial<EnhancedModelSettings>) => {
    const updated = { ...settings, ...newSettings }
    setSettings(updated)
    onSettingsChange(updated)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-600 to-blue-600 p-6 text-white">
        <div className="flex items-center">
          <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center mr-3">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold">Enhanced AI Generation</h2>
            <p className="text-purple-100 text-sm">Professional-grade video synthesis controls</p>
          </div>
        </div>
      </div>

      <div className="p-6 space-y-8">
        {/* Core AI Models */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-red-500 rounded-full mr-2"></span>
            AI Model Pipeline
          </h3>
          <div className="grid grid-cols-1 gap-3">
            {[
              { key: 'useEmage', label: 'EMAGE - Realistic Body Expressions', desc: 'Full-body animation with natural gestures', color: 'red' },
              { key: 'useWav2Lip2', label: 'Wav2Lip2 FP8 AOTI - Premium Lip-Sync', desc: 'High-performance lip synchronization', color: 'blue' },
              { key: 'useSadTalkerFull', label: 'SadTalker Full Neural - Facial Animation', desc: 'Complete neural facial expressions', color: 'green' }
            ].map(model => (
              <label key={model.key} className="flex items-start p-3 border border-gray-200 rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={settings[model.key as keyof EnhancedModelSettings] as boolean}
                  onChange={(e) => updateSettings({ [model.key]: e.target.checked })}
                  className={`w-4 h-4 text-${model.color}-600 bg-gray-100 border-gray-300 rounded focus:ring-${model.color}-500 mt-0.5`}
                />
                <div className="ml-3">
                  <span className="text-sm font-medium text-gray-700">{model.label}</span>
                  <p className="text-xs text-gray-500">{model.desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Quality & Performance */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-blue-500 rounded-full mr-2"></span>
            Quality & Performance
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Resolution</label>
              <select
                value={settings.resolution}
                onChange={(e) => updateSettings({ resolution: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="480p">480p (854×480)</option>
                <option value="720p">720p HD (1280×720)</option>
                <option value="1080p">1080p FHD (1920×1080)</option>
                <option value="1440p">1440p QHD (2560×1440)</option>
                <option value="4k">4K UHD (3840×2160)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Frame Rate</label>
              <select
                value={settings.fps}
                onChange={(e) => updateSettings({ fps: parseInt(e.target.value) })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value={24}>24 FPS (Cinema)</option>
                <option value={25}>25 FPS (PAL)</option>
                <option value={30}>30 FPS (Standard)</option>
                <option value={50}>50 FPS (Smooth)</option>
                <option value={60}>60 FPS (Ultra Smooth)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Render Quality</label>
              <select
                value={settings.renderQuality}
                onChange={(e) => updateSettings({ renderQuality: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="draft">Draft (Fast)</option>
                <option value="standard">Standard</option>
                <option value="high">High Quality</option>
                <option value="ultra">Ultra (Slow)</option>
                <option value="production">Production (Best)</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Lip-Sync Quality</label>
              <select
                value={settings.lipSyncQuality}
                onChange={(e) => updateSettings({ lipSyncQuality: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="fast">Fast Processing</option>
                <option value="balanced">Balanced</option>
                <option value="high">High Quality</option>
                <option value="ultra">Ultra (FP8 AOTI)</option>
                <option value="perfect">Perfect Sync</option>
              </select>
            </div>
          </div>
        </div>

        {/* Animation Controls */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-green-500 rounded-full mr-2"></span>
            Animation Controls
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Emotion</label>
              <select
                value={settings.emotion}
                onChange={(e) => updateSettings({ emotion: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 text-sm"
              >
                <option value="neutral">😐 Neutral</option>
                <option value="happy">😊 Happy</option>
                <option value="sad">😢 Sad</option>
                <option value="angry">😠 Angry</option>
                <option value="excited">🤩 Excited</option>
                <option value="surprised">😲 Surprised</option>
                <option value="confident">😎 Confident</option>
                <option value="thoughtful">🤔 Thoughtful</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Emotion Intensity: {settings.emotionIntensity.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={settings.emotionIntensity}
                onChange={(e) => updateSettings({ emotionIntensity: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Body Style</label>
              <select
                value={settings.bodyStyle}
                onChange={(e) => updateSettings({ bodyStyle: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 text-sm"
              >
                <option value="natural">Natural</option>
                <option value="formal">Formal/Business</option>
                <option value="casual">Casual/Relaxed</option>
                <option value="energetic">Energetic</option>
                <option value="subtle">Subtle/Minimal</option>
                <option value="expressive">Highly Expressive</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Gesture Amplitude: {settings.gestureAmplitude.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="2"
                step="0.1"
                value={settings.gestureAmplitude}
                onChange={(e) => updateSettings({ gestureAmplitude: parseFloat(e.target.value) })}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </div>

        {/* Visual Enhancement */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-purple-500 rounded-full mr-2"></span>
            Visual Enhancement
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Background</label>
              <select
                value={settings.backgroundType}
                onChange={(e) => updateSettings({ backgroundType: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-sm"
              >
                <option value="original">Keep Original</option>
                <option value="blur">Blur Background</option>
                <option value="remove">Remove Background</option>
                <option value="studio">Studio Background</option>
                <option value="office">Office Environment</option>
                <option value="nature">Nature Scene</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Lighting Style</label>
              <select
                value={settings.lightingStyle}
                onChange={(e) => updateSettings({ lightingStyle: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-sm"
              >
                <option value="natural">Natural</option>
                <option value="studio">Studio Lighting</option>
                <option value="soft">Soft Lighting</option>
                <option value="dramatic">Dramatic</option>
                <option value="warm">Warm Tone</option>
                <option value="cool">Cool Tone</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Post Processing</label>
              <select
                value={settings.postProcessing}
                onChange={(e) => updateSettings({ postProcessing: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-sm"
              >
                <option value="none">None</option>
                <option value="basic">Basic Enhancement</option>
                <option value="enhanced">Enhanced Quality</option>
                <option value="cinematic">Cinematic</option>
                <option value="professional">Professional Grade</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Head Movement</label>
              <select
                value={settings.headMovement}
                onChange={(e) => updateSettings({ headMovement: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 text-sm"
              >
                <option value="minimal">Minimal</option>
                <option value="natural">Natural</option>
                <option value="expressive">Expressive</option>
                <option value="dynamic">Dynamic</option>
              </select>
            </div>
          </div>
        </div>

        {/* Advanced Features */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-orange-500 rounded-full mr-2"></span>
            Advanced Features
          </h3>
          <div className="grid grid-cols-2 gap-3">
            {[
              { key: 'enhanceFace', label: 'Face Enhancement', desc: 'AI-powered face super-resolution' },
              { key: 'stabilization', label: 'Video Stabilization', desc: 'Reduce camera shake and jitter' },
              { key: 'eyeTracking', label: 'Eye Tracking', desc: 'Natural eye movement and blinking' },
              { key: 'breathingEffect', label: 'Breathing Effect', desc: 'Subtle chest movement simulation' },
              { key: 'microExpressions', label: 'Micro Expressions', desc: 'Subtle facial micro-movements' },
              { key: 'memoryOptimization', label: 'Memory Optimization', desc: 'Efficient GPU memory usage' },
              { key: 'gpuAcceleration', label: 'GPU Acceleration', desc: 'CUDA/OpenCL acceleration' },
              { key: 'batchProcessing', label: 'Batch Processing', desc: 'Process multiple frames together' }
            ].map(feature => (
              <label key={feature.key} className="flex items-start p-3 border border-gray-200 rounded-lg hover:bg-gray-50">
                <input
                  type="checkbox"
                  checked={settings[feature.key as keyof EnhancedModelSettings] as boolean}
                  onChange={(e) => updateSettings({ [feature.key]: e.target.checked })}
                  className="w-4 h-4 text-orange-600 bg-gray-100 border-gray-300 rounded focus:ring-orange-500 mt-0.5"
                />
                <div className="ml-3">
                  <span className="text-sm font-medium text-gray-700">{feature.label}</span>
                  <p className="text-xs text-gray-500">{feature.desc}</p>
                </div>
              </label>
            ))}
          </div>
        </div>

        {/* Cultural & Style */}
        <div>
          <h3 className="font-semibold text-gray-900 mb-4 flex items-center">
            <span className="w-2 h-2 bg-indigo-500 rounded-full mr-2"></span>
            Cultural & Style
          </h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cultural Style</label>
              <select
                value={settings.culturalStyle}
                onChange={(e) => updateSettings({ culturalStyle: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 text-sm"
              >
                <option value="global">Global/Universal</option>
                <option value="western">Western</option>
                <option value="east_asian">East Asian</option>
                <option value="south_asian">South Asian</option>
                <option value="middle_eastern">Middle Eastern</option>
                <option value="latin_american">Latin American</option>
                <option value="african">African</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Voice Sync</label>
              <select
                value={settings.voiceSync}
                onChange={(e) => updateSettings({ voiceSync: e.target.value })}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 text-sm"
              >
                <option value="precise">Precise Sync</option>
                <option value="natural">Natural Timing</option>
                <option value="expressive">Expressive Timing</option>
                <option value="relaxed">Relaxed Timing</option>
              </select>
            </div>
          </div>
        </div>

        {/* Performance Info */}
        <div className="bg-gradient-to-r from-gray-50 to-blue-50 border border-gray-200 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-2">Estimated Performance</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-600">Processing Time:</span>
              <p className="font-medium text-gray-900">
                {settings.renderQuality === 'ultra' ? '5-10 min' : 
                 settings.renderQuality === 'high' ? '2-5 min' : '1-2 min'}
              </p>
            </div>
            <div>
              <span className="text-gray-600">VRAM Usage:</span>
              <p className="font-medium text-gray-900">
                {settings.resolution === '4k' ? '12-16 GB' :
                 settings.resolution === '1440p' ? '8-12 GB' :
                 settings.resolution === '1080p' ? '4-8 GB' : '2-4 GB'}
              </p>
            </div>
            <div>
              <span className="text-gray-600">Quality Score:</span>
              <p className="font-medium text-green-600">
                {(settings.useEmage && settings.useWav2Lip2 && settings.useSadTalkerFull) ? '95%' :
                 (settings.useWav2Lip2 && settings.useSadTalkerFull) ? '85%' :
                 settings.useSadTalkerFull ? '75%' : '65%'}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
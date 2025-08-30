import { useState } from 'react'

interface ModelSettings {
  useEmage: boolean
  useWav2Lip2: boolean
  useSadTalkerFull: boolean
  emotion: string
  bodyStyle: string
  avatarType: string
  lipSyncQuality: string
}

interface AdvancedModelControlsProps {
  onSettingsChange: (settings: ModelSettings) => void
}

export default function AdvancedModelControls({ onSettingsChange }: AdvancedModelControlsProps) {
  const [settings, setSettings] = useState<ModelSettings>({
    useEmage: true,
    useWav2Lip2: true,
    useSadTalkerFull: true,
    emotion: 'neutral',
    bodyStyle: 'natural',
    avatarType: 'realistic',
    lipSyncQuality: 'high'
  })

  const updateSettings = (newSettings: Partial<ModelSettings>) => {
    const updated = { ...settings, ...newSettings }
    setSettings(updated)
    onSettingsChange(updated)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-red-500 to-pink-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Advanced AI Models</h2>
      </div>

      {/* Model Selection */}
      <div className="space-y-4 mb-6">
        <h3 className="font-medium text-gray-900">AI Model Pipeline</h3>
        
        <div className="space-y-3">
          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.useEmage}
              onChange={(e) => updateSettings({ useEmage: e.target.checked })}
              className="w-4 h-4 text-red-600 bg-gray-100 border-gray-300 rounded focus:ring-red-500"
            />
            <div className="ml-3">
              <span className="text-sm font-medium text-gray-700">EMAGE - Realistic Body Expressions</span>
              <p className="text-xs text-gray-500">State-of-the-art full body animation with natural gestures</p>
            </div>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.useWav2Lip2}
              onChange={(e) => updateSettings({ useWav2Lip2: e.target.checked })}
              className="w-4 h-4 text-blue-600 bg-gray-100 border-gray-300 rounded focus:ring-blue-500"
            />
            <div className="ml-3">
              <span className="text-sm font-medium text-gray-700">Wav2Lip2 FP8 AOTI - Premium Lip-Sync</span>
              <p className="text-xs text-gray-500">High-performance lip synchronization with FP8 optimization</p>
            </div>
          </label>

          <label className="flex items-center">
            <input
              type="checkbox"
              checked={settings.useSadTalkerFull}
              onChange={(e) => updateSettings({ useSadTalkerFull: e.target.checked })}
              className="w-4 h-4 text-green-600 bg-gray-100 border-gray-300 rounded focus:ring-green-500"
            />
            <div className="ml-3">
              <span className="text-sm font-medium text-gray-700">SadTalker Full Neural - Facial Animation</span>
              <p className="text-xs text-gray-500">Complete neural network implementation for facial expressions</p>
            </div>
          </label>
        </div>
      </div>

      {/* EMAGE Settings */}
      {settings.useEmage && (
        <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
          <h4 className="font-medium text-red-900 mb-3">EMAGE Body Expression Settings</h4>
          
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-red-700 mb-1">Emotion</label>
              <select
                value={settings.emotion}
                onChange={(e) => updateSettings({ emotion: e.target.value })}
                className="w-full p-2 border border-red-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 text-sm"
              >
                <option value="neutral">😐 Neutral</option>
                <option value="happy">😊 Happy</option>
                <option value="sad">😢 Sad</option>
                <option value="angry">😠 Angry</option>
                <option value="excited">🤩 Excited</option>
                <option value="surprised">😲 Surprised</option>
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-red-700 mb-1">Body Style</label>
              <select
                value={settings.bodyStyle}
                onChange={(e) => updateSettings({ bodyStyle: e.target.value })}
                className="w-full p-2 border border-red-300 rounded-lg focus:ring-2 focus:ring-red-500 focus:border-red-500 text-sm"
              >
                <option value="natural">Natural</option>
                <option value="formal">Formal/Business</option>
                <option value="casual">Casual/Relaxed</option>
                <option value="energetic">Energetic</option>
                <option value="subtle">Subtle/Minimal</option>
              </select>
            </div>
          </div>

          <div className="mt-3">
            <label className="block text-sm font-medium text-red-700 mb-1">Avatar Type</label>
            <div className="flex space-x-4">
              <label className="flex items-center">
                <input
                  type="radio"
                  name="avatarType"
                  value="realistic"
                  checked={settings.avatarType === 'realistic'}
                  onChange={(e) => updateSettings({ avatarType: e.target.value })}
                  className="w-4 h-4 text-red-600"
                />
                <span className="ml-2 text-sm text-red-700">Realistic SMPL-X</span>
              </label>
              <label className="flex items-center">
                <input
                  type="radio"
                  name="avatarType"
                  value="simple"
                  checked={settings.avatarType === 'simple'}
                  onChange={(e) => updateSettings({ avatarType: e.target.value })}
                  className="w-4 h-4 text-red-600"
                />
                <span className="ml-2 text-sm text-red-700">Simple Skeleton</span>
              </label>
            </div>
          </div>
        </div>
      )}

      {/* Wav2Lip2 Settings */}
      {settings.useWav2Lip2 && (
        <div className="mb-6 p-4 bg-blue-50 border border-blue-200 rounded-lg">
          <h4 className="font-medium text-blue-900 mb-3">Wav2Lip2 AOTI Settings</h4>
          
          <div>
            <label className="block text-sm font-medium text-blue-700 mb-1">Lip-Sync Quality</label>
            <select
              value={settings.lipSyncQuality}
              onChange={(e) => updateSettings({ lipSyncQuality: e.target.value })}
              className="w-full p-2 border border-blue-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
            >
              <option value="ultra">Ultra (FP8 AOTI)</option>
              <option value="high">High Quality</option>
              <option value="balanced">Balanced</option>
              <option value="fast">Fast Processing</option>
            </select>
          </div>

          <div className="mt-3 text-xs text-blue-600">
            <p>• FP8 precision for 2x faster inference</p>
            <p>• AOTI optimization for production deployment</p>
            <p>• Real-time face detection and landmark tracking</p>
          </div>
        </div>
      )}

      {/* Model Priority Info */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <h4 className="font-medium text-gray-900 mb-2">Processing Priority</h4>
        <div className="text-sm text-gray-600 space-y-1">
          <div className="flex items-center">
            <span className="w-6 h-6 bg-red-500 text-white rounded-full flex items-center justify-center text-xs mr-2">1</span>
            <span>EMAGE - Full body realistic expressions</span>
          </div>
          <div className="flex items-center">
            <span className="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs mr-2">2</span>
            <span>Wav2Lip2 AOTI - Premium lip synchronization</span>
          </div>
          <div className="flex items-center">
            <span className="w-6 h-6 bg-green-500 text-white rounded-full flex items-center justify-center text-xs mr-2">3</span>
            <span>SadTalker Full - Neural facial animation</span>
          </div>
          <div className="flex items-center">
            <span className="w-6 h-6 bg-gray-500 text-white rounded-full flex items-center justify-center text-xs mr-2">4</span>
            <span>Basic OpenCV - Fallback animation</span>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Models are tried in order. If one fails, the next is automatically used.
        </p>
      </div>

      {/* Performance Warning */}
      {(settings.useEmage && settings.useWav2Lip2 && settings.useSadTalkerFull) && (
        <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
          <div className="flex items-center">
            <svg className="w-5 h-5 text-yellow-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 15.5c-.77.833.192 2.5 1.732 2.5z" />
            </svg>
            <div>
              <p className="text-sm font-medium text-yellow-800">High Performance Mode</p>
              <p className="text-xs text-yellow-600">All models enabled. Requires 16GB+ VRAM for optimal performance.</p>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
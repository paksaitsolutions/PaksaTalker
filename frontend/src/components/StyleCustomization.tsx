import { useState, useEffect } from 'react'
import { createStylePreset, listStylePresets, interpolatePresets, createCulturalVariants, suggestStylePresets } from '../utils/api'

interface StylePreset {
  preset_id: string
  name: string
  description: string
  intensity: number
  smoothness: number
  expressiveness: number
  cultural_context: string
  formality: number
  gesture_frequency: number
  gesture_amplitude: number
  created_at: string
  updated_at: string
}

interface StyleCustomizationProps {
  onStyleChange?: (preset: StylePreset) => void
}

export default function StyleCustomization({ onStyleChange }: StyleCustomizationProps) {
  const [presets, setPresets] = useState<StylePreset[]>([])
  const [selectedPreset, setSelectedPreset] = useState<StylePreset | null>(null)
  const [isCreating, setIsCreating] = useState(false)
  const [isInterpolating, setIsInterpolating] = useState(false)
  const [loading, setLoading] = useState(false)
  
  // Custom preset form state
  const [customPreset, setCustomPreset] = useState({
    name: '',
    description: '',
    intensity: 0.7,
    smoothness: 0.8,
    expressiveness: 0.7,
    cultural_context: 'GLOBAL',
    formality: 0.5,
    gesture_frequency: 0.7,
    gesture_amplitude: 1.0
  })
  
  // Interpolation state
  const [interpolation, setInterpolation] = useState({
    preset1: '',
    preset2: '',
    ratio: 0.5,
    result: null as StylePreset | null
  })

  useEffect(() => {
    loadPresets()
  }, [])

  const loadPresets = async () => {
    try {
      setLoading(true)
      const response = await listStylePresets()
      if (response.success) {
        setPresets(response.presets)
        if (response.presets.length > 0 && !selectedPreset) {
          setSelectedPreset(response.presets[0])
          onStyleChange?.(response.presets[0])
        }
      }
    } catch (error) {
      console.error('Failed to load presets:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSuggest = async () => {
    try {
      setLoading(true)
      const params: any = {}
      if (selectedPreset) {
        params.cultural_context = selectedPreset.cultural_context
        params.formality = selectedPreset.formality
      }
      const res = await suggestStylePresets(params)
      if (res && res.success && Array.isArray(res.suggestions) && res.suggestions.length) {
        setPresets(prev => {
          const ids = new Set(prev.map(p => p.preset_id))
          const merged = [...prev]
          res.suggestions.forEach((p: any) => { if (!ids.has(p.preset_id)) merged.push(p) })
          return merged
        })
        setSelectedPreset(res.suggestions[0])
        onStyleChange?.(res.suggestions[0])
      }
    } catch (e) {
      console.error('Failed to suggest styles:', e)
    } finally {
      setLoading(false)
    }
  }

  const handleCreatePreset = async () => {
    try {
      setLoading(true)
      const response = await createStylePreset(customPreset)
      if (response.success) {
        await loadPresets()
        setIsCreating(false)
        setCustomPreset({
          name: '',
          description: '',
          intensity: 0.7,
          smoothness: 0.8,
          expressiveness: 0.7,
          cultural_context: 'GLOBAL',
          formality: 0.5,
          gesture_frequency: 0.7,
          gesture_amplitude: 1.0
        })
      }
    } catch (error) {
      console.error('Failed to create preset:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleInterpolate = async () => {
    if (!interpolation.preset1 || !interpolation.preset2) return
    
    try {
      setLoading(true)
      const response = await interpolatePresets(
        interpolation.preset1,
        interpolation.preset2,
        interpolation.ratio
      )
      if (response.success) {
        setInterpolation(prev => ({ ...prev, result: response.interpolated_preset }))
      }
    } catch (error) {
      console.error('Failed to interpolate presets:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleCreateCulturalVariants = async (presetId: string) => {
    try {
      setLoading(true)
      const response = await createCulturalVariants(presetId)
      if (response.success) {
        await loadPresets()
      }
    } catch (error) {
      console.error('Failed to create cultural variants:', error)
    } finally {
      setLoading(false)
    }
  }

  const handlePresetSelect = (preset: StylePreset) => {
    setSelectedPreset(preset)
    onStyleChange?.(preset)
  }

  const culturalContexts = [
    { value: 'GLOBAL', label: 'Global' },
    { value: 'WESTERN', label: 'Western' },
    { value: 'EAST_ASIAN', label: 'East Asian' },
    { value: 'MIDDLE_EASTERN', label: 'Middle Eastern' },
    { value: 'SOUTH_ASIAN', label: 'South Asian' },
    { value: 'LATIN_AMERICAN', label: 'Latin American' },
    { value: 'AFRICAN', label: 'African' }
  ]

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 21a4 4 0 01-4-4V5a2 2 0 012-2h4a2 2 0 012 2v12a4 4 0 01-4 4zM21 5a2 2 0 00-2-2h-4a2 2 0 00-2 2v12a4 4 0 004 4h4a2 2 0 002-2V5z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Style Customization</h2>
      </div>

      {/* Preset Selection */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Select Style Preset
        </label>
        <div className="flex items-center justify-between mb-2">
          <div className="text-xs text-gray-500">Choose from existing presets or get AI suggestions</div>
          <button onClick={handleSuggest} disabled={loading} className="text-xs bg-indigo-600 text-white px-3 py-1.5 rounded hover:bg-indigo-700 disabled:bg-gray-400">{loading ? 'Suggesting...' : 'Suggest Styles'}</button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
          {presets.map((preset) => (
            <div
              key={preset.preset_id}
              onClick={() => handlePresetSelect(preset)}
              className={`p-3 border rounded-lg cursor-pointer transition-all ${
                selectedPreset?.preset_id === preset.preset_id
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <div className="font-medium text-sm">{preset.name}</div>
              <div className="text-xs text-gray-500 mt-1">{preset.description}</div>
              <div className="flex items-center justify-between mt-2">
                <span className="text-xs bg-gray-100 px-2 py-1 rounded">
                  {preset.cultural_context}
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation()
                    handleCreateCulturalVariants(preset.preset_id)
                  }}
                  className="text-xs text-blue-600 hover:text-blue-800"
                  disabled={loading}
                >
                  Create Variants
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Action Buttons */}
      <div className="flex gap-3 mb-6">
        <button
          onClick={() => setIsCreating(!isCreating)}
          className="flex-1 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 transition-colors text-sm font-medium"
        >
          Create Custom Preset
        </button>
        <button
          onClick={() => setIsInterpolating(!isInterpolating)}
          className="flex-1 bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 transition-colors text-sm font-medium"
        >
          Interpolate Presets
        </button>
      </div>

      {/* Create Custom Preset Form */}
      {isCreating && (
        <div className="border border-gray-200 rounded-lg p-4 mb-6 bg-gray-50">
          <h3 className="font-medium text-gray-900 mb-4">Create Custom Preset</h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
              <input
                type="text"
                value={customPreset.name}
                onChange={(e) => setCustomPreset(prev => ({ ...prev, name: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
                placeholder="My Custom Style"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cultural Context</label>
              <select
                value={customPreset.cultural_context}
                onChange={(e) => setCustomPreset(prev => ({ ...prev, cultural_context: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              >
                {culturalContexts.map(context => (
                  <option key={context.value} value={context.value}>{context.label}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">Description</label>
            <textarea
              value={customPreset.description}
              onChange={(e) => setCustomPreset(prev => ({ ...prev, description: e.target.value }))}
              className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              rows={2}
              placeholder="Describe your custom style..."
            />
          </div>

          {/* Parameter Sliders */}
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            {[
              { key: 'intensity', label: 'Intensity', min: 0, max: 1, step: 0.1 },
              { key: 'smoothness', label: 'Smoothness', min: 0, max: 1, step: 0.1 },
              { key: 'expressiveness', label: 'Expressiveness', min: 0, max: 1, step: 0.1 },
              { key: 'formality', label: 'Formality', min: 0, max: 1, step: 0.1 },
              { key: 'gesture_frequency', label: 'Gesture Frequency', min: 0, max: 1, step: 0.1 },
              { key: 'gesture_amplitude', label: 'Gesture Amplitude', min: 0, max: 2, step: 0.1 }
            ].map(param => (
              <div key={param.key}>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  {param.label}: {customPreset[param.key as keyof typeof customPreset]}
                </label>
                <input
                  type="range"
                  min={param.min}
                  max={param.max}
                  step={param.step}
                  value={customPreset[param.key as keyof typeof customPreset]}
                  onChange={(e) => setCustomPreset(prev => ({ 
                    ...prev, 
                    [param.key]: parseFloat(e.target.value) 
                  }))}
                  className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                />
              </div>
            ))}
          </div>

          <div className="flex gap-3">
            <button
              onClick={handleCreatePreset}
              disabled={loading || !customPreset.name}
              className="bg-green-600 text-white py-2 px-4 rounded-lg hover:bg-green-700 disabled:bg-gray-400 transition-colors text-sm font-medium"
            >
              {loading ? 'Creating...' : 'Create Preset'}
            </button>
            <button
              onClick={() => setIsCreating(false)}
              className="bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors text-sm font-medium"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      {/* Interpolation Panel */}
      {isInterpolating && (
        <div className="border border-gray-200 rounded-lg p-4 mb-6 bg-gray-50">
          <h3 className="font-medium text-gray-900 mb-4">Interpolate Between Presets</h3>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">First Preset</label>
              <select
                value={interpolation.preset1}
                onChange={(e) => setInterpolation(prev => ({ ...prev, preset1: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              >
                <option value="">Select preset...</option>
                {presets.map(preset => (
                  <option key={preset.preset_id} value={preset.preset_id}>{preset.name}</option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Second Preset</label>
              <select
                value={interpolation.preset2}
                onChange={(e) => setInterpolation(prev => ({ ...prev, preset2: e.target.value }))}
                className="w-full p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
              >
                <option value="">Select preset...</option>
                {presets.map(preset => (
                  <option key={preset.preset_id} value={preset.preset_id}>{preset.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="mb-4">
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Interpolation Ratio: {interpolation.ratio}
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={interpolation.ratio}
              onChange={(e) => setInterpolation(prev => ({ ...prev, ratio: parseFloat(e.target.value) }))}
              className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
            />
            <div className="flex justify-between text-xs text-gray-500 mt-1">
              <span>100% First</span>
              <span>50/50 Mix</span>
              <span>100% Second</span>
            </div>
          </div>

          <div className="flex gap-3 mb-4">
            <button
              onClick={handleInterpolate}
              disabled={loading || !interpolation.preset1 || !interpolation.preset2}
              className="bg-purple-600 text-white py-2 px-4 rounded-lg hover:bg-purple-700 disabled:bg-gray-400 transition-colors text-sm font-medium"
            >
              {loading ? 'Interpolating...' : 'Generate Blend'}
            </button>
            <button
              onClick={() => setIsInterpolating(false)}
              className="bg-gray-500 text-white py-2 px-4 rounded-lg hover:bg-gray-600 transition-colors text-sm font-medium"
            >
              Cancel
            </button>
          </div>

          {/* Interpolation Result */}
          {interpolation.result && (
            <div className="bg-white border border-gray-200 rounded-lg p-3">
              <h4 className="font-medium text-gray-900 mb-2">Interpolated Result</h4>
              <div className="text-sm text-gray-600 mb-2">{interpolation.result.name}</div>
              <div className="grid grid-cols-2 gap-2 text-xs">
                <div>Intensity: {interpolation.result.intensity.toFixed(2)}</div>
                <div>Smoothness: {interpolation.result.smoothness.toFixed(2)}</div>
                <div>Expressiveness: {interpolation.result.expressiveness.toFixed(2)}</div>
                <div>Formality: {interpolation.result.formality.toFixed(2)}</div>
              </div>
              <button
                onClick={() => handlePresetSelect(interpolation.result!)}
                className="mt-2 w-full bg-blue-600 text-white py-1 px-3 rounded text-xs hover:bg-blue-700 transition-colors"
              >
                Use This Style
              </button>
            </div>
          )}
        </div>
      )}

      {/* Current Style Preview */}
      {selectedPreset && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
          <h3 className="font-medium text-gray-900 mb-3">Current Style: {selectedPreset.name}</h3>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-600">Intensity:</span>
              <span className="font-medium">{selectedPreset.intensity.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Smoothness:</span>
              <span className="font-medium">{selectedPreset.smoothness.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Expressiveness:</span>
              <span className="font-medium">{selectedPreset.expressiveness.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Formality:</span>
              <span className="font-medium">{selectedPreset.formality.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Gesture Freq:</span>
              <span className="font-medium">{selectedPreset.gesture_frequency.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-600">Gesture Amp:</span>
              <span className="font-medium">{selectedPreset.gesture_amplitude.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

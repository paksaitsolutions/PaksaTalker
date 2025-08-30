import { useState, useEffect } from 'react'

interface PersonaInfo {
  name: string
  description: string
  guidelines: string[]
  response_format: string
  constraints: string[]
}

interface GenerationResult {
  enhanced_prompt: string
  generated_content: string
  quality_metrics: {
    quality_score: number
    word_count: number
    estimated_duration: number
    safety_passed: boolean
  }
  enhancements: string[]
  persona_applied: string
  safety_level: string
}

export default function PromptEngineering() {
  const [personas, setPersonas] = useState<Record<string, PersonaInfo>>({})
  const [selectedPersona, setSelectedPersona] = useState('professional')
  const [topic, setTopic] = useState('')
  const [duration, setDuration] = useState(60)
  const [emotion, setEmotion] = useState('neutral')
  const [context, setContext] = useState('')
  const [safetyLevel, setSafetyLevel] = useState('moderate')
  const [includeExamples, setIncludeExamples] = useState(true)
  const [result, setResult] = useState<GenerationResult | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  const [activeTab, setActiveTab] = useState('generate')

  useEffect(() => {
    fetchPersonas()
  }, [])

  const fetchPersonas = async () => {
    try {
      const response = await fetch('/api/v1/prompt/personas')
      const data = await response.json()
      if (data.success) {
        setPersonas(data.personas)
      }
    } catch (error) {
      console.error('Failed to fetch personas:', error)
    }
  }

  const generateContent = async () => {
    if (!topic.trim()) return

    setIsGenerating(true)
    try {
      const formData = new FormData()
      formData.append('topic', topic)
      formData.append('persona', selectedPersona)
      formData.append('duration', duration.toString())
      formData.append('emotion', emotion)
      formData.append('context', context)
      formData.append('safety_level', safetyLevel)
      formData.append('include_examples', includeExamples.toString())

      const response = await fetch('/api/v1/prompt/generate', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()
      if (data.success) {
        setResult(data)
      } else {
        console.error('Generation failed:', data.detail)
      }
    } catch (error) {
      console.error('Generation error:', error)
    } finally {
      setIsGenerating(false)
    }
  }

  const getQualityColor = (score: number) => {
    if (score >= 80) return 'text-green-600'
    if (score >= 60) return 'text-yellow-600'
    return 'text-red-600'
  }

  const getQualityLabel = (score: number) => {
    if (score >= 80) return 'Excellent'
    if (score >= 60) return 'Good'
    return 'Needs Improvement'
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200">
      {/* Header */}
      <div className="bg-gradient-to-r from-indigo-600 to-purple-600 p-6 text-white rounded-t-xl">
        <div className="flex items-center">
          <div className="w-10 h-10 bg-white/20 rounded-xl flex items-center justify-center mr-3">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
            </svg>
          </div>
          <div>
            <h2 className="text-xl font-semibold">Advanced Prompt Engineering</h2>
            <p className="text-indigo-100 text-sm">AI-powered content generation with persona-based prompting</p>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200">
        <nav className="flex space-x-8 px-6">
          {[
            { id: 'generate', label: 'Generate Content', icon: '🎯' },
            { id: 'personas', label: 'Personas', icon: '👤' },
            { id: 'examples', label: 'Examples', icon: '📝' },
            { id: 'safety', label: 'Safety Filters', icon: '🛡️' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`py-4 px-2 border-b-2 font-medium text-sm ${
                activeTab === tab.id
                  ? 'border-indigo-500 text-indigo-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
              }`}
            >
              <span className="mr-2">{tab.icon}</span>
              {tab.label}
            </button>
          ))}
        </nav>
      </div>

      <div className="p-6">
        {/* Generate Content Tab */}
        {activeTab === 'generate' && (
          <div className="space-y-6">
            {/* Input Form */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Topic</label>
                  <textarea
                    value={topic}
                    onChange={(e) => setTopic(e.target.value)}
                    placeholder="Enter the topic you want to create content about..."
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:border-indigo-500"
                    rows={3}
                  />
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Persona</label>
                    <select
                      value={selectedPersona}
                      onChange={(e) => setSelectedPersona(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      {Object.entries(personas).map(([key, persona]) => (
                        <option key={key} value={key}>{persona.name}</option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Duration: {duration}s
                    </label>
                    <input
                      type="range"
                      min="30"
                      max="300"
                      step="15"
                      value={duration}
                      onChange={(e) => setDuration(parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Emotion</label>
                    <select
                      value={emotion}
                      onChange={(e) => setEmotion(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="neutral">😐 Neutral</option>
                      <option value="happy">😊 Happy</option>
                      <option value="excited">🤩 Excited</option>
                      <option value="confident">😎 Confident</option>
                      <option value="thoughtful">🤔 Thoughtful</option>
                      <option value="enthusiastic">🎉 Enthusiastic</option>
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">Safety Level</label>
                    <select
                      value={safetyLevel}
                      onChange={(e) => setSafetyLevel(e.target.value)}
                      className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="strict">🔒 Strict</option>
                      <option value="moderate">⚖️ Moderate</option>
                      <option value="relaxed">🔓 Relaxed</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Context (Optional)</label>
                  <input
                    type="text"
                    value={context}
                    onChange={(e) => setContext(e.target.value)}
                    placeholder="Additional context or specific requirements..."
                    className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-indigo-500"
                  />
                </div>

                <div className="flex items-center">
                  <input
                    type="checkbox"
                    id="includeExamples"
                    checked={includeExamples}
                    onChange={(e) => setIncludeExamples(e.target.checked)}
                    className="w-4 h-4 text-indigo-600 bg-gray-100 border-gray-300 rounded focus:ring-indigo-500"
                  />
                  <label htmlFor="includeExamples" className="ml-2 text-sm text-gray-700">
                    Include few-shot learning examples
                  </label>
                </div>

                <button
                  onClick={generateContent}
                  disabled={!topic.trim() || isGenerating}
                  className="w-full bg-indigo-600 text-white py-3 px-4 rounded-lg hover:bg-indigo-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
                >
                  {isGenerating ? (
                    <>
                      <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                      Generating...
                    </>
                  ) : (
                    <>
                      <svg className="w-5 h-5 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                      </svg>
                      Generate Content
                    </>
                  )}
                </button>
              </div>

              {/* Persona Info */}
              <div className="bg-gray-50 rounded-lg p-4">
                <h3 className="font-medium text-gray-900 mb-3">Selected Persona</h3>
                {personas[selectedPersona] && (
                  <div className="space-y-3">
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Description</h4>
                      <p className="text-sm text-gray-600">{personas[selectedPersona].description}</p>
                    </div>
                    
                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Guidelines</h4>
                      <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                        {personas[selectedPersona].guidelines.slice(0, 3).map((guideline, index) => (
                          <li key={index}>{guideline}</li>
                        ))}
                      </ul>
                    </div>

                    <div>
                      <h4 className="text-sm font-medium text-gray-700">Response Format</h4>
                      <p className="text-sm text-gray-600">{personas[selectedPersona].response_format}</p>
                    </div>
                  </div>
                )}
              </div>
            </div>

            {/* Results */}
            {result && (
              <div className="border-t border-gray-200 pt-6">
                <h3 className="text-lg font-medium text-gray-900 mb-4">Generated Content</h3>
                
                {/* Quality Metrics */}
                <div className="grid grid-cols-4 gap-4 mb-6">
                  <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
                    <div className={`text-2xl font-bold ${getQualityColor(result.quality_metrics.quality_score)}`}>
                      {result.quality_metrics.quality_score.toFixed(0)}%
                    </div>
                    <div className="text-sm text-gray-600">Quality Score</div>
                    <div className={`text-xs ${getQualityColor(result.quality_metrics.quality_score)}`}>
                      {getQualityLabel(result.quality_metrics.quality_score)}
                    </div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-blue-600">{result.quality_metrics.word_count}</div>
                    <div className="text-sm text-gray-600">Words</div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
                    <div className="text-2xl font-bold text-green-600">
                      {result.quality_metrics.estimated_duration.toFixed(0)}s
                    </div>
                    <div className="text-sm text-gray-600">Duration</div>
                  </div>
                  
                  <div className="bg-white border border-gray-200 rounded-lg p-4 text-center">
                    <div className={`text-2xl font-bold ${result.quality_metrics.safety_passed ? 'text-green-600' : 'text-red-600'}`}>
                      {result.quality_metrics.safety_passed ? '✓' : '✗'}
                    </div>
                    <div className="text-sm text-gray-600">Safety Check</div>
                  </div>
                </div>

                {/* Generated Content */}
                <div className="bg-gray-50 rounded-lg p-4 mb-4">
                  <h4 className="font-medium text-gray-900 mb-2">Generated Script</h4>
                  <p className="text-gray-700 leading-relaxed">{result.generated_content}</p>
                </div>

                {/* Enhancements */}
                {result.enhancements.length > 0 && (
                  <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
                    <h4 className="font-medium text-yellow-800 mb-2">💡 Enhancement Suggestions</h4>
                    <ul className="text-sm text-yellow-700 space-y-1">
                      {result.enhancements.map((enhancement, index) => (
                        <li key={index}>• {enhancement}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Personas Tab */}
        {activeTab === 'personas' && (
          <div className="space-y-4">
            <h3 className="text-lg font-medium text-gray-900">Available Personas</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {Object.entries(personas).map(([key, persona]) => (
                <div key={key} className="border border-gray-200 rounded-lg p-4">
                  <h4 className="font-medium text-gray-900 mb-2">{persona.name}</h4>
                  <p className="text-sm text-gray-600 mb-3">{persona.description}</p>
                  
                  <div className="space-y-2">
                    <div>
                      <span className="text-xs font-medium text-gray-500">GUIDELINES:</span>
                      <ul className="text-xs text-gray-600 list-disc list-inside mt-1">
                        {persona.guidelines.slice(0, 2).map((guideline, index) => (
                          <li key={index}>{guideline}</li>
                        ))}
                      </ul>
                    </div>
                    
                    <div>
                      <span className="text-xs font-medium text-gray-500">FORMAT:</span>
                      <p className="text-xs text-gray-600">{persona.response_format}</p>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Other tabs would be implemented similarly */}
        {activeTab === 'examples' && (
          <div className="text-center py-8">
            <div className="text-gray-500">Few-shot examples coming soon...</div>
          </div>
        )}

        {activeTab === 'safety' && (
          <div className="text-center py-8">
            <div className="text-gray-500">Safety filter configuration coming soon...</div>
          </div>
        )}
      </div>
    </div>
  )
}
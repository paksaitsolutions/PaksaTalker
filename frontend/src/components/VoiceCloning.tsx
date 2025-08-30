import { useState, useRef } from 'react'

export default function VoiceCloning() {
  const [audioFile, setAudioFile] = useState<File | null>(null)
  const [speakerName, setSpeakerName] = useState('')
  const [isTraining, setIsTraining] = useState(false)
  const [clonedVoices, setClonedVoices] = useState<any[]>([])
  const audioInputRef = useRef<HTMLInputElement>(null)

  const handleVoiceCloning = async () => {
    if (!audioFile || !speakerName) return

    setIsTraining(true)
    try {
      const formData = new FormData()
      formData.append('audio', audioFile)
      formData.append('speaker_name', speakerName)

      const response = await fetch('/api/voices', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const data = await response.json()
        setClonedVoices(prev => [...prev, data])
        setAudioFile(null)
        setSpeakerName('')
      }
    } catch (error) {
      console.error('Voice cloning failed:', error)
    }
    setIsTraining(false)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-purple-500 to-indigo-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Voice Cloning</h2>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Speaker Name
          </label>
          <input
            type="text"
            value={speakerName}
            onChange={(e) => setSpeakerName(e.target.value)}
            placeholder="Enter speaker name"
            className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-purple-500 focus:border-purple-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Voice Sample (30+ seconds recommended)
          </label>
          <div 
            onClick={() => audioInputRef.current?.click()}
            className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center cursor-pointer hover:border-purple-400 hover:bg-purple-50 transition-all duration-200"
          >
            {audioFile ? (
              <div className="flex items-center justify-center">
                <div className="w-8 h-8 bg-purple-100 rounded-lg flex items-center justify-center mr-2">
                  <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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
                <p className="text-sm text-gray-600 font-medium">Click to upload voice sample</p>
                <p className="text-xs text-gray-400 mt-1">WAV, MP3 up to 100MB</p>
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

        <button
          onClick={handleVoiceCloning}
          disabled={!audioFile || !speakerName || isTraining}
          className={`w-full py-3 px-6 rounded-lg font-medium text-sm transition-colors ${
            audioFile && speakerName && !isTraining
              ? 'bg-purple-600 text-white hover:bg-purple-700'
              : 'bg-gray-300 text-gray-500 cursor-not-allowed'
          }`}
        >
          {isTraining ? 'Training Voice Model...' : 'Clone Voice'}
        </button>

        {clonedVoices.length > 0 && (
          <div className="mt-6">
            <h3 className="font-medium text-gray-900 mb-3">Cloned Voices</h3>
            <div className="space-y-2">
              {clonedVoices.map((voice, idx) => (
                <div key={idx} className="flex items-center justify-between p-3 bg-purple-50 rounded-lg">
                  <span className="text-sm font-medium text-purple-900">{voice.speaker_name}</span>
                  <span className="text-xs text-purple-600">Ready</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
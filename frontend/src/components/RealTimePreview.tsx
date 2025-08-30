import { useState, useEffect } from 'react'

interface RealTimePreviewProps {
  imageFile: File | null
  audioFile: File | null
  settings: any
}

export default function RealTimePreview({ imageFile, audioFile, settings }: RealTimePreviewProps) {
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isGenerating, setIsGenerating] = useState(false)
  // const canvasRef = useRef<HTMLCanvasElement>(null)

  const generatePreview = async () => {
    if (!imageFile) return

    setIsGenerating(true)
    try {
      const formData = new FormData()
      formData.append('image', imageFile)
      if (audioFile) formData.append('audio', audioFile)
      formData.append('preview', 'true')
      formData.append('duration', '3')

      const response = await fetch('/api/generate/preview', {
        method: 'POST',
        body: formData
      })

      if (response.ok) {
        const blob = await response.blob()
        const url = URL.createObjectURL(blob)
        setPreviewUrl(url)
      }
    } catch (error) {
      console.error('Preview generation failed:', error)
    }
    setIsGenerating(false)
  }

  useEffect(() => {
    if (imageFile) {
      generatePreview()
    }
  }, [imageFile, audioFile, settings])

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-4">
        <div className="w-8 h-8 bg-gradient-to-br from-yellow-500 to-orange-500 rounded-lg flex items-center justify-center mr-3">
          <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
          </svg>
        </div>
        <h3 className="text-lg font-semibold text-gray-900">Real-Time Preview</h3>
      </div>

      <div className="aspect-video bg-gray-100 rounded-lg overflow-hidden">
        {isGenerating ? (
          <div className="w-full h-full flex items-center justify-center">
            <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
            <span className="ml-2 text-sm text-gray-600">Generating preview...</span>
          </div>
        ) : previewUrl ? (
          <video 
            src={previewUrl} 
            controls 
            className="w-full h-full object-cover"
            autoPlay
            muted
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500">
            <div className="text-center">
              <svg className="w-12 h-12 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" />
              </svg>
              <p className="text-sm">Upload image to see preview</p>
            </div>
          </div>
        )}
      </div>

      <button
        onClick={generatePreview}
        disabled={!imageFile || isGenerating}
        className="w-full mt-4 py-2 px-4 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:bg-gray-400 text-sm font-medium"
      >
        {isGenerating ? 'Generating...' : 'Refresh Preview'}
      </button>
    </div>
  )
}
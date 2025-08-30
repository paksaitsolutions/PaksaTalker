import { useState, useRef } from 'react'

interface BatchJob {
  id: string
  name: string
  status: 'pending' | 'processing' | 'completed' | 'failed'
  progress: number
  files: { image: File; audio?: File; text?: string }
}

export default function BatchProcessing() {
  const [jobs, setJobs] = useState<BatchJob[]>([])
  const [isProcessing, setIsProcessing] = useState(false)
  const folderInputRef = useRef<HTMLInputElement>(null)

  const handleFolderUpload = (event: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(event.target.files || [])
    const imageFiles = files.filter(f => f.type.startsWith('image/'))
    const audioFiles = files.filter(f => f.type.startsWith('audio/'))

    const newJobs: BatchJob[] = imageFiles.map((image, idx) => ({
      id: `job-${Date.now()}-${idx}`,
      name: image.name,
      status: 'pending',
      progress: 0,
      files: {
        image,
        audio: audioFiles[idx] || undefined
      }
    }))

    setJobs(prev => [...prev, ...newJobs])
  }

  const processBatch = async () => {
    setIsProcessing(true)
    
    for (const job of jobs.filter(j => j.status === 'pending')) {
      setJobs(prev => prev.map(j => 
        j.id === job.id ? { ...j, status: 'processing' } : j
      ))

      try {
        const formData = new FormData()
        formData.append('image', job.files.image)
        if (job.files.audio) formData.append('audio', job.files.audio)
        if (job.files.text) formData.append('text', job.files.text)

        const response = await fetch('/api/generate/video', {
          method: 'POST',
          body: formData
        })

        if (response.ok) {
          setJobs(prev => prev.map(j => 
            j.id === job.id ? { ...j, status: 'completed', progress: 100 } : j
          ))
        } else {
          throw new Error('Generation failed')
        }
      } catch (error) {
        setJobs(prev => prev.map(j => 
          j.id === job.id ? { ...j, status: 'failed' } : j
        ))
      }
    }

    setIsProcessing(false)
  }

  const clearCompleted = () => {
    setJobs(prev => prev.filter(j => j.status !== 'completed'))
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-orange-500 to-red-500 rounded-xl flex items-center justify-center mr-3 shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11H5m14 0a2 2 0 012 2v6a2 2 0 01-2 2H5a2 2 0 01-2-2v-6a2 2 0 012-2m14 0V9a2 2 0 00-2-2M5 11V9a2 2 0 012-2m0 0V5a2 2 0 012-2h6a2 2 0 012 2v2M7 7h10" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Batch Processing</h2>
      </div>

      <div className="space-y-4">
        <div>
          <button
            onClick={() => folderInputRef.current?.click()}
            className="w-full py-3 px-6 border-2 border-dashed border-gray-300 rounded-lg hover:border-orange-400 hover:bg-orange-50 transition-all duration-200 text-center"
          >
            <div className="flex items-center justify-center">
              <svg className="w-6 h-6 text-gray-400 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <span className="text-sm font-medium text-gray-600">Upload Folder with Images & Audio</span>
            </div>
          </button>
          <input
            ref={folderInputRef}
            type="file"
            multiple
            {...({ webkitdirectory: '' } as any)}
            className="hidden"
            onChange={handleFolderUpload}
          />
        </div>

        {jobs.length > 0 && (
          <>
            <div className="flex justify-between items-center">
              <span className="text-sm font-medium text-gray-700">
                {jobs.length} jobs • {jobs.filter(j => j.status === 'completed').length} completed
              </span>
              <div className="space-x-2">
                <button
                  onClick={processBatch}
                  disabled={isProcessing || jobs.filter(j => j.status === 'pending').length === 0}
                  className="px-4 py-2 bg-orange-600 text-white rounded-lg hover:bg-orange-700 disabled:bg-gray-400 text-sm font-medium"
                >
                  {isProcessing ? 'Processing...' : 'Start Batch'}
                </button>
                <button
                  onClick={clearCompleted}
                  className="px-4 py-2 bg-gray-500 text-white rounded-lg hover:bg-gray-600 text-sm font-medium"
                >
                  Clear Completed
                </button>
              </div>
            </div>

            <div className="max-h-64 overflow-y-auto space-y-2">
              {jobs.map(job => (
                <div key={job.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                  <div className="flex items-center">
                    <div className={`w-3 h-3 rounded-full mr-3 ${
                      job.status === 'pending' ? 'bg-gray-400' :
                      job.status === 'processing' ? 'bg-orange-500 animate-pulse' :
                      job.status === 'completed' ? 'bg-green-500' : 'bg-red-500'
                    }`}></div>
                    <span className="text-sm font-medium text-gray-700">{job.name}</span>
                  </div>
                  <span className="text-xs text-gray-500 capitalize">{job.status}</span>
                </div>
              ))}
            </div>
          </>
        )}
      </div>
    </div>
  )
}
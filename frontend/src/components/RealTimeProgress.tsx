import { useState, useEffect } from 'react'

interface ProgressStep {
  id: string
  title: string
  status: 'pending' | 'active' | 'completed' | 'failed'
  progress: number
  details: string
  timestamp: string
}

interface ProgressData {
  type: string
  task_id: string
  overall_progress: number
  current_step: ProgressStep | null
  steps: ProgressStep[]
  timestamp: string
  started_at?: string
  elapsed_seconds?: number
  eta_seconds?: number | null
  estimated_total_seconds?: number | null
}

interface RealTimeProgressProps {
  taskId: string
  onComplete?: (taskId: string) => void
  onError?: (error: string) => void
}

export default function RealTimeProgress({ taskId, onComplete, onError }: RealTimeProgressProps) {
  const [progress, setProgress] = useState<ProgressData | null>(null)
  const [isConnected, setIsConnected] = useState(false)
  const [, setWs] = useState<WebSocket | null>(null)

  useEffect(() => {
    if (!taskId) return

    // Connect to WebSocket for real-time updates
    const websocket = new WebSocket(`ws://localhost:8000/ws/progress/${taskId}`)
    
    websocket.onopen = () => {
      setIsConnected(true)
      console.log(`Connected to progress tracking for task ${taskId}`)
    }
    
    websocket.onmessage = (event) => {
      try {
        const data: ProgressData = JSON.parse(event.data)
        setProgress(data)
        
        // Check if generation is complete
        if (data.overall_progress === 100) {
          onComplete?.(taskId)
        }
      } catch (error) {
        console.error('Error parsing progress data:', error)
      }
    }
    
    websocket.onerror = (error) => {
      console.error('WebSocket error:', error)
      onError?.('Connection error')
    }
    
    websocket.onclose = () => {
      setIsConnected(false)
      console.log('WebSocket connection closed')
    }
    
    setWs(websocket)
    
    return () => {
      websocket.close()
    }
  }, [taskId, onComplete, onError])

  if (!progress) {
    return (
      <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
        <div className="flex items-center justify-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
          <span className="ml-3 text-gray-600">Connecting to progress tracker...</span>
        </div>
      </div>
    )
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'completed':
        return (
          <div className="w-6 h-6 bg-green-500 rounded-full flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
            </svg>
          </div>
        )
      case 'active':
        return (
          <div className="w-6 h-6 bg-blue-500 rounded-full flex items-center justify-center">
            <div className="w-3 h-3 bg-white rounded-full animate-pulse"></div>
          </div>
        )
      case 'failed':
        return (
          <div className="w-6 h-6 bg-red-500 rounded-full flex items-center justify-center">
            <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </div>
        )
      default:
        return (
          <div className="w-6 h-6 bg-gray-300 rounded-full flex items-center justify-center">
            <div className="w-3 h-3 bg-gray-500 rounded-full"></div>
          </div>
        )
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'completed': return 'text-green-600'
      case 'active': return 'text-blue-600'
      case 'failed': return 'text-red-600'
      default: return 'text-gray-500'
    }
  }

  return (
    <div className="bg-white rounded-xl shadow-lg border border-gray-200 overflow-hidden">
      {/* Header */}
      <div className="bg-gradient-to-r from-blue-600 to-purple-600 p-6 text-white">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-xl font-semibold">Video Generation Progress</h3>
            <p className="text-blue-100 text-sm">Task ID: {taskId}</p>
          </div>
          <div className="text-right">
            <div className="text-3xl font-bold">{progress.overall_progress}%</div>
            <div className="flex items-center gap-3 text-sm">
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`}></div>
              <span>{isConnected ? 'Connected' : 'Disconnected'}</span>
              {typeof progress.elapsed_seconds === 'number' && (
                <span className="opacity-90">Elapsed: {formatTime(progress.elapsed_seconds)}</span>
              )}
              {typeof progress.eta_seconds === 'number' && progress.eta_seconds !== null && (
                <span className="opacity-90">ETA: {formatTime(progress.eta_seconds)}</span>
              )}
            </div>
          </div>
        </div>
        
        {/* Overall Progress Bar */}
        <div className="mt-4">
          <div className="w-full bg-blue-800/30 rounded-full h-3">
            <div 
              className="bg-white h-3 rounded-full transition-all duration-500 ease-out"
              style={{ width: `${progress.overall_progress}%` }}
            ></div>
          </div>
        </div>
      </div>

      {/* Current Step Highlight */}
      {progress.current_step && (
        <div className="bg-blue-50 border-b border-blue-200 p-4">
          <div className="flex items-center">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-600 mr-3"></div>
            <div>
              <div className="font-medium text-blue-900">{progress.current_step.title}</div>
              <div className="text-sm text-blue-600">{progress.current_step.details}</div>
            </div>
          </div>
          {progress.current_step.progress > 0 && (
            <div className="mt-2">
              <div className="w-full bg-blue-200 rounded-full h-2">
                <div 
                  className="bg-blue-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress.current_step.progress}%` }}
                ></div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Steps List */}
      <div className="p-6">
        <h4 className="font-semibold text-gray-900 mb-4">Generation Steps</h4>
        
        <div className="space-y-3">
          {progress.steps.map((step) => (
            <div key={step.id} className="flex items-start">
              <div className="flex-shrink-0 mr-4">
                {getStatusIcon(step.status)}
              </div>
              
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <h5 className={`font-medium ${getStatusColor(step.status)}`}>
                    {step.title}
                  </h5>
                  {step.status === 'completed' && step.timestamp && (
                    <span className="text-xs text-gray-500">
                      {new Date(step.timestamp).toLocaleTimeString()}
                    </span>
                  )}
                </div>
                
                {step.details && (
                  <p className="text-sm text-gray-600 mt-1">{step.details}</p>
                )}
                
                {step.status === 'active' && step.progress > 0 && (
                  <div className="mt-2">
                    <div className="flex items-center justify-between text-xs text-gray-500 mb-1">
                      <span>Progress</span>
                      <span>{step.progress}%</span>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-1.5">
                      <div 
                        className="bg-blue-600 h-1.5 rounded-full transition-all duration-300"
                        style={{ width: `${step.progress}%` }}
                      ></div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Completed Tasks Summary */}
      <div className="bg-gray-50 border-t border-gray-200 p-4">
        <h5 className="font-medium text-gray-900 mb-2">Completed Tasks</h5>
        <div className="space-y-1">
          {progress.steps
            .filter(step => step.status === 'completed')
            .map(step => (
              <div key={step.id} className="flex items-center text-sm text-green-600">
                <svg className="w-4 h-4 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                {step.title}
              </div>
            ))}
        </div>
        
        {progress.overall_progress === 100 && (
          <div className="mt-4 p-3 bg-green-100 border border-green-200 rounded-lg">
            <div className="flex items-center">
              <svg className="w-5 h-5 text-green-600 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <div>
                <div className="font-medium text-green-800">Generation Complete!</div>
                <div className="text-sm text-green-600">Your video is ready for download.</div>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

function formatTime(totalSeconds: number): string {
  const s = Math.max(0, Math.floor(totalSeconds))
  const h = Math.floor(s / 3600)
  const m = Math.floor((s % 3600) / 60)
  const sec = s % 60
  if (h > 0) {
    return `${h}:${m.toString().padStart(2, '0')}:${sec.toString().padStart(2, '0')}`
  }
  return `${m}:${sec.toString().padStart(2, '0')}`
}

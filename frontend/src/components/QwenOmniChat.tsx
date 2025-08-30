import { useState, useRef } from 'react'

interface QwenMessage {
  role: 'user' | 'assistant'
  content: string
  image?: string
  audio?: string
  timestamp: Date
}

interface QwenOmniChatProps {
  onScriptGenerated?: (script: string) => void
}

export default function QwenOmniChat({ onScriptGenerated }: QwenOmniChatProps) {
  const [messages, setMessages] = useState<QwenMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [selectedImage, setSelectedImage] = useState<File | null>(null)
  const [selectedAudio, setSelectedAudio] = useState<File | null>(null)
  
  const imageInputRef = useRef<HTMLInputElement>(null)
  const audioInputRef = useRef<HTMLInputElement>(null)

  const sendMessage = async () => {
    if (!input.trim() && !selectedImage && !selectedAudio) return

    const userMessage: QwenMessage = {
      role: 'user',
      content: input,
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setIsLoading(true)

    try {
      const formData = new FormData()
      if (input) formData.append('text', input)
      if (selectedImage) formData.append('image', selectedImage)
      if (selectedAudio) formData.append('audio', selectedAudio)

      const response = await fetch('/api/qwen/chat', {
        method: 'POST',
        body: formData
      })

      const data = await response.json()

      if (data.success) {
        const assistantMessage: QwenMessage = {
          role: 'assistant',
          content: data.response,
          timestamp: new Date()
        }
        setMessages(prev => [...prev, assistantMessage])

        // If this looks like a script, offer to use it
        if (data.response.length > 50 && onScriptGenerated) {
          onScriptGenerated(data.response)
        }
      }
    } catch (error) {
      console.error('Chat error:', error)
    }

    setInput('')
    setSelectedImage(null)
    setSelectedAudio(null)
    setIsLoading(false)
  }

  const generateAvatarScript = async (topic: string) => {
    setIsLoading(true)
    try {
      const response = await fetch('/api/qwen/generate-script', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ 
          topic,
          style: 'professional',
          duration: 30
        })
      })

      const data = await response.json()
      if (data.success && onScriptGenerated) {
        onScriptGenerated(data.script)
      }
    } catch (error) {
      console.error('Script generation error:', error)
    }
    setIsLoading(false)
  }

  return (
    <div className="bg-white rounded-xl shadow-lg p-6 border border-gray-200">
      <div className="flex items-center mb-6">
        <div className="w-10 h-10 bg-gradient-to-br from-indigo-500 to-purple-600 rounded-xl flex items-center justify-center mr-3 shadow-lg">
          <svg className="w-6 h-6 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z" />
          </svg>
        </div>
        <h2 className="text-xl font-semibold text-gray-900">Qwen2.5-Omni AI Chat</h2>
      </div>

      {/* Chat Messages */}
      <div className="h-64 overflow-y-auto mb-4 border border-gray-200 rounded-lg p-3 bg-gray-50">
        {messages.length === 0 ? (
          <div className="text-center text-gray-500 mt-8">
            <p className="text-sm">Start a conversation with Qwen2.5-Omni</p>
            <p className="text-xs mt-1">Supports text, images, and audio</p>
          </div>
        ) : (
          messages.map((msg, idx) => (
            <div key={idx} className={`mb-3 ${msg.role === 'user' ? 'text-right' : 'text-left'}`}>
              <div className={`inline-block max-w-xs p-2 rounded-lg text-sm ${
                msg.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-white border border-gray-200'
              }`}>
                {msg.content}
              </div>
            </div>
          ))
        )}
        {isLoading && (
          <div className="text-left mb-3">
            <div className="inline-block bg-white border border-gray-200 p-2 rounded-lg">
              <div className="flex items-center space-x-1">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce"></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.1s'}}></div>
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{animationDelay: '0.2s'}}></div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Media Attachments */}
      {(selectedImage || selectedAudio) && (
        <div className="mb-4 p-3 bg-blue-50 border border-blue-200 rounded-lg">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              {selectedImage && (
                <div className="flex items-center text-sm text-blue-700">
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                  </svg>
                  {selectedImage.name}
                </div>
              )}
              {selectedAudio && (
                <div className="flex items-center text-sm text-blue-700">
                  <svg className="w-4 h-4 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
                  </svg>
                  {selectedAudio.name}
                </div>
              )}
            </div>
            <button
              onClick={() => {
                setSelectedImage(null)
                setSelectedAudio(null)
              }}
              className="text-blue-600 hover:text-blue-800"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            </button>
          </div>
        </div>
      )}

      {/* Input Area */}
      <div className="space-y-3">
        <div className="flex space-x-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && sendMessage()}
            placeholder="Type your message..."
            className="flex-1 p-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 text-sm"
          />
          <button
            onClick={() => imageInputRef.current?.click()}
            className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            title="Add image"
          >
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
            </svg>
          </button>
          <button
            onClick={() => audioInputRef.current?.click()}
            className="p-2 border border-gray-300 rounded-lg hover:bg-gray-50"
            title="Add audio"
          >
            <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 11a7 7 0 01-7 7m0 0a7 7 0 01-7-7m7 7v4m0 0H8m4 0h4m-4-8a3 3 0 01-3-3V5a3 3 0 116 0v6a3 3 0 01-3 3z" />
            </svg>
          </button>
          <button
            onClick={sendMessage}
            disabled={isLoading || (!input.trim() && !selectedImage && !selectedAudio)}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 text-sm font-medium"
          >
            Send
          </button>
        </div>

        {/* Quick Actions */}
        <div className="flex flex-wrap gap-2">
          <button
            onClick={() => generateAvatarScript('product presentation')}
            className="px-3 py-1 bg-purple-100 text-purple-700 rounded-full text-xs hover:bg-purple-200"
          >
            Product Demo Script
          </button>
          <button
            onClick={() => generateAvatarScript('educational content')}
            className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-xs hover:bg-green-200"
          >
            Educational Script
          </button>
          <button
            onClick={() => generateAvatarScript('company announcement')}
            className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-xs hover:bg-blue-200"
          >
            Announcement Script
          </button>
        </div>
      </div>

      {/* Hidden file inputs */}
      <input
        ref={imageInputRef}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => setSelectedImage(e.target.files?.[0] || null)}
      />
      <input
        ref={audioInputRef}
        type="file"
        accept="audio/*"
        className="hidden"
        onChange={(e) => setSelectedAudio(e.target.files?.[0] || null)}
      />
    </div>
  )
}
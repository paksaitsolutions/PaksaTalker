const API_BASE_URL = '/api/v1'

const handleResponse = async (response: Response) => {
  if (!response.ok) {
    const text = await response.text()
    let errorMessage = `HTTP ${response.status}: ${response.statusText}`
    
    try {
      const errorData = JSON.parse(text)
      errorMessage = errorData.detail || errorData.message || errorMessage
    } catch {
      // If not JSON, use the text as error message
      errorMessage = text || errorMessage
    }
    
    throw new Error(errorMessage)
  }
  
  const contentType = response.headers.get('content-type')
  if (contentType && contentType.includes('application/json')) {
    return response.json()
  }
  
  throw new Error('Expected JSON response but got: ' + contentType)
}

export const generateVideo = async (formData: FormData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate/video`, {
      method: 'POST',
      body: formData
    })
    const data = await handleResponse(response)
    return { ...data, ok: response.ok }
  } catch (error) {
    console.error('Generate video error:', error)
    throw error
  }
}

export const getTaskStatus = async (taskId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/status/${taskId}`)
    return handleResponse(response)
  } catch (error) {
    console.error('Get task status error:', error)
    throw error
  }
}

export const generateText = async (prompt: string) => {
  try {
    const formData = new FormData()
    formData.append('prompt', prompt)
    formData.append('max_length', '100')
    formData.append('temperature', '0.7')
    
    const response = await fetch(`${API_BASE_URL}/generate/text`, {
      method: 'POST',
      body: formData
    })
    return handleResponse(response)
  } catch (error) {
    console.error('Generate text error:', error)
    throw error
  }
}

export const registerSpeaker = async (formData: FormData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/speakers/register`, {
      method: 'POST',
      body: formData
    })
    return handleResponse(response)
  } catch (error) {
    console.error('Register speaker error:', error)
    throw error
  }
}

export const generateGestures = async (text: string, emotion: string) => {
  try {
    const formData = new FormData()
    formData.append('text', text)
    formData.append('emotion', emotion)
    formData.append('intensity', '0.7')
    formData.append('duration', '5.0')
    
    const response = await fetch(`${API_BASE_URL}/generate-gestures`, {
      method: 'POST',
      body: formData
    })
    return handleResponse(response)
  } catch (error) {
    console.error('Generate gestures error:', error)
    throw error
  }
}

export const generateVideoFromPrompt = async (formData: FormData) => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate/video-from-prompt`, {
      method: 'POST',
      body: formData
    })
    const data = await handleResponse(response)
    return { ...data, ok: response.ok }
  } catch (error) {
    console.error('Generate video from prompt error:', error)
    throw error
  }
}

export const getExpressionCapabilities = async () => {
  const res = await fetch(`${API_BASE_URL}/expressions/capabilities`)
  return handleResponse(res)
}

export const estimateExpressions = async (image: File, engine: string = 'auto') => {
  const fd = new FormData()
  fd.append('image', image)
  fd.append('engine', engine)
  const res = await fetch(`${API_BASE_URL}/expressions/estimate`, { method: 'POST', body: fd })
  return handleResponse(res)
}

// Style Preset API functions
export const createStylePreset = async (presetData: any) => {
  const formData = new FormData()
  Object.keys(presetData).forEach(key => {
    formData.append(key, presetData[key])
  })
  
  const response = await fetch(`${API_BASE_URL}/style-presets`, {
    method: 'POST',
    body: formData
  })
  return response.json()
}

export const listStylePresets = async () => {
  const response = await fetch(`${API_BASE_URL}/style-presets`)
  return response.json()
}

export const interpolatePresets = async (preset1Id: string, preset2Id: string, ratio: number) => {
  const formData = new FormData()
  formData.append('preset1_id', preset1Id)
  formData.append('preset2_id', preset2Id)
  formData.append('ratio', ratio.toString())
  
  const response = await fetch(`${API_BASE_URL}/style-presets/interpolate`, {
    method: 'POST',
    body: formData
  })
  return response.json()
}

export const createCulturalVariants = async (presetId: string) => {
  const response = await fetch(`${API_BASE_URL}/style-presets/${presetId}/cultural-variants`, {
    method: 'POST'
  })
  return response.json()
}

export const suggestStylePresets = async (params: { prompt?: string; emotion?: string; cultural_context?: string; formality?: number }) => {
  const form = new FormData()
  if (params.prompt) form.append('prompt', params.prompt)
  if (params.emotion) form.append('emotion', params.emotion)
  if (params.cultural_context) form.append('cultural_context', params.cultural_context)
  if (typeof params.formality === 'number') form.append('formality', String(params.formality))
  const response = await fetch(`${API_BASE_URL}/style-presets/suggest`, { method: 'POST', body: form })
  return response.json()
}

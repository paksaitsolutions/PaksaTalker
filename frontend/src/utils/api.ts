const API_BASE_URL = '/api/v1'

export const generateVideo = async (formData: FormData) => {
  const response = await fetch(`${API_BASE_URL}/generate/video`, {
    method: 'POST',
    body: formData
  })
  const data = await response.json()
  return { ...data, ok: response.ok }
}

export const getTaskStatus = async (taskId: string) => {
  try {
    const response = await axios.get(`${API_BASE_URL}/status/${taskId}`)
    return response.data
  } catch (error) {
    throw error
  }
}

export const generateText = async (prompt: string) => {
  const formData = new FormData()
  formData.append('prompt', prompt)
  formData.append('max_length', '100')
  formData.append('temperature', '0.7')
  
  const response = await fetch(`${API_BASE_URL}/generate/text`, {
    method: 'POST',
    body: formData
  })
  return response.json()
}

export const registerSpeaker = async (formData: FormData) => {
  try {
    const response = await axios.post(`${API_BASE_URL}/speakers/register`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    })
    return response.data
  } catch (error) {
    throw error
  }
}

export const generateGestures = async (text: string, emotion: string) => {
  const formData = new FormData()
  formData.append('text', text)
  formData.append('emotion', emotion)
  formData.append('intensity', '0.7')
  formData.append('duration', '5.0')
  
  const response = await fetch(`${API_BASE_URL}/generate-gestures`, {
    method: 'POST',
    body: formData
  })
  return response.json()
}

export const generateVideoFromPrompt = async (formData: FormData) => {
  const response = await fetch(`${API_BASE_URL}/generate/video-from-prompt`, {
    method: 'POST',
    body: formData
  })
  const data = await response.json()
  return { ...data, ok: response.ok }
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
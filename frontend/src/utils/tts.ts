// Real Text-to-Speech implementation
export const generateTTS = async (text: string, voice: string): Promise<Blob> => {
  const response = await fetch('/api/tts', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text, voice })
  })
  
  if (!response.ok) throw new Error('TTS generation failed')
  return await response.blob()
}

export const validateVoice = async (voiceId: string): Promise<boolean> => {
  const response = await fetch(`/api/voices/validate/${voiceId}`)
  const data = await response.json()
  return data.supported
}
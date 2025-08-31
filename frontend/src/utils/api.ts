import type { StyleSettings, VideoGenerationRequest } from '../types/styles';

const API_BASE_URL = 'http://localhost:8000/api/v1';

export type { StyleSettings, VideoGenerationRequest };

// Generate video with style customization
export const generateVideo = async (formData: FormData): Promise<{ success: boolean; task_id?: string; error?: string }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/generate/video`, {
      method: 'POST',
      body: formData,
    });
    return await response.json();
  } catch (error) {
    console.error('Error generating video:', error);
    return { success: false, error: 'Failed to connect to server' };
  }
};

// Get task status
export const getTaskStatus = async (taskId: string) => {
  try {
    const response = await fetch(`${API_BASE_URL}/status/${taskId}`);
    return await response.json();
  } catch (error) {
    console.error('Error fetching task status:', error);
    return { success: false, error: 'Failed to fetch status' };
  }
};

// Get available style presets
export const getStylePresets = async (): Promise<{ [key: string]: StyleSettings }> => {
  try {
    const response = await fetch(`${API_BASE_URL}/style-presets`);
    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }
    return await response.json();
  } catch (error) {
    console.error('Error fetching style presets:', error);
    return {};
  }
};

// Save custom style preset
export const saveStylePreset = async (preset: StyleSettings & { name: string }) => {
  try {
    const response = await fetch(`${API_BASE_URL}/styles/presets`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(preset),
    });
    return await response.json();
  } catch (error) {
    console.error('Error saving style preset:', error);
    return { success: false, error: 'Failed to save preset' };
  }
};

// Apply style transfer
export const applyStyleTransfer = async (videoId: string, styleSettings: StyleSettings) => {
  try {
    const response = await fetch(`${API_BASE_URL}/style/transfer`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ videoId, ...styleSettings }),
    });
    return await response.json();
  } catch (error) {
    console.error('Error applying style transfer:', error);
    return { success: false, error: 'Failed to apply style transfer' };
  }
};

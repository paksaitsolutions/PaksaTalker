// Type definitions for the API module

declare module '../utils/api' {
  export interface StyleSettings {
    styleType: 'professional' | 'casual' | 'friendly' | 'enthusiastic';
    intensity: number;
    culturalInfluence?: string;
    mannerisms?: string[];
  }

  export interface VideoGenerationRequest {
    image: File;
    audio?: File;
    text?: string;
    styleSettings?: StyleSettings;
  }

  export const generateVideo: (formData: FormData) => Promise<{ success: boolean; task_id?: string; error?: string }>;
  export const getTaskStatus: (taskId: string) => Promise<any>;
  export const getStylePresets: () => Promise<{ [key: string]: StyleSettings }>;
  export const saveStylePreset: (preset: StyleSettings & { name: string }) => Promise<any>;
  export const applyStyleTransfer: (videoId: string, styleSettings: StyleSettings) => Promise<any>;
}

export type StyleType = 'professional' | 'casual' | 'friendly' | 'enthusiastic';

export interface StyleSettings {
  styleType: StyleType;
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

export interface StylePreset {
  id: string;
  name: string;
  description: string;
  settings: StyleSettings;
  thumbnail: string;
}

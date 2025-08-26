export interface Speaker {
  id: string;
  name: string;
  avatarUrl?: string;
  description?: string;
  defaultStyle?: string;
}

export interface EmotionConfig {
  type: 'neutral' | 'happy' | 'sad' | 'angry' | 'surprised' | 'disgusted' | 'fearful';
  intensity: number; // 0-1
  secondaryEmotion: EmotionConfig['type'] | null;
  blendAmount: number; // 0-1
  transitionDuration?: number; // in seconds
}

export interface AnimationConfig {
  speakerId: string;
  style: string;
  resolution: '720p' | '1080p' | '1440p' | '2160p';
  frameRate: number;
  quality: number; // 0-1
  includeAudio: boolean;
  emotionConfig: EmotionConfig;
}

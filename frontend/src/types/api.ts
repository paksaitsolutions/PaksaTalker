export interface ApiError {
  message: string;
  statusCode: number;
  error?: string;
  details?: Record<string, any>;
}

export interface ApiResponse<T = any> {
  data: T;
  status: number;
  statusText: string;
}

export interface AnimationRequest {
  audio: File;
  emotion: string;
  intensity: number;
  config?: AnimationConfig;
}

export interface AnimationConfig {
  speakerId?: string;
  style?: string;
  resolution?: '720p' | '1080p' | '1440p' | '2160p';
  frameRate?: number;
  quality?: number;
  includeAudio?: boolean;
  emotionConfig?: EmotionConfig;
}

export interface EmotionConfig {
  type: string;
  intensity: number;
  secondaryEmotion?: string | null;
  blendAmount?: number;
  transitionDuration?: number;
}

export interface Speaker {
  id: string;
  name: string;
  avatarUrl?: string;
  description?: string;
  defaultStyle?: string;
  styles?: string[];
}

export interface AnimationJob {
  id: string;
  status: 'pending' | 'processing' | 'completed' | 'failed';
  progress: number;
  createdAt: string;
  updatedAt: string;
  resultUrl?: string;
  error?: string;
  metadata?: {
    duration?: number;
    resolution?: string;
    frameRate?: number;
    fileSize?: number;
  };
}

export interface AnimationHistoryItem {
  id: string;
  jobId: string;
  userId: string;
  status: string;
  createdAt: string;
  updatedAt: string;
  resultUrl?: string;
  thumbnailUrl?: string;
  metadata: {
    emotion: string;
    intensity: number;
    duration: number;
    resolution: string;
    style?: string;
  };
}

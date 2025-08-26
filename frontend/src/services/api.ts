import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosResponse, AxiosError } from 'axios';

// Get environment variables with fallbacks
const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

// Define types for the API responses
interface ApiResponse<T = unknown> {
  data: T;
  message?: string;
  success: boolean;
}

// Define types for the API error
interface ApiError {
  message: string;
  status?: number;
  data?: unknown;
}

const api: AxiosInstance = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Request interceptor for API calls
api.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    if (typeof window !== 'undefined' && typeof localStorage !== 'undefined') {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers = config.headers || {};
        config.headers.Authorization = `Bearer ${token}`;
      }
    }
    return config;
  },
  (error: unknown) => {
    return Promise.reject(error);
  }
);

// Response interceptor for API calls
api.interceptors.response.use(
  (response: AxiosResponse) => response,
  (error: unknown) => {
    const axiosError = error as AxiosError;
    if (axiosError.response?.status === 401 && 
        typeof window !== 'undefined' && 
        typeof localStorage !== 'undefined') {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const authAPI = {
  login: (email: string, password: string): Promise<ApiResponse<{ token: string }>> =>
    api.post('/auth/login', { email, password }),
  
  register: (userData: {
    email: string;
    password: string;
    name: string;
    // Add other registration fields as needed
  }): Promise<ApiResponse<{ id: string; email: string; name: string }>> =>
    api.post('/auth/register', userData),
    
  refreshToken: (): Promise<ApiResponse<{ token: string }>> =>
    api.post('/auth/refresh-token'),
};

interface Video {
  id: string;
  title: string;
  status: 'processing' | 'completed' | 'failed';
  url?: string;
  createdAt: string;
  updatedAt: string;
}

export const videoAPI = {
  generate: (formData: globalThis.FormData): Promise<ApiResponse<{ taskId: string }>> =>
    api.post('/generate/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }),
    
  getStatus: (taskId: string): Promise<ApiResponse<Video>> =>
    api.get(`/status/${taskId}`),
    
  listVideos: (): Promise<ApiResponse<Video[]>> =>
    api.get('/videos'),
    
  getVideo: (id: string): Promise<globalThis.Blob> =>
    api.get(`/videos/${id}`, { responseType: 'blob' }),
};

interface Settings {
  theme: 'light' | 'dark' | 'system';
  notifications: boolean;
  language: string;
  // Add other settings as needed
}

export const settingsAPI = {
  getSettings: (): Promise<ApiResponse<Settings>> =>
    api.get('/settings'),
    
  updateSettings: (settings: Partial<Settings>): Promise<ApiResponse<Settings>> =>
    api.put('/settings', settings),
};

// Export the API instance
export default api;

// Export types for use in components
export type { ApiResponse, ApiError, Video, Settings };

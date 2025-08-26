import axios, { AxiosError, AxiosInstance, AxiosRequestConfig, AxiosResponse } from 'axios';
import { toast } from 'react-hot-toast';
import {
  AnimationConfig,
  AnimationHistoryItem,
  AnimationJob,
  AnimationRequest,
  ApiError,
  ApiResponse,
  EmotionConfig,
  Speaker,
} from '@/types/api';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

// Create axios instance with default config
const createApiClient = (): AxiosInstance => {
  const instance = axios.create({
    baseURL: API_URL,
    timeout: 30000, // 30 seconds
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/json',
    },
    withCredentials: true,
  });

  // Request interceptor
  instance.interceptors.request.use(
    (config) => {
      const token = localStorage.getItem('token');
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  // Response interceptor
  instance.interceptors.response.use(
    (response) => response,
    (error: AxiosError<ApiError>) => {
      const errorMessage = error.response?.data?.message || error.message;
      const status = error.response?.status;

      // Handle specific status codes
      if (status === 401) {
        // Handle unauthorized access
        localStorage.removeItem('token');
        window.location.href = '/login';
      } else if (status === 403) {
        toast.error('You do not have permission to perform this action');
      } else if (status === 404) {
        toast.error('The requested resource was not found');
      } else if (status === 500) {
        toast.error('An unexpected server error occurred');
      } else if (errorMessage) {
        toast.error(errorMessage);
      }

      return Promise.reject(error);
    }
  );

  return instance;
};

const api = createApiClient();

/**
 * Animation API Service
 * Handles all API calls related to animation generation and management
 */
const animationApi = {
  /**
   * Submit a new animation job
   */
  async createAnimationJob(
    data: AnimationRequest
  ): Promise<ApiResponse<{ jobId: string }>> {
    const formData = new FormData();
    formData.append('audio', data.audio);
    formData.append('emotion', data.emotion);
    formData.append('intensity', data.intensity.toString());

    if (data.config) {
      formData.append('config', JSON.stringify(data.config));
    }

    const response = await api.post<{ jobId: string }>(
      '/api/v1/animations',
      formData,
      {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      }
    );

    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Get the status of an animation job
   */
  async getJobStatus(jobId: string): Promise<ApiResponse<AnimationJob>> {
    const response = await api.get<AnimationJob>(`/api/v1/animations/${jobId}`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Cancel a running animation job
   */
  async cancelJob(jobId: string): Promise<ApiResponse<void>> {
    const response = await api.post(`/api/v1/animations/${jobId}/cancel`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Get animation job result (video)
   */
  async getJobResult(
    jobId: string,
    onDownloadProgress?: (progressEvent: ProgressEvent) => void
  ): Promise<Blob> {
    const response = await api.get(`/api/v1/animations/${jobId}/result`, {
      responseType: 'blob',
      onDownloadProgress,
    });
    return response.data;
  },

  /**
   * Get list of available speakers
   */
  async getSpeakers(): Promise<ApiResponse<Speaker[]>> {
    const response = await api.get<Speaker[]>('/api/v1/speakers');
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Get animation history for the current user
   */
  async getHistory(): Promise<ApiResponse<AnimationHistoryItem[]>> {
    const response = await api.get<AnimationHistoryItem[]>('/api/v1/animations/history');
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Delete an animation from history
   */
  async deleteAnimation(animationId: string): Promise<ApiResponse<void>> {
    const response = await api.delete(`/api/v1/animations/history/${animationId}`);
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Get system status and limits
   */
  async getSystemStatus(): Promise<ApiResponse<{
    maxFileSize: number;
    maxDuration: number;
    supportedFormats: string[];
    maxConcurrentJobs: number;
    currentJobs: number;
  }>> {
    const response = await api.get('/api/v1/system/status');
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },

  /**
   * Get default animation settings
   */
  async getDefaultSettings(): Promise<ApiResponse<AnimationConfig>> {
    const response = await api.get<AnimationConfig>('/api/v1/settings/defaults');
    return {
      data: response.data,
      status: response.status,
      statusText: response.statusText,
    };
  },
};

export default animationApi;

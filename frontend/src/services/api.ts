import axios from 'axios';

// Get environment variables with fallbacks
const API_BASE_URL = process.env.VITE_API_URL || '/api/v1';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
  withCredentials: true,
});

// Request interceptor for API calls
api.interceptors.request.use(
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

// Response interceptor for API calls
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized access
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API endpoints
export const authAPI = {
  login: (email: string, password: string) => 
    api.post('/auth/login', { email, password }),
  register: (userData: any) => 
    api.post('/auth/register', userData),
  refreshToken: () => 
    api.post('/auth/refresh-token'),
};

export const videoAPI = {
  generate: (formData: FormData) => 
    api.post('/generate/video', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    }),
  getStatus: (taskId: string) => 
    api.get(`/status/${taskId}`),
  listVideos: () => 
    api.get('/videos'),
  getVideo: (id: string) => 
    api.get(`/videos/${id}`, { responseType: 'blob' }),
};

export const settingsAPI = {
  getSettings: () => 
    api.get('/settings'),
  updateSettings: (settings: any) => 
    api.put('/settings', settings),
};

// Export the API instance
export default api;

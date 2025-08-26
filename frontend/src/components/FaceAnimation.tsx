import React, { useState, useRef, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import animationApi, { AnimationRequest } from '../services/animationApi';
import { Speaker, EmotionConfig, AnimationJob } from '../types/api';
import { CheckCircleIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';

interface FaceAnimationProps {
  onAnimationStart?: () => void;
  onAnimationEnd?: () => void;
  onError?: (error: Error) => void;
  defaultSpeaker?: string;
  availableSpeakers?: Speaker[];
  onSpeakerChange?: (speakerId: string) => void;
}

const FaceAnimation: React.FC<FaceAnimationProps> = ({
  onAnimationStart,
  onAnimationEnd,
  onError,
  defaultSpeaker = 'default',
  availableSpeakers = [],
  onSpeakerChange,
}) => {
  const navigate = useNavigate();
  const [isLoading, setIsLoading] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [progress, setProgress] = useState(0);
  const [jobStatus, setJobStatus] = useState<AnimationJob | null>(null);
  const [availableSpeakers, setAvailableSpeakers] = useState<Speaker[]>([]);
  
  // Animation state
  const [emotion, setEmotion] = useState<EmotionConfig>({
    type: 'neutral',
    intensity: 0.5,
    secondaryEmotion: null,
    blendAmount: 0,
  });
  
  const [selectedSpeaker, setSelectedSpeaker] = useState<string>(defaultSpeaker);
  const [speakerStyle, setSpeakerStyle] = useState<string>('default');
  const [outputSettings, setOutputSettings] = useState({
    resolution: '1080p' as const,
    frameRate: 30,
    quality: 0.9,
    includeAudio: true,
  });
  
  // Polling for job status
  const pollInterval = useRef<NodeJS.Timeout>();
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const emotions = [
    { value: 'neutral', label: 'Neutral' },
    { value: 'happy', label: 'Happy' },
    { value: 'sad', label: 'Sad' },
    { value: 'angry', label: 'Angry' },
    { value: 'surprised', label: 'Surprised' },
    { value: 'disgusted', label: 'Disgusted' },
    { value: 'fearful', label: 'Fearful' },
  ];

  const speakerStyles = [
    { value: 'default', label: 'Default' },
    { value: 'professional', label: 'Professional' },
    { value: 'casual', label: 'Casual' },
    { value: 'enthusiastic', label: 'Enthusiastic' },
  ];

  const resolutions = [
    { value: '720p', label: '720p' },
    { value: '1080p', label: '1080p' },
    { value: '1440p', label: '1440p (2K)' },
    { value: '2160p', label: '2160p (4K)' },
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setAudioFile(e.target.files[0]);
      setPreviewUrl(URL.createObjectURL(e.target.files[0]));
    }
  };

  const handleEmotionChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setEmotion(prev => ({
      ...prev,
      type: e.target.value as EmotionConfig['type']
    }));
  };

  const handleIntensityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmotion(prev => ({
      ...prev,
      intensity: parseFloat(e.target.value)
    }));
  };

  const handleBlendChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setEmotion(prev => ({
      ...prev,
      blendAmount: parseFloat(e.target.value)
    }));
  };

  const handleSpeakerChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const speakerId = e.target.value;
    const speaker = availableSpeakers.find(s => s.id === speakerId);
    if (speaker) {
      setSelectedSpeaker(speakerId);
      setSpeakerStyle(speaker.defaultStyle || 'default');
      onSpeakerChange?.(speakerId);
    }
  };

  const handleStyleChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSpeakerStyle(e.target.value);
  };

  const handleOutputSettingChange = (e: React.ChangeEvent<HTMLSelectElement | HTMLInputElement>) => {
    const { name, value, type } = e.target;
    setOutputSettings(prev => ({
      ...prev,
      [name]: type === 'number' || type === 'range' 
        ? parseFloat(value) 
        : value
    }));
  };

  // Fetch available speakers on component mount
  useEffect(() => {
    const loadSpeakers = async () => {
      try {
        const response = await animationApi.getSpeakers();
        setAvailableSpeakers(response.data);
        if (response.data.length > 0) {
          setSelectedSpeaker(response.data[0].id);
        }
      } catch (error) {
        console.error('Failed to load speakers:', error);
        toast.error('Failed to load speakers');
      } finally {
        setIsLoading(false);
      }
    };

    loadSpeakers();

    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
    };
  }, []);

  // Poll job status
  const pollJobStatus = useCallback(async (jobId: string) => {
    if (pollInterval.current) {
      clearInterval(pollInterval.current);
    }

    pollInterval.current = setInterval(async () => {
      try {
        const response = await animationApi.getJobStatus(jobId);
        setJobStatus(response.data);
        setProgress(response.data.progress);

        if (['completed', 'failed'].includes(response.data.status)) {
          clearInterval(pollInterval.current);
          
          if (response.data.status === 'completed' && response.data.resultUrl) {
            onAnimationEnd?.();
            toast.success('Animation generated successfully!');
            
            // Load and display the result
            const videoBlob = await animationApi.getJobResult(jobId, (progressEvent) => {
              const percentCompleted = Math.round(
                (progressEvent.loaded * 100) / (progressEvent.total || 1)
              );
              setProgress(percentCompleted);
            });
            
            const videoUrl = URL.createObjectURL(videoBlob);
            if (videoRef.current) {
              videoRef.current.src = videoUrl;
              await videoRef.current.play();
            }
          } else if (response.data.status === 'failed') {
            onError?.(new Error(response.data.error || 'Animation generation failed'));
          }
          
          setIsSubmitting(false);
        }
      } catch (error) {
        console.error('Error polling job status:', error);
        clearInterval(pollInterval.current);
        setIsSubmitting(false);
        onError?.(error as Error);
      }
    }, 2000); // Poll every 2 seconds

    return () => {
      if (pollInterval.current) {
        clearInterval(pollInterval.current);
      }
    };
  }, [onAnimationEnd, onError]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!audioFile) {
      onError?.(new Error('Please select an audio file'));
      return;
    }

    try {
      setIsSubmitting(true);
      setProgress(0);
      onAnimationStart?.();

      const request: AnimationRequest = {
        audio: audioFile,
        emotion: emotion.type,
        intensity: emotion.intensity,
        config: {
          speakerId: selectedSpeaker,
          style: speakerStyle,
          resolution: outputSettings.resolution,
          frameRate: outputSettings.frameRate,
          quality: outputSettings.quality,
          includeAudio: outputSettings.includeAudio,
          emotionConfig: {
            ...emotion,
            transitionDuration: 0.5, // seconds
          },
        },
      };

      // Submit the job
      const response = await animationApi.createAnimationJob(request);
      
      // Start polling for job status
      await pollJobStatus(response.data.jobId);
      
    } catch (error) {
      console.error('Animation error:', error);
      setIsSubmitting(false);
      onError?.(error as Error);
    }
  };

  useEffect(() => {
    return () => {
      if (previewUrl) {
        URL.revokeObjectURL(previewUrl);
      }
    };
  }, [previewUrl]);

  return (
    <div className="space-y-6">
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h2 className="text-xl font-semibold mb-4">Face Animation</h2>
        
        <form onSubmit={handleSubmit} className="space-y-4">
          {/* Audio Input */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Audio File
            </label>
            <input
              type="file"
              accept="audio/*"
              onChange={handleFileChange}
              ref={fileInputRef}
              className="block w-full text-sm text-gray-500
                file:mr-4 file:py-2 file:px-4
                file:rounded-md file:border-0
                file:text-sm file:font-semibold
                file:bg-blue-50 file:text-blue-700
                hover:file:bg-blue-100"
            />
            {previewUrl && (
              <audio
                src={previewUrl}
                controls
                className="mt-2 w-full"
              />
            )}
          </div>

          {/* Speaker Selection */}
          {availableSpeakers.length > 0 && (
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Speaker
              </label>
              <select
                value={selectedSpeaker}
                onChange={handleSpeakerChange}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                disabled={isAnimating}
              >
                {availableSpeakers.map((speaker) => (
                  <option key={speaker.id} value={speaker.id}>
                    {speaker.name}
                  </option>
                ))}
              </select>
            </div>
          )}

          {/* Speaker Style */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Speaking Style
            </label>
            <select
              value={speakerStyle}
              onChange={handleStyleChange}
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              disabled={isAnimating}
            >
              {speakerStyles.map((style) => (
                <option key={style.value} value={style.value}>
                  {style.label}
                </option>
              ))}
            </select>
          </div>

          {/* Emotion Controls */}
          <div className="space-y-4 p-4 border rounded-lg">
            <h3 className="text-sm font-medium text-gray-700">Emotion Controls</h3>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Primary Emotion
              </label>
              <select
                value={emotion.type}
                onChange={handleEmotionChange}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                disabled={isAnimating}
              >
                {emotions.map((e) => (
                  <option key={e.value} value={e.value}>
                    {e.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Emotion Intensity: {emotion.intensity.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={emotion.intensity}
                onChange={handleIntensityChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                disabled={isAnimating}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Emotion Blend: {emotion.blendAmount.toFixed(1)}
              </label>
              <input
                type="range"
                min="0"
                max="1"
                step="0.1"
                value={emotion.blendAmount}
                onChange={handleBlendChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                disabled={isAnimating}
              />
              <p className="mt-1 text-xs text-gray-500">
                Blends between primary and secondary emotions
              </p>
            </div>
          </div>

          {/* Output Settings */}
          <div className="space-y-4 p-4 border rounded-lg">
            <h3 className="text-sm font-medium text-gray-700">Output Settings</h3>
            
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Resolution
              </label>
              <select
                name="resolution"
                value={outputSettings.resolution}
                onChange={handleOutputSettingChange}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
                disabled={isAnimating}
              >
                {resolutions.map((res) => (
                  <option key={res.value} value={res.value}>
                    {res.label}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Frame Rate: {outputSettings.frameRate} FPS
              </label>
              <input
                type="range"
                name="frameRate"
                min="24"
                max="60"
                step="1"
                value={outputSettings.frameRate}
                onChange={handleOutputSettingChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                disabled={isAnimating}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Quality: {Math.round(outputSettings.quality * 100)}%
              </label>
              <input
                type="range"
                name="quality"
                min="0.5"
                max="1"
                step="0.1"
                value={outputSettings.quality}
                onChange={handleOutputSettingChange}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                disabled={isAnimating}
              />
            </div>

            <div className="flex items-center">
              <input
                type="checkbox"
                name="includeAudio"
                checked={outputSettings.includeAudio}
                onChange={(e) => 
                  setOutputSettings(prev => ({
                    ...prev,
                    includeAudio: e.target.checked
                  }))
                }
                className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                disabled={isAnimating}
              />
              <label className="ml-2 block text-sm text-gray-700">
                Include Audio
              </label>
            </div>
          </div>

          {/* Submit Button and Status */}
          <div className="pt-2 space-y-2">
            <button
              type="submit"
              disabled={isSubmitting || !audioFile || isLoading}
              className={`w-full flex justify-center py-2 px-4 border border-transparent rounded-md shadow-sm text-sm font-medium text-white 
                ${
                  isSubmitting || !audioFile || isLoading
                    ? 'bg-gray-400 cursor-not-allowed'
                    : 'bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500'
                }`}
            >
              {isSubmitting ? (
                <span className="flex items-center">
                  <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Processing...
                </span>
              ) : 'Generate Animation'}
            </button>

            {isSubmitting && (
              <div className="mt-2">
                <div className="flex justify-between text-xs text-gray-600 mb-1">
                  <span>Generating animation...</span>
                  <span>{progress}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full transition-all duration-300 ease-in-out"
                    style={{ width: `${progress}%` }}
                  ></div>
                </div>
                {jobStatus?.status === 'processing' && (
                  <p className="mt-1 text-xs text-gray-500">
                    This may take a few minutes. You can navigate away and we'll notify you when it's ready.
                  </p>
                )}
              </div>
            )}

            {jobStatus?.status === 'completed' && (
              <div className="mt-2 p-2 bg-green-50 rounded-md flex items-start">
                <CheckCircleIcon className="h-5 w-5 text-green-500 mr-2 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-green-800">Animation ready!</p>
                  <p className="text-xs text-green-700">Your animation has been generated successfully.</p>
                </div>
              </div>
            )}

            {jobStatus?.status === 'failed' && (
              <div className="mt-2 p-2 bg-red-50 rounded-md flex items-start">
                <ExclamationCircleIcon className="h-5 w-5 text-red-500 mr-2 mt-0.5 flex-shrink-0" />
                <div>
                  <p className="text-sm font-medium text-red-800">Generation failed</p>
                  <p className="text-xs text-red-700">{jobStatus.error || 'An error occurred during generation'}</p>
                  <button
                    onClick={() => window.location.reload()}
                    className="mt-1 text-xs text-red-600 hover:text-red-800 font-medium"
                  >
                    Try again
                  </button>
                </div>
              </div>
            )}
          </div>
        </form>

        {/* Progress Bar */}
        {isAnimating && (
          <div className="mt-4">
            <div className="flex justify-between text-sm text-gray-600 mb-1">
              <span>Generating animation...</span>
              <span>{progress}%</span>
            </div>
            <div className="w-full bg-gray-200 rounded-full h-2.5">
              <div
                className="bg-blue-600 h-2.5 rounded-full"
                style={{ width: `${progress}%` }}
              ></div>
            </div>
          </div>
        )}
      </div>

      {/* Video Preview */}
      <div className="bg-white p-6 rounded-lg shadow-md">
        <h3 className="text-lg font-medium mb-2">Animation Preview</h3>
        <div className="aspect-w-16 aspect-h-9 bg-black rounded-lg overflow-hidden">
          <video
            ref={videoRef}
            controls
            className="w-full h-full object-cover"
            onEnded={() => onAnimationEnd?.()}
          >
            Your browser does not support the video tag.
          </video>
        </div>
      </div>
    </div>
  );
};

export default FaceAnimation;

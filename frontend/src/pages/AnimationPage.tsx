import React, { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { toast } from 'react-hot-toast';
import FaceAnimation from '../components/FaceAnimation';
import AnimationHistory from '../components/AnimationHistory';
import { Speaker } from '../types/api';
import { Loader2, AlertCircle, History, Video } from 'lucide-react';
import { Button } from '../components/ui/button';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs';
import { useAnimationHistory } from '../contexts/AnimationHistoryContext';
import ErrorBoundary from '../components/ErrorBoundary';

const AnimationPage: React.FC = () => {
  const navigate = useNavigate();
  const { history, addToHistory } = useAnimationHistory();
  const [isLoading, setIsLoading] = useState(true);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [isInitialized, setIsInitialized] = useState(false);

  // Load available speakers
  const loadSpeakers = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      // TODO: Replace with actual API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const defaultSpeakers: Speaker[] = [
        { 
          id: 'speaker1', 
          name: 'Default Speaker', 
          description: 'Standard voice model',
          gender: 'neutral',
          defaultStyle: 'default',
          availableStyles: ['default', 'excited', 'calm'],
          previewUrl: '/speakers/speaker1-preview.jpg'
        },
        { 
          id: 'speaker2', 
          name: 'Professional', 
          description: 'Formal business voice',
          gender: 'male',
          defaultStyle: 'professional',
          availableStyles: ['professional', 'presentation', 'interview'],
          previewUrl: '/speakers/speaker2-preview.jpg'
        },
        { 
          id: 'speaker3', 
          name: 'Friendly', 
          description: 'Casual and approachable',
          gender: 'female',
          defaultStyle: 'friendly',
          availableStyles: ['friendly', 'happy', 'warm'],
          previewUrl: '/speakers/speaker3-preview.jpg'
        },
      ];
      
      setSpeakers(defaultSpeakers);
      setIsInitialized(true);
    } catch (err) {
      console.error('Failed to load speakers:', err);
      setError('Failed to load speakers. Please refresh the page or try again later.');
    } finally {
      setIsLoading(false);
    }
  }, []);

  // Initialize the page
  useEffect(() => {
    if (!isInitialized) {
      loadSpeakers();
    }
  }, [isInitialized, loadSpeakers]);

  const [selectedSpeaker, setSelectedSpeaker] = useState<string>('default');
  const [speakerStyle, setSpeakerStyle] = useState<string>('default');

  const handleAnimationStart = () => {
    toast.loading('Starting animation generation...', { 
      id: 'animation-start',
      duration: 3000
    });
  };

  const handleAnimationEnd = (videoUrl?: string) => {
    toast.success('Animation generated successfully!', { 
      id: 'animation-complete',
      duration: 5000
    });
    
    // Add to history if we have a video URL
    if (videoUrl) {
      // Create a mock job object for the history
      const job: any = {
        jobId: `job-${Date.now()}`,
        status: 'completed',
        progress: 100,
        resultUrl: videoUrl,
        config: {
          speakerId: selectedSpeaker,
          style: speakerStyle,
          // Add other config as needed
        },
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString()
      };
      
      addToHistory(job, videoUrl, `Animation ${new Date().toLocaleString()}`);
    }
  };

  const handleError = (error: Error) => {
    console.error('Animation error:', error);
    toast.error(`Error: ${error.message}`, { 
      id: 'animation-error',
      duration: 5000
    });
  };

  const handleSpeakerChange = (speakerId: string) => {
    setSelectedSpeaker(speakerId);
    // Reset to default style when changing speakers
    const speaker = speakers.find(s => s.id === speakerId);
    if (speaker) {
      setSpeakerStyle(speaker.defaultStyle || 'default');
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[70vh] p-6">
        <Loader2 className="h-12 w-12 text-blue-500 animate-spin mb-4" />
        <h2 className="text-xl font-semibold text-gray-800 mb-2">Loading Animation Studio</h2>
        <p className="text-gray-600 text-center max-w-md">
          Getting everything ready for you. This should just take a moment...
        </p>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="max-w-4xl mx-auto p-6">
        <div className="bg-red-50 rounded-lg border border-red-200 p-6">
          <div className="flex">
            <div className="flex-shrink-0">
              <AlertCircle className="h-5 w-5 text-red-400" />
            </div>
            <div className="ml-3">
              <h3 className="text-sm font-medium text-red-800">
                Oops! Something went wrong
              </h3>
              <div className="mt-2 text-sm text-red-700">
                <p>{error}</p>
              </div>
              <div className="mt-4">
                <Button
                  variant="outline"
                  onClick={loadSpeakers}
                  className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
                >
                  <RefreshCw className="-ml-1 mr-2 h-4 w-4" />
                  Try Again
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
      <div className="text-center mb-10">
        <h1 className="text-4xl font-bold text-gray-900 sm:text-5xl">
          AI Face Animation Studio
        </h1>
        <p className="mt-4 max-w-3xl mx-auto text-xl text-gray-600">
          Bring your content to life with realistic AI-powered facial animations
        </p>
      </div>

      <Tabs defaultValue="create" className="space-y-6">
        <TabsList className="grid w-full max-w-md grid-cols-2 mx-auto">
          <TabsTrigger value="create" className="flex items-center gap-2">
            <Video className="h-4 w-4" />
            <span>Create Animation</span>
          </TabsTrigger>
          <TabsTrigger value="history" className="flex items-center gap-2">
            <History className="h-4 w-4" />
            <span>My Animations</span>
            {history.length > 0 && (
              <span className="inline-flex items-center justify-center px-2 py-0.5 text-xs font-medium rounded-full bg-blue-100 text-blue-800 ml-1">
                {history.length}
              </span>
            )}
          </TabsTrigger>
        </TabsList>

        <TabsContent value="create" className="space-y-6">
          <div className="bg-white shadow-xl rounded-2xl overflow-hidden transition-all duration-300 hover:shadow-2xl">
            <div className="px-6 py-8 sm:p-8">
              <div className="mb-8">
                <h2 className="text-2xl font-semibold text-gray-900">Create New Animation</h2>
                <p className="mt-1 text-gray-500">
                  Upload an audio file and customize the animation settings below
                </p>
              </div>
              
              <ErrorBoundary
                fallback={
                  <div className="p-4 bg-red-50 text-red-700 rounded-lg">
                    <h3 className="font-medium">Error loading animation editor</h3>
                    <p className="text-sm">Please refresh the page or try again later.</p>
                  </div>
                }
              >
                <FaceAnimation
                  onAnimationStart={handleAnimationStart}
                  onAnimationEnd={(videoUrl) => {
                    handleAnimationEnd();
                    // Switch to history tab after successful animation
                    document.querySelector('button[data-value="history"]')?.scrollIntoView({
                      behavior: 'smooth'
                    });
                    setTimeout(() => {
                      document.querySelector('button[data-value="history"]')?.click();
                    }, 100);
                  }}
                  onError={handleError}
                  onSpeakerChange={handleSpeakerChange}
                  availableSpeakers={speakers}
                />
              </ErrorBoundary>
            </div>
            
            <div className="bg-gray-50 px-6 py-4 border-t border-gray-200">
              <div className="flex flex-col sm:flex-row justify-between items-center">
                <p className="text-sm text-gray-500 mb-2 sm:mb-0">
                  Need help? Check out our guides or contact support.
                </p>
                <div className="flex space-x-3">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => window.open('/docs/animation-guide', '_blank')}
                  >
                    <span className="hidden sm:inline">View </span>Guide
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => navigate('/contact')}
                  >
                    Contact Support
                  </Button>
                </div>
              </div>
            </div>
          </div>
          
          <div className="bg-blue-50 rounded-2xl p-6">
            <h3 className="text-lg font-medium text-blue-800 mb-3">Tips for Best Results</h3>
            <ul className="grid gap-2 text-blue-700">
              <li className="flex items-start">
                <svg className="h-5 w-5 text-blue-500 mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Use high-quality audio with clear speech for best lip-sync results</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-blue-500 mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>Keep background noise to a minimum</span>
              </li>
              <li className="flex items-start">
                <svg className="h-5 w-5 text-blue-500 mr-2 mt-0.5 flex-shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                </svg>
                <span>For longer videos, consider breaking them into smaller segments</span>
              </li>
            </ul>
          </div>
        </TabsContent>

        <TabsContent value="history" className="pt-4">
          <ErrorBoundary
            fallback={
              <div className="p-4 bg-red-50 text-red-700 rounded-lg">
                <h3 className="font-medium">Error loading animation history</h3>
                <p className="text-sm">Please refresh the page or try again later.</p>
              </div>
            }
          >
            <AnimationHistory />
          </ErrorBoundary>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AnimationPage;

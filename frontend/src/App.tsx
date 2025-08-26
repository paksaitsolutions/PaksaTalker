import { Routes, Route, Navigate } from 'react-router-dom';
import { lazy, Suspense } from 'react';
import { Toaster } from 'react-hot-toast';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactQueryDevtools } from '@tanstack/react-query-devtools';
import Navbar from '@/components/Navbar';
import Footer from '@/components/Footer';
import LoadingSpinner from '@/components/ui/LoadingSpinner';
import ErrorBoundary from '@/components/ErrorBoundary';
import { AnimationHistoryProvider } from '@/contexts/AnimationHistoryContext';

// Lazy load pages for better performance
const Home = lazy(() => import('@/pages/Home'));
const Pricing = lazy(() => import('@/pages/Pricing'));
const Features = lazy(() => import('@/pages/Features'));
const Contact = lazy(() => import('@/pages/Contact'));
const Dashboard = lazy(() => import('@/pages/Dashboard'));
const AnimationPage = lazy(() => import('@/pages/AnimationPage'));

// Create a client for React Query
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
});

// Error boundary fallback component
const ErrorFallback = ({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) => (
  <div className="min-h-screen flex items-center justify-center p-4">
    <div className="max-w-md w-full p-6 bg-white rounded-lg shadow-lg text-center">
      <h2 className="text-2xl font-bold text-red-600 mb-4">Something went wrong</h2>
      <p className="text-gray-700 mb-6">{error.message}</p>
      <button
        onClick={resetErrorBoundary}
        className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
      >
        Try again
      </button>
    </div>
  </div>
);

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary FallbackComponent={ErrorFallback}>
        <AnimationHistoryProvider>
          <div className="min-h-screen flex flex-col bg-gray-50">
            <Navbar />
            <main className="flex-grow">
              <Suspense 
                fallback={
                  <div className="flex items-center justify-center min-h-[60vh]">
                    <LoadingSpinner size="lg" />
                  </div>
                }
              >
                <Routes>
                  <Route path="/" element={<Home />} />
                  <Route path="/pricing" element={<Pricing />} />
                  <Route path="/features" element={<Features />} />
                  <Route path="/contact" element={<Contact />} />
                  <Route path="/dashboard/*" element={
                    <ErrorBoundary 
                      fallback={
                        <div className="p-6">
                          <h2 className="text-xl font-semibold mb-4">Error loading dashboard</h2>
                          <p className="text-gray-600">There was a problem loading the dashboard. Please try refreshing the page.</p>
                        </div>
                      }
                    >
                      <Dashboard />
                    </ErrorBoundary>
                  } 
                  />
                  <Route 
                    path="/animation" 
                    element={
                      <ErrorBoundary 
                        fallback={
                          <div className="p-6">
                            <h2 className="text-xl font-semibold mb-4">Animation Studio Error</h2>
                            <p className="text-gray-600">There was a problem loading the animation studio. Please try again later.</p>
                          </div>
                        }
                      >
                        <AnimationPage />
                      </ErrorBoundary>
                    } 
                  />
                  <Route path="*" element={<Navigate to="/" replace />} />
                </Routes>
              </Suspense>
            </main>
            <Footer />
            <Toaster 
              position="bottom-right" 
              toastOptions={{
                duration: 5000,
                style: {
                  background: '#ffffff',
                  color: '#1f2937',
                  boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06)',
                  borderRadius: '0.5rem',
                  padding: '1rem',
                  fontSize: '0.875rem',
                },
                success: {
                  iconTheme: {
                    primary: '#10B981',
                    secondary: '#ffffff',
                  },
                },
                error: {
                  iconTheme: {
                    primary: '#EF4444',
                    secondary: '#ffffff',
                  },
                },
                loading: {
                  iconTheme: {
                    primary: '#3B82F6',
                    secondary: '#EFF6FF',
                  },
                },
              }}
            />
            {import.meta.env.DEV && <ReactQueryDevtools initialIsOpen={false} />}
          </div>
        </AnimationHistoryProvider>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

export default App;

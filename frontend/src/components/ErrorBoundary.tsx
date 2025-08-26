import React, { Component, ErrorInfo, ReactNode } from 'react';
import { AlertTriangle, RefreshCw } from 'lucide-react';
import { Button } from '@/components/ui/button';

interface Props {
  children: ReactNode;
  fallback?: ReactNode;
  onError?: (error: Error, errorInfo: ErrorInfo) => void;
}

interface State {
  hasError: boolean;
  error: Error | null;
  errorInfo: ErrorInfo | null;
}

class ErrorBoundary extends Component<Props, State> {
  public state: State = {
    hasError: false,
    error: null,
    errorInfo: null,
  };

  public static getDerivedStateFromError(error: Error): State {
    // Update state so the next render will show the fallback UI
    return { hasError: true, error, errorInfo: null };
  }

  public componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    // Log the error to an error reporting service
    console.error('ErrorBoundary caught an error:', error, errorInfo);
    
    // Call the onError handler if provided
    if (this.props.onError) {
      this.props.onError(error, errorInfo);
    }

    this.setState({ error, errorInfo });
  }

  private handleReset = () => {
    this.setState({ hasError: false, error: null, errorInfo: null });
  };

  public render() {
    if (this.state.hasError) {
      // Custom fallback UI
      if (this.props.fallback) {
        return this.props.fallback;
      }

      // Default fallback UI
      return (
        <div className="flex flex-col items-center justify-center p-6 bg-red-50 rounded-lg border border-red-200">
          <div className="flex items-center mb-4">
            <AlertTriangle className="h-8 w-8 text-red-500 mr-2" />
            <h2 className="text-xl font-semibold text-red-800">
              Something went wrong
            </h2>
          </div>
          
          <div className="mb-4 text-sm text-red-700 bg-white p-3 rounded border border-red-100 max-w-lg">
            <p className="font-medium mb-1">{this.state.error?.name || 'Error'}</p>
            <p className="mb-2">{this.state.error?.message || 'An unexpected error occurred'}</p>
            
            {process.env.NODE_ENV === 'development' && this.state.errorInfo && (
              <details className="mt-2 text-xs">
                <summary className="cursor-pointer text-red-600 hover:text-red-800">
                  Show error details
                </summary>
                <pre className="mt-2 p-2 bg-gray-50 rounded overflow-auto max-h-40">
                  {this.state.errorInfo.componentStack}
                </pre>
              </details>
            )}
          </div>
          
          <Button
            onClick={this.handleReset}
            variant="outline"
            className="flex items-center text-sm"
          >
            <RefreshCw className="mr-2 h-4 w-4" />
            Try again
          </Button>
        </div>
      );
    }

    return this.props.children;
  }
}

export default ErrorBoundary;

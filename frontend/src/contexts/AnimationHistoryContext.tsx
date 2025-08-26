import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { AnimationJob } from '../types/api';
import { animationStorage, AnimationHistoryItem } from '../utils/storage';

type AnimationHistoryContextType = {
  history: AnimationHistoryItem[];
  favorites: AnimationHistoryItem[];
  isLoading: boolean;
  error: Error | null;
  addToHistory: (job: AnimationJob, previewUrl?: string, name?: string) => void;
  removeFromHistory: (id: string) => void;
  toggleFavorite: (id: string) => void;
  updateHistoryItem: (id: string, updates: Partial<AnimationHistoryItem>) => void;
  clearHistory: () => void;
  getHistoryItem: (id: string) => AnimationHistoryItem | undefined;
};

const AnimationHistoryContext = createContext<AnimationHistoryContextType | undefined>(undefined);

export const AnimationHistoryProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [history, setHistory] = useState<AnimationHistoryItem[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);

  // Load history from storage when the component mounts
  useEffect(() => {
    const loadHistory = () => {
      try {
        setIsLoading(true);
        const storedHistory = animationStorage.getHistoryItems();
        setHistory(storedHistory);
        setError(null);
      } catch (err) {
        console.error('Failed to load animation history:', err);
        setError(err instanceof Error ? err : new Error('Failed to load history'));
      } finally {
        setIsLoading(false);
      }
    };

    loadHistory();

    // Listen for storage events to sync across tabs
    const handleStorageChange = (e: StorageEvent) => {
      if (e.key === 'paksatalker_animation_history' || !e.key) {
        loadHistory();
      }
    };

    window.addEventListener('storage', handleStorageChange);
    return () => window.removeEventListener('storage', handleStorageChange);
  }, []);

  const addToHistory = useCallback((job: AnimationJob, previewUrl?: string, name?: string) => {
    try {
      animationStorage.addToHistory(job, previewUrl, name);
      setHistory(prev => {
        const newItem: AnimationHistoryItem = {
          id: job.jobId,
          job,
          timestamp: Date.now(),
          previewUrl,
          name: name || `Animation ${new Date().toLocaleString()}`,
          isFavorite: false
        };
        return [newItem, ...prev];
      });
    } catch (err) {
      console.error('Failed to add to history:', err);
      throw err;
    }
  }, []);

  const removeFromHistory = useCallback((id: string) => {
    try {
      animationStorage.removeFromHistory(id);
      setHistory(prev => prev.filter(item => item.id !== id));
    } catch (err) {
      console.error('Failed to remove from history:', err);
      throw err;
    }
  }, []);

  const toggleFavorite = useCallback((id: string) => {
    try {
      animationStorage.toggleFavorite(id);
      setHistory(prev => 
        prev.map(item => 
          item.id === id 
            ? { ...item, isFavorite: !item.isFavorite } 
            : item
        )
      );
    } catch (err) {
      console.error('Failed to toggle favorite:', err);
      throw err;
    }
  }, []);

  const updateHistoryItem = useCallback((id: string, updates: Partial<AnimationHistoryItem>) => {
    try {
      animationStorage.updateHistoryItem(id, updates);
      setHistory(prev => 
        prev.map(item => 
          item.id === id 
            ? { ...item, ...updates } 
            : item
        )
      );
    } catch (err) {
      console.error('Failed to update history item:', err);
      throw err;
    }
  }, []);

  const clearHistory = useCallback(() => {
    try {
      animationStorage.clearHistory();
      setHistory([]);
    } catch (err) {
      console.error('Failed to clear history:', err);
      throw err;
    }
  }, []);

  const getHistoryItem = useCallback((id: string) => {
    return history.find(item => item.id === id);
  }, [history]);

  // Filter favorites from history
  const favorites = history.filter(item => item.isFavorite);

  const value = {
    history,
    favorites,
    isLoading,
    error,
    addToHistory,
    removeFromHistory,
    toggleFavorite,
    updateHistoryItem,
    clearHistory,
    getHistoryItem,
  };

  return (
    <AnimationHistoryContext.Provider value={value}>
      {children}
    </AnimationHistoryContext.Provider>
  );
};

export const useAnimationHistory = (): AnimationHistoryContextType => {
  const context = useContext(AnimationHistoryContext);
  if (context === undefined) {
    throw new Error('useAnimationHistory must be used within an AnimationHistoryProvider');
  }
  return context;
};

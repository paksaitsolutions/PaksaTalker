import { AnimationJob } from '../types/api';

const STORAGE_KEY = 'paksatalker_animation_history';
const MAX_HISTORY_ITEMS = 50; // Limit the number of history items to store

export interface AnimationHistoryItem {
  id: string;
  job: AnimationJob;
  timestamp: number;
  previewUrl?: string;
  name?: string;
  isFavorite?: boolean;
}

class AnimationStorage {
  private getHistory(): AnimationHistoryItem[] {
    if (typeof window === 'undefined') return [];
    
    try {
      const stored = localStorage.getItem(STORAGE_KEY);
      return stored ? JSON.parse(stored) : [];
    } catch (error) {
      console.error('Failed to parse animation history from localStorage', error);
      return [];
    }
  }

  private saveHistory(history: AnimationHistoryItem[]): void {
    if (typeof window === 'undefined') return;
    
    try {
      // Keep only the most recent items
      const recentHistory = history
        .sort((a, b) => b.timestamp - a.timestamp)
        .slice(0, MAX_HISTORY_ITEMS);
      
      localStorage.setItem(STORAGE_KEY, JSON.stringify(recentHistory));
    } catch (error) {
      console.error('Failed to save animation history to localStorage', error);
    }
  }

  public addToHistory(job: AnimationJob, previewUrl?: string, name?: string): void {
    const history = this.getHistory();
    const existingIndex = history.findIndex(item => item.id === job.jobId);
    
    const historyItem: AnimationHistoryItem = {
      id: job.jobId,
      job,
      timestamp: Date.now(),
      previewUrl,
      name: name || `Animation ${new Date().toLocaleString()}`,
      isFavorite: false
    };

    if (existingIndex >= 0) {
      // Update existing item
      history[existingIndex] = {
        ...history[existingIndex],
        ...historyItem,
        isFavorite: history[existingIndex].isFavorite // Preserve favorite status
      };
    } else {
      // Add new item
      history.unshift(historyItem);
    }

    this.saveHistory(history);
  }

  public getHistoryItems(): AnimationHistoryItem[] {
    return this.getHistory();
  }

  public getHistoryItem(id: string): AnimationHistoryItem | undefined {
    return this.getHistory().find(item => item.id === id);
  }

  public toggleFavorite(id: string): void {
    const history = this.getHistory();
    const itemIndex = history.findIndex(item => item.id === id);
    
    if (itemIndex >= 0) {
      history[itemIndex].isFavorite = !history[itemIndex].isFavorite;
      this.saveHistory(history);
    }
  }

  public updateHistoryItem(id: string, updates: Partial<AnimationHistoryItem>): void {
    const history = this.getHistory();
    const itemIndex = history.findIndex(item => item.id === id);
    
    if (itemIndex >= 0) {
      history[itemIndex] = {
        ...history[itemIndex],
        ...updates,
        id, // Prevent changing the ID
        timestamp: updates.timestamp || history[itemIndex].timestamp
      };
      
      this.saveHistory(history);
    }
  }

  public removeFromHistory(id: string): void {
    const history = this.getHistory().filter(item => item.id !== id);
    this.saveHistory(history);
  }

  public clearHistory(): void {
    if (typeof window === 'undefined') return;
    localStorage.removeItem(STORAGE_KEY);
  }
}

export const animationStorage = new AnimationStorage();

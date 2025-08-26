import React, { useState, useMemo } from 'react';
import { useAnimationHistory } from '../contexts/AnimationHistoryContext';
import { formatDistanceToNow } from 'date-fns';
import { Play, Trash2, Star, Download, Clock, Filter } from 'lucide-react';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Input } from './ui/input';
import { motion, AnimatePresence } from 'framer-motion';

export const AnimationHistory: React.FC = () => {
  const { 
    history, 
    favorites, 
    removeFromHistory, 
    toggleFavorite, 
    updateHistoryItem,
    isLoading,
    error
  } = useAnimationHistory();
  
  const [searchQuery, setSearchQuery] = useState('');
  const [activeTab, setActiveTab] = useState<'all' | 'favorites'>('all');
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editName, setEditName] = useState('');

  const filteredHistory = useMemo(() => {
    const items = activeTab === 'favorites' ? favorites : history;
    
    if (!searchQuery.trim()) return items;
    
    const query = searchQuery.toLowerCase();
    return items.filter(item => 
      item.name?.toLowerCase().includes(query) ||
      item.job.status?.toLowerCase().includes(query) ||
      item.job.config?.speakerId?.toLowerCase().includes(query)
    );
  }, [history, favorites, activeTab, searchQuery]);

  const handleEditStart = (item: any) => {
    setEditingId(item.id);
    setEditName(item.name || '');
  };

  const handleEditSave = (id: string) => {
    if (editName.trim()) {
      updateHistoryItem(id, { name: editName.trim() });
    }
    setEditingId(null);
  };

  const handlePlayAnimation = (item: any) => {
    // TODO: Implement play animation in the viewer
    console.log('Play animation:', item);
  };

  const handleDownload = (item: any) => {
    if (item.job.resultUrl) {
      const link = document.createElement('a');
      link.href = item.job.resultUrl;
      link.download = `${item.name || 'animation'}.mp4`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center p-8">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500"></div>
        <span className="ml-3 text-gray-600">Loading history...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 text-red-700 rounded-lg">
        <p>Error loading animation history: {error.message}</p>
      </div>
    );
  }

  if (history.length === 0) {
    return (
      <div className="text-center p-8 text-gray-500">
        <Clock className="mx-auto h-12 w-12 text-gray-300 mb-4" />
        <h3 className="text-lg font-medium text-gray-900">No animations yet</h3>
        <p className="mt-1 text-sm text-gray-500">
          Your animation history will appear here once you create some.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row justify-between gap-4">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
            <Filter className="h-4 w-4 text-gray-400" />
          </div>
          <Input
            type="text"
            placeholder="Search animations..."
            className="pl-10"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>
        
        <div className="inline-flex rounded-md shadow-sm">
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium rounded-l-md ${
              activeTab === 'all' 
                ? 'bg-blue-600 text-white' 
                : 'bg-white text-gray-700 hover:bg-gray-50 border border-gray-300'
            }`}
            onClick={() => setActiveTab('all')}
          >
            All ({history.length})
          </button>
          <button
            type="button"
            className={`px-4 py-2 text-sm font-medium rounded-r-md ${
              activeTab === 'favorites'
                ? 'bg-yellow-500 text-white' 
                : 'bg-white text-gray-700 hover:bg-gray-50 border-t border-b border-r border-gray-300'
            }`}
            onClick={() => setActiveTab('favorites')}
          >
            Favorites ({favorites.length})
          </button>
        </div>
      </div>

      <div className="overflow-hidden bg-white shadow rounded-lg divide-y divide-gray-200">
        <AnimatePresence>
          {filteredHistory.length > 0 ? (
            <ul className="divide-y divide-gray-200">
              {filteredHistory.map((item) => (
                <motion.li 
                  key={item.id}
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, x: -10 }}
                  className="p-4 hover:bg-gray-50 transition-colors duration-150"
                >
                  <div className="flex items-center justify-between">
                    <div className="flex-1 min-w-0">
                      {editingId === item.id ? (
                        <div className="flex items-center space-x-2">
                          <Input
                            value={editName}
                            onChange={(e) => setEditName(e.target.value)}
                            className="h-8 w-full max-w-xs"
                            autoFocus
                          />
                          <Button 
                            size="sm" 
                            variant="outline" 
                            onClick={() => handleEditSave(item.id)}
                          >
                            Save
                          </Button>
                          <Button 
                            size="sm" 
                            variant="ghost" 
                            onClick={() => setEditingId(null)}
                          >
                            Cancel
                          </Button>
                        </div>
                      ) : (
                        <div className="flex items-center space-x-2">
                          <h3 
                            className="text-sm font-medium text-gray-900 truncate cursor-pointer hover:text-blue-600"
                            onClick={() => handleEditStart(item)}
                          >
                            {item.name}
                          </h3>
                          {item.isFavorite && (
                            <Star className="h-4 w-4 text-yellow-500 fill-current" />
                          )}
                        </div>
                      )}
                      <div className="mt-1 flex flex-wrap gap-2 text-xs text-gray-500">
                        <span>
                          {formatDistanceToNow(new Date(item.timestamp), { addSuffix: true })}
                        </span>
                        {item.job.status && (
                          <Badge variant={item.job.status === 'completed' ? 'success' : 'outline'}>
                            {item.job.status}
                          </Badge>
                        )}
                        {item.job.config?.speakerId && (
                          <Badge variant="secondary">
                            {item.job.config.speakerId}
                          </Badge>
                        )}
                      </div>
                    </div>
                    <div className="ml-4 flex-shrink-0 flex space-x-2">
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="text-gray-400 hover:text-blue-600"
                        onClick={() => toggleFavorite(item.id)}
                        title={item.isFavorite ? 'Remove from favorites' : 'Add to favorites'}
                      >
                        <Star 
                          className={`h-4 w-4 ${item.isFavorite ? 'text-yellow-500 fill-current' : ''}`} 
                        />
                      </Button>
                      {item.job.status === 'completed' && (
                        <>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="text-gray-400 hover:text-green-600"
                            onClick={() => handlePlayAnimation(item)}
                            title="Play animation"
                          >
                            <Play className="h-4 w-4" />
                          </Button>
                          <Button 
                            variant="ghost" 
                            size="icon" 
                            className="text-gray-400 hover:text-blue-600"
                            onClick={() => handleDownload(item)}
                            title="Download"
                          >
                            <Download className="h-4 w-4" />
                          </Button>
                        </>
                      )}
                      <Button 
                        variant="ghost" 
                        size="icon" 
                        className="text-gray-400 hover:text-red-600"
                        onClick={() => removeFromHistory(item.id)}
                        title="Delete"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </div>
                </motion.li>
              ))}
            </ul>
          ) : (
            <div className="p-8 text-center text-gray-500">
              <p>No animations found matching your search.</p>
            </div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default AnimationHistory;

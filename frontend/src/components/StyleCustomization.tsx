import { useState, useEffect } from 'react';
import type { StyleSettings } from '../types/styles';
import { StylePresets } from './StylePresets';

interface StyleCustomizationProps {
  onStyleChange: (settings: StyleSettings) => void;
  initialSettings?: StyleSettings;
}

const STYLE_TYPES = [
  { id: 'professional', label: 'Professional' },
  { id: 'casual', label: 'Casual' },
  { id: 'friendly', label: 'Friendly' },
  { id: 'enthusiastic', label: 'Enthusiastic' },
] as const;

const CULTURAL_INFLUENCES = [
  'None',
  'North American',
  'European',
  'East Asian',
  'South Asian',
  'Middle Eastern',
  'Latin American',
  'African',
];

const MANNERISMS = [
  'Nodding',
  'Hand gestures',
  'Head tilts',
  'Eyebrow raises',
  'Smiling',
  'Serious expression',
  'Minimal movement',
  'Expressive hands',
];

export const StyleCustomization: React.FC<StyleCustomizationProps> = ({
  onStyleChange,
  initialSettings = {
    styleType: 'professional',
    intensity: 5,
    culturalInfluence: 'None',
    mannerisms: [],
  },
}) => {
  const [settings, setSettings] = useState<StyleSettings>(initialSettings);
  const [showAdvanced, setShowAdvanced] = useState(false);

  useEffect(() => {
    onStyleChange(settings);
  }, [settings, onStyleChange]);

  const handleStyleTypeChange = (styleType: StyleSettings['styleType']) => {
    setSettings(prev => ({ ...prev, styleType }));
  };

  const handleIntensityChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSettings(prev => ({ ...prev, intensity: parseInt(e.target.value) }));
  };

  const handleCulturalInfluenceChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSettings(prev => ({
      ...prev,
      culturalInfluence: e.target.value === 'None' ? undefined : e.target.value,
    }));
  };

  const toggleMannerism = (mannerism: string) => {
    setSettings(prev => ({
      ...prev,
      mannerisms: prev.mannerisms?.includes(mannerism)
        ? prev.mannerisms.filter(m => m !== mannerism)
        : [...(prev.mannerisms || []), mannerism],
    }));
  };

  return (
    <div className="space-y-6 p-4 bg-white rounded-lg shadow-md border border-gray-100">
      <div className="border-b border-gray-200 pb-4">
        <h3 className="text-lg font-semibold text-gray-900">Style Customization</h3>
        <p className="text-sm text-gray-500 mt-1">Customize the speaking style of your avatar</p>
      </div>
      
      <StylePresets 
        onSelect={(settings) => {
          setSettings(prev => ({
            ...prev,
            ...settings,
            // Preserve any existing settings not overridden by the preset
          }));
        }} 
        activeStyleType={settings.styleType}
      />
      
      <div className="space-y-4 pt-2">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Style Type</label>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            {STYLE_TYPES.map(style => (
              <button
                key={style.id}
                type="button"
                onClick={() => handleStyleTypeChange(style.id as StyleSettings['styleType'])}
                className={`px-3 py-2 text-sm rounded-md transition-colors ${
                  settings.styleType === style.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {style.label}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Intensity: {settings.intensity}/10
          </label>
          <input
            type="range"
            min="1"
            max="10"
            value={settings.intensity}
            onChange={handleIntensityChange}
            className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
          />
        </div>

        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="text-sm text-blue-600 hover:text-blue-800 flex items-center"
        >
          {showAdvanced ? 'Hide Advanced' : 'Show Advanced Options'}
          <svg
            className={`ml-1 w-4 h-4 transition-transform ${showAdvanced ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
            xmlns="http://www.w3.org/2000/svg"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showAdvanced && (
          <div className="space-y-4 pt-2 border-t border-gray-200">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Cultural Influence</label>
              <select
                value={settings.culturalInfluence || 'None'}
                onChange={handleCulturalInfluenceChange}
                className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
              >
                {CULTURAL_INFLUENCES.map(influence => (
                  <option key={influence} value={influence}>
                    {influence}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Mannerisms</label>
              <div className="flex flex-wrap gap-2">
                {MANNERISMS.map((mannerism) => (
                  <button
                    key={mannerism}
                    type="button"
                    onClick={() => toggleMannerism(mannerism)}
                    className={`px-3 py-1 text-sm rounded-full transition-colors ${
                      settings.mannerisms?.includes(mannerism)
                        ? 'bg-blue-100 text-blue-800 border border-blue-300'
                        : 'bg-gray-100 text-gray-700 border border-gray-200 hover:bg-gray-200'
                    }`}
                  >
                    {mannerism}
                  </button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default StyleCustomization;

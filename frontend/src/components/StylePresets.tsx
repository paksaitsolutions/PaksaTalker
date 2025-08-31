import type { StyleSettings, StylePreset } from '../types/styles';

interface StylePresetsProps {
  onSelect: (settings: StyleSettings) => void;
  activeStyleType?: string;
}

const PRESETS: StylePreset[] = [
  {
    id: 'professional',
    name: 'Professional',
    description: 'Formal and business-appropriate style',
    settings: {
      styleType: 'professional',
      intensity: 5,
      culturalInfluence: undefined,
      mannerisms: ['Minimal movement', 'Serious expression'],
    },
    thumbnail: '👔',
  },
  {
    id: 'casual',
    name: 'Casual',
    description: 'Relaxed and friendly style',
    settings: {
      styleType: 'casual',
      intensity: 7,
      culturalInfluence: 'North American',
      mannerisms: ['Nodding', 'Smiling'],
    },
    thumbnail: '😊',
  },
  {
    id: 'enthusiastic',
    name: 'Enthusiastic',
    description: 'Energetic and engaging style',
    settings: {
      styleType: 'enthusiastic',
      intensity: 9,
      culturalInfluence: 'Latin American',
      mannerisms: ['Expressive hands', 'Eyebrow raises', 'Smiling'],
    },
    thumbnail: '🎉',
  },
  {
    id: 'academic',
    name: 'Academic',
    description: 'Thoughtful and precise style',
    settings: {
      styleType: 'professional',
      intensity: 4,
      culturalInfluence: 'European',
      mannerisms: ['Head tilts', 'Minimal movement'],
    },
    thumbnail: '🎓',
  },
];

export const StylePresets: React.FC<StylePresetsProps> = ({ onSelect, activeStyleType }) => {
  return (
    <div className="space-y-4">
      <h3 className="text-sm font-medium text-gray-700">Quick Presets</h3>
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {PRESETS.map((preset) => (
          <button
            key={preset.id}
            type="button"
            onClick={() => onSelect(preset.settings)}
            className={`flex flex-col items-center p-3 rounded-lg border-2 transition-all ${
              activeStyleType === preset.id
                ? 'border-blue-500 bg-blue-50'
                : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
            }`}
          >
            <span className="text-2xl mb-2">{preset.thumbnail}</span>
            <span className="font-medium text-sm">{preset.name}</span>
            <span className="text-xs text-gray-500 text-center mt-1">
              {preset.description}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
};

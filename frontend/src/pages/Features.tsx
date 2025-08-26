import React from 'react';
import {
  CheckCircleIcon,
  CloudArrowUpIcon,
  CodeBracketSquareIcon,
} from '@heroicons/react/20/solid';

const Features = () => {
  return (
    <div className="py-12 bg-white">
      <div className="max-w-xl mx-auto px-4 sm:px-6 lg:px-8 lg:max-w-7xl">
        <h2 className="text-3xl font-extrabold text-gray-900 sm:text-4xl text-center">
          Key Features
        </h2>
        <p className="mt-3 text-xl text-gray-500 sm:mt-5 text-center">
          Everything you need to create amazing videos
        </p>

        <div className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          <div className="bg-gray-50 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
            <CheckCircleIcon className="h-6 w-6 text-green-500 mb-2" aria-hidden="true" />
            <h3 className="text-lg font-medium text-gray-900">AI-Powered Video Generation</h3>
            <p className="mt-2 text-gray-500">
              Create stunning videos with AI in minutes, not hours.
            </p>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
            <CloudArrowUpIcon className="h-6 w-6 text-green-500 mb-2" aria-hidden="true" />
            <h3 className="text-lg font-medium text-gray-900">Realistic Avatars</h3>
            <p className="mt-2 text-gray-500">
              Choose from a variety of realistic AI avatars or create your own.
            </p>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
            <CodeBracketSquareIcon className="h-6 w-6 text-green-500 mb-2" aria-hidden="true" />
            <h3 className="text-lg font-medium text-gray-900">Multi-Language Support</h3>
            <p className="mt-2 text-gray-500">
              Generate videos in multiple languages with perfect lip-sync.
            </p>
          </div>

          <div className="bg-gray-50 p-6 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300">
            <CheckCircleIcon className="h-6 w-6 text-green-500 mb-2" aria-hidden="true" />
            <h3 className="text-lg font-medium text-gray-900">Customizable Templates</h3>
            <p className="mt-2 text-gray-500">Start with our professionally designed templates.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Features;

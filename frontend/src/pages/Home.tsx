import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRightIcon, CheckCircleIcon, PlayIcon } from '@heroicons/react/20/solid';

const features = [
  {
    name: 'AI-Powered Video Generation',
    description: 'Create stunning videos with AI in minutes, not hours.',
    icon: CheckCircleIcon,
  },
  {
    name: 'Realistic Avatars',
    description: 'Choose from a variety of realistic AI avatars or create your own.',
    icon: CheckCircleIcon,
  },
  {
    name: 'Multi-Language Support',
    description: 'Generate videos in multiple languages with perfect lip-sync.',
    icon: CheckCircleIcon,
  },
  {
    name: 'Customizable Templates',
    description: 'Start with our professionally designed templates.',
    icon: CheckCircleIcon,
  },
];

const Home: React.FC = () => {
  return (
    <div className="bg-gray-100">
      {/* Hero Section */}
      <div className="bg-white py-24 sm:py-32">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:mx-0">
            <h2 className="text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Bring your stories to life with AI-powered videos
            </h2>
            <p className="mt-2 text-lg leading-8 text-gray-600">
              PaksaTalker is the easiest way to create engaging videos with AI. Simply upload your image and script, and let our AI do the rest.
            </p>
          </div>
          <div className="mx-auto mt-10 max-w-2xl lg:mx-0 lg:max-w-none">
            <div className="grid grid-cols-1 gap-x-8 gap-y-6 sm:grid-cols-2 md:flex lg:gap-x-10">
              <Link
                to="/api/docs"
                className="rounded-md bg-blue-600 px-3.5 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-blue-600"
              >
                Generate a Video <ArrowRightIcon className="-mr-0.5 ml-2 h-5 w-5 inline" aria-hidden="true" />
              </Link>
              <Link to="/features" className="text-sm font-semibold leading-6 text-gray-900">
                Learn more <span aria-hidden="true">→</span>
              </Link>
            </div>
          </div>
        </div>
      </div>

      {/* Features Section */}
      <div className="bg-gray-50 py-12">
        <div className="mx-auto max-w-7xl px-6 lg:px-8">
          <div className="mx-auto max-w-2xl lg:text-center">
            <h2 className="text-lg font-semibold text-blue-600">Key Features</h2>
            <p className="mt-2 text-3xl font-bold tracking-tight text-gray-900 sm:text-4xl">
              Why Choose PaksaTalker?
            </p>
            <p className="mt-3 text-lg leading-8 text-gray-600">
              We offer a range of features designed to make video creation easy and effective.
            </p>
          </div>
          <div className="mt-16 grid grid-cols-1 gap-x-8 gap-y-12 sm:grid-cols-2 sm:gap-y-16 lg:grid-cols-3 lg:gap-x-10 lg:gap-y-16">
            {features.map((feature) => (
              <div key={feature.name} className="text-center md:text-left">
                <div className="mx-auto h-12 w-12 rounded-full bg-blue-100 lg:mx-0">
                  <feature.icon className="h-6 w-6 text-blue-600 mx-auto mt-3" aria-hidden="true" />
                </div>
                <h3 className="mt-4 text-lg font-medium text-gray-900">{feature.name}</h3>
                <p className="mt-2 text-base text-gray-500">{feature.description}</p>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Call to Action Section */}
      <div className="bg-blue-800 py-16">
        <div className="mx-auto max-w-7xl px-6 lg:px-8 text-center">
          <h2 className="text-3xl font-bold tracking-tight text-white sm:text-4xl">
            Ready to transform your content?
          </h2>
          <p className="mt-4 text-lg text-blue-100">
            Start creating amazing videos today with our free trial.
          </p>
          <Link
            to="/signup"
            className="mt-8 inline-flex items-center justify-center rounded-md bg-white px-5 py-3 text-base font-medium text-blue-700 hover:bg-blue-50"
          >
            Sign up for free
          </Link>
        </div>
      </div>
    </div>
  );
};

export default Home;

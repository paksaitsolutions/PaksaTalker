import React from 'react';
import { Link } from 'react-router-dom';
import { ArrowRight, CheckCircle, Play } from 'lucide-react';

const features = [
  {
    name: 'AI-Powered Video Generation',
    description: 'Create stunning videos with AI in minutes, not hours.',
    icon: CheckCircle,
  },
  {
    name: 'Realistic Avatars',
    description: 'Choose from a variety of realistic AI avatars or create your own.',
    icon: CheckCircle,
  },
  {
    name: 'Multi-Language Support',
    description: 'Generate videos in multiple languages with perfect lip-sync.',
    icon: CheckCircle,
  },
  {
    name: 'Customizable Templates',
    description: 'Start with our professionally designed templates.',
    icon: CheckCircle,
  },
];


const Home: React.FC = () => {
 return (
  <div className="bg-gray-100 min-h-screen">
   <div className="container mx-auto py-20 text-center">
    <h1 className="text-5xl font-bold text-gray-800 mb-8">
     Welcome to PaksaTalker
    </h1>
    <p className="text-xl text-gray-600 mb-12">
     Create amazing talking head videos with AI.
    </p>
    <Link
     to="/demo"
     className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-3 px-8 rounded"
    >
     Try it out!
    </Link>
   </div>
  </div>
 );
};

export default Home;

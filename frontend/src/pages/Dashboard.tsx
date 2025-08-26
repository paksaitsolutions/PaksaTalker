import React from 'react';

const Dashboard = () => {
  return (
    <div className="bg-gray-100 min-h-screen">
      <div className="container mx-auto py-16">
        <h1 className="text-3xl font-extrabold text-gray-900 text-center">
          Welcome to your Dashboard
        </h1>
        <p className="mt-4 text-gray-600 text-center">
          Manage your videos, track your progress, and explore new features.
        </p>

        <div className="mt-12 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {/* Example Dashboard Item */}
          <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 className="text-lg font-medium text-gray-900">Recent Videos</h3>
            <p className="mt-2 text-gray-500">View and manage your recently generated videos.</p>
            <a href="#" className="mt-4 inline-block text-blue-600 hover:text-blue-800">
              View All
            </a>
          </div>

          {/* Example Dashboard Item */}
          <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 className="text-lg font-medium text-gray-900">Account Information</h3>
            <p className="mt-2 text-gray-500">
              Update your profile and manage your account settings.
            </p>
            <a href="#" className="mt-4 inline-block text-blue-600 hover:text-blue-800">
              Edit Profile
            </a>
          </div>

          {/* Example Dashboard Item */}
          <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 className="text-lg font-medium text-gray-900">Explore New Features</h3>
            <p className="mt-2 text-gray-500">
              Discover the latest features and tools available on PaksaTalker.
            </p>
            <a href="#" className="mt-4 inline-block text-blue-600 hover:text-blue-800">
              Learn More
            </a>
          </div>

          {/* Link to Video Generation */}
          <div className="bg-white rounded-lg shadow-md p-6 hover:shadow-lg transition-shadow duration-300">
            <h3 className="text-lg font-medium text-gray-900">Generate Video</h3>
            <p className="mt-2 text-gray-500">Create a new talking head video.</p>
            <a href="/" className="mt-4 inline-block text-blue-600 hover:text-blue-800">
              Generate Now
            </a>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;

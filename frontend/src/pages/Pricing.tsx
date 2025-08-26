import React from 'react';

const Pricing = () => {
  return (
    <div className="bg-gray-100 py-20">
      <div className="container mx-auto">
        <h1 className="text-5xl font-bold text-gray-800 text-center mb-8">
          Choose Your Perfect Plan
        </h1>
        <p className="text-xl text-gray-600 text-center mb-12">
          Start creating amazing videos today with our flexible pricing plans.
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
          {/* Basic Plan */}
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Basic</h2>
            <div className="text-5xl font-bold text-blue-500 mb-2">$9</div>
            <p className="text-gray-600 mb-4">per month</p>
            <ul className="list-disc list-inside text-gray-700 mb-6">
              <li>10 Videos per month</li>
              <li>Standard Quality</li>
              <li>Basic Support</li>
            </ul>
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Get Started
            </button>
          </div>

          {/* Pro Plan */}
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Pro</h2>
            <div className="text-5xl font-bold text-blue-500 mb-2">$29</div>
            <p className="text-gray-600 mb-4">per month</p>
            <ul className="list-disc list-inside text-gray-700 mb-6">
              <li>50 Videos per month</li>
              <li>High Quality</li>
              <li>Priority Support</li>
            </ul>
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Get Started
            </button>
          </div>

          {/* Enterprise Plan */}
          <div className="bg-white rounded-lg shadow-md p-8">
            <h2 className="text-2xl font-semibold text-gray-800 mb-4">Enterprise</h2>
            <div className="text-5xl font-bold text-blue-500 mb-2">Contact Us</div>
            <p className="text-gray-600 mb-4">Custom pricing</p>
            <ul className="list-disc list-inside text-gray-700 mb-6">
              <li>Unlimited Videos</li>
              <li>4K Quality</li>
              <li>Dedicated Support</li>
            </ul>
            <button className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
              Contact Us
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Pricing;
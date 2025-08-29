export default function TestGrid() {
  return (
    <div className="min-h-screen bg-gray-50 p-4">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-4xl font-bold text-center mb-8">Test Grid Layout</h1>
        
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-blue-500">
            <h2 className="text-xl font-bold text-blue-600">Section 1</h2>
            <p className="text-gray-600">This should be top-left</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-green-500">
            <h2 className="text-xl font-bold text-green-600">Section 2</h2>
            <p className="text-gray-600">This should be top-right</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-purple-500">
            <h2 className="text-xl font-bold text-purple-600">Section 3</h2>
            <p className="text-gray-600">This should be bottom-left</p>
          </div>
          
          <div className="bg-white p-6 rounded-lg shadow-lg border-2 border-red-500">
            <h2 className="text-xl font-bold text-red-600">Section 4</h2>
            <p className="text-gray-600">This should be bottom-right</p>
          </div>
        </div>
      </div>
    </div>
  )
}
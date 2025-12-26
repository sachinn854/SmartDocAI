export default function Loader() {
  return (
    <div className="fixed inset-0 bg-gray-900 bg-opacity-50 flex items-center justify-center z-50 backdrop-blur-sm animate-fade-in">
      <div className="bg-white rounded-xl shadow-2xl p-6 flex items-center space-x-4 animate-scale-in">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-600"></div>
        <p className="text-gray-700 font-medium">Processing...</p>
      </div>
    </div>
  );
}

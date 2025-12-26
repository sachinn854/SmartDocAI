import { uploadDocument } from '../api/upload';

export default function UploadDocument({ onUploadSuccess, loading }) {
  const handleFileUpload = async (e) => {
    e.preventDefault();
    
    const fileInput = document.getElementById('fileInput');
    const file = fileInput.files[0];
    
    if (!file) {
      return;
    }
    
    try {
      const data = await uploadDocument(file);
      onUploadSuccess(data);
      fileInput.value = '';
    } catch (error) {
      throw error;
    }
  };
  
  return (
    <div className="relative group">
      <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-2xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
      <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:border-purple-500/50 transition-all duration-300">
        <div className="text-center mb-8">
          <div className="text-7xl mb-6">📤</div>
          <h2 className="text-3xl font-bold text-white mb-3">Upload Document</h2>
          <p className="text-gray-400 text-lg max-w-md mx-auto leading-relaxed">
            Drag & drop or select PDF, DOCX, or TXT files to get started with AI analysis
          </p>
        </div>
        
        <form onSubmit={handleFileUpload} className="space-y-6">
          <div className="relative">
            <input 
              type="file" 
              id="fileInput" 
              accept=".pdf,.txt,.docx" 
              disabled={loading}
              className="block w-full text-sm text-gray-400 file:mr-4 file:py-3 file:px-6 file:rounded-xl file:border-0 file:text-sm file:font-semibold file:bg-gradient-to-r file:from-purple-600 file:to-cyan-600 file:text-white hover:file:from-purple-500 hover:file:to-cyan-500 file:transition-all file:duration-300 disabled:opacity-50 disabled:cursor-not-allowed transition-all backdrop-blur-sm"
            />
          </div>
          
          <button 
            type="submit" 
            disabled={loading}
            className="group relative w-full bg-gradient-to-r from-purple-600 to-cyan-600 text-white font-semibold py-4 px-6 rounded-xl hover:from-purple-500 hover:to-cyan-500 focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 focus:ring-offset-slate-800 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
          >
            <span className="relative z-10">
              {loading ? (
                <div className="flex items-center justify-center">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3"></div>
                  Processing...
                </div>
              ) : (
                <div className="flex items-center justify-center">
                  <span className="mr-2">🚀</span>
                  Upload & Analyze
                </div>
              )}
            </span>
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl blur opacity-0 group-hover:opacity-50 transition-opacity duration-300"></div>
          </button>
        </form>
        
        <div className="mt-8 text-center">
          <p className="text-sm text-gray-500">
            Supported formats: PDF, DOCX, TXT • Maximum file size: 10MB
          </p>
        </div>
      </div>
    </div>
  );
}

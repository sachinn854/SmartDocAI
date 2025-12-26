import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDocuments } from '../context/DocumentContext';
import UploadDocument from '../components/UploadDocument';
import { summarizeDocument, indexDocument } from '../api/upload';

export default function DashboardPage() {
  const navigate = useNavigate();
  const { documents, addDocument } = useDocuments();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // Calculate stats from actual data
  const totalDocuments = documents.length;
  const totalAnalyses = documents.filter(doc => doc.summary).length;
  const totalQuestions = 0; // This would need to be tracked separately if needed

  const handleUploadSuccess = async (uploadData) => {
    setLoading(true);
    setError(null);
    
    try {
      // Summarize document
      const summaryData = await summarizeDocument(uploadData.doc_id);
      
      // Index document
      await indexDocument(uploadData.doc_id);
      
      // Add to documents with summary
      addDocument(uploadData.doc_id, uploadData.filename, summaryData);
      
      // Navigate to document page
      navigate(`/document/${uploadData.doc_id}`);
      
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 rounded-2xl blur opacity-30"></div>
        <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8">
          <div className="flex items-center justify-between">
            <div className="text-center md:text-left">
              <h1 className="text-5xl font-bold bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent mb-4">
                Dashboard
              </h1>
              <p className="text-gray-400 text-xl max-w-2xl">
                Upload and analyze your documents with cutting-edge AI technology
              </p>
            </div>
            <div className="hidden lg:block">
              <div className="text-8xl opacity-40">📊</div>
            </div>
          </div>
        </div>
      </div>

      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="group relative">
          <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
          <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300">
            <div className="text-center">
              <div className="text-4xl mb-4">📄</div>
              <div className="text-3xl font-bold text-white mb-2">{totalDocuments}</div>
              <div className="text-gray-400 font-medium">Documents Uploaded</div>
            </div>
          </div>
        </div>

        <div className="group relative">
          <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
          <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-cyan-500/50 transition-all duration-300">
            <div className="text-center">
              <div className="text-4xl mb-4">🧠</div>
              <div className="text-3xl font-bold text-white mb-2">{totalAnalyses}</div>
              <div className="text-gray-400 font-medium">AI Analyses Complete</div>
            </div>
          </div>
        </div>

        <div className="group relative">
          <div className="absolute inset-0 bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
          <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-pink-500/50 transition-all duration-300">
            <div className="text-center">
              <div className="text-4xl mb-4">💬</div>
              <div className="text-3xl font-bold text-white mb-2">{totalQuestions}</div>
              <div className="text-gray-400 font-medium">Questions Answered</div>
            </div>
          </div>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl p-4 backdrop-blur-sm">
          <div className="flex items-center">
            <div className="text-red-400 mr-3">⚠️</div>
            <div>
              <p className="text-red-300 font-medium">{error}</p>
              <button 
                onClick={() => setError(null)}
                className="text-red-400 hover:text-red-300 text-sm mt-1 underline"
              >
                Dismiss
              </button>
            </div>
          </div>
        </div>
      )}
      
      {/* Loading State */}
      {loading && (
        <div className="flex items-center justify-center py-12">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
            <p className="text-gray-400">Processing your document...</p>
          </div>
        </div>
      )}
      
      {/* Upload Section */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/10 to-cyan-600/10 rounded-2xl blur"></div>
        <div className="relative">
          <UploadDocument 
            onUploadSuccess={handleUploadSuccess} 
            loading={loading}
          />
        </div>
      </div>

      {/* Recent Documents */}
      {documents.length > 0 && (
        <div className="relative">
          <div className="absolute inset-0 bg-gradient-to-r from-slate-600/10 to-slate-700/10 rounded-2xl blur"></div>
          <div className="relative bg-slate-800/30 backdrop-blur-sm border border-slate-700/30 rounded-2xl p-6">
            <h2 className="text-2xl font-bold text-white mb-4 flex items-center">
              <span className="text-3xl mr-3">📋</span>
              Recent Documents
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {documents.slice(-6).map((doc) => (
                <button
                  key={doc.docId}
                  onClick={() => navigate(`/document/${doc.docId}`)}
                  className="group relative p-4 bg-slate-700/30 hover:bg-slate-700/50 rounded-xl border border-slate-600/30 hover:border-slate-500/50 transition-all duration-300 text-left"
                >
                  <div className="flex items-start space-x-3">
                    <div className="text-2xl">📄</div>
                    <div className="flex-1 min-w-0">
                      <p className="font-medium text-white truncate group-hover:text-purple-300 transition-colors">
                        {doc.fileName}
                      </p>
                      <p className="text-xs text-gray-400 mt-1">{doc.uploadedAt}</p>
                      {doc.summary && (
                        <div className="flex items-center mt-2">
                          <div className="w-2 h-2 bg-green-400 rounded-full mr-2"></div>
                          <span className="text-xs text-green-400">Analyzed</span>
                        </div>
                      )}
                    </div>
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

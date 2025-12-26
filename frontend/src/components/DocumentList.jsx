import { useNavigate } from 'react-router-dom';
import { useDocuments } from '../context/DocumentContext';

export default function DocumentList() {
  const navigate = useNavigate();
  const { documents, activeDocId, setActiveDocId } = useDocuments();

  const handleDocumentClick = (docId) => {
    setActiveDocId(docId);
    navigate(`/document/${docId}`);
  };

  if (documents.length === 0) {
    return (
      <div className="h-full p-6">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-white mb-2">Documents</h2>
          <p className="text-gray-400 text-sm">Your uploaded files</p>
        </div>
        
        <div className="text-center py-12">
          <div className="text-6xl mb-4 opacity-50">📄</div>
          <p className="text-gray-400 mb-2">No documents yet</p>
          <p className="text-gray-500 text-sm">Upload your first document to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      <div className="p-6 border-b border-slate-700/50">
        <h2 className="text-xl font-bold text-white mb-2">Documents</h2>
        <p className="text-gray-400 text-sm">{documents.length} file{documents.length !== 1 ? 's' : ''} uploaded</p>
      </div>
      
      <div className="p-4 space-y-3">
        {documents.map((doc) => (
          <button
            key={doc.docId}
            onClick={() => handleDocumentClick(doc.docId)}
            className={`group relative w-full text-left transition-all duration-300 ${
              activeDocId === doc.docId ? 'scale-105' : 'hover:scale-102'
            }`}
          >
            <div className={`absolute inset-0 rounded-xl blur transition duration-300 ${
              activeDocId === doc.docId 
                ? 'bg-gradient-to-r from-purple-600 to-cyan-600 opacity-30' 
                : 'bg-gradient-to-r from-slate-600 to-slate-700 opacity-20 group-hover:opacity-30'
            }`}></div>
            
            <div className={`relative bg-slate-800/50 backdrop-blur-sm border rounded-xl p-4 transition-all duration-300 ${
              activeDocId === doc.docId
                ? 'border-purple-500/50 bg-slate-700/50'
                : 'border-slate-700/50 hover:border-slate-600/50'
            }`}>
              <div className="flex items-start space-x-3">
                <div className="text-2xl flex-shrink-0">📄</div>
                <div className="flex-1 min-w-0">
                  <p className={`font-medium truncate transition-colors ${
                    activeDocId === doc.docId ? 'text-purple-300' : 'text-white group-hover:text-gray-200'
                  }`}>
                    {doc.fileName}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">{doc.uploadedAt}</p>
                  {activeDocId === doc.docId && (
                    <div className="flex items-center mt-2">
                      <div className="w-2 h-2 bg-green-400 rounded-full mr-2 animate-pulse"></div>
                      <span className="text-xs text-green-400">Active</span>
                    </div>
                  )}
                </div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}

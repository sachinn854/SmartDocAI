import { createContext, useContext, useState } from 'react';

const DocumentContext = createContext();

export const useDocuments = () => {
  const context = useContext(DocumentContext);
  if (!context) {
    throw new Error('useDocuments must be used within DocumentProvider');
  }
  return context;
};

export const DocumentProvider = ({ children }) => {
  const [documents, setDocuments] = useState([]);
  const [activeDocId, setActiveDocId] = useState(null);

  const addDocument = (docId, fileName, summary) => {
    setDocuments(prev => {
      // Prevent duplicate entries
      if (prev.some(doc => doc.docId === docId)) {
        return prev;
      }
      
      const newDoc = {
        docId,
        fileName,
        summary,
        uploadedAt: new Date().toLocaleString()
      };
      return [...prev, newDoc];
    });
    setActiveDocId(docId);
  };

  const getActiveDocument = () => {
    return documents.find(doc => doc.docId === activeDocId);
  };

  return (
    <DocumentContext.Provider value={{
      documents,
      activeDocId,
      setActiveDocId,
      addDocument,
      getActiveDocument
    }}>
      {children}
    </DocumentContext.Provider>
  );
};

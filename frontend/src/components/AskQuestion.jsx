import { useState } from 'react';
import { askQuestion } from '../api/ask';

export default function AskQuestion({ docId, onAskQuestion, onAskStart, loading }) {
  const [isAsking, setIsAsking] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    
    const questionInput = document.getElementById('questionInput');
    const question = questionInput.value.trim();
    
    if (!question || !docId) {
      return;
    }
    
    setIsAsking(true);
    if (onAskStart) onAskStart();
    
    try {
      const data = await askQuestion(docId, question);
      onAskQuestion(data);
      questionInput.value = '';
    } finally {
      setIsAsking(false);
    }
  };
  
  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md p-4 sm:p-6 transition-all duration-200">
      <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-3 sm:mb-4">Ask a Question</h2>
      <form onSubmit={handleSubmit} className="space-y-3 sm:space-y-4">
        <input 
          type="text" 
          id="questionInput" 
          placeholder="Ask a question about the document..." 
          disabled={isAsking || loading || !docId}
          className="w-full px-3 sm:px-4 py-3 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-green-500 focus:border-transparent disabled:bg-gray-100 dark:disabled:bg-gray-800 disabled:cursor-not-allowed transition-all duration-200 hover:border-gray-400 dark:hover:border-gray-500 text-sm sm:text-base min-h-[44px]"
        />
        <button 
          type="submit" 
          disabled={isAsking || loading || !docId}
          className="w-full bg-green-600 text-white font-semibold py-3 px-4 rounded-lg hover:bg-green-700 active:scale-95 focus:outline-none focus:ring-2 focus:ring-green-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 shadow-sm hover:shadow-md text-sm sm:text-base min-h-[44px]"
        >
          {isAsking ? (
            <span className="flex items-center justify-center">
              <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" fill="none" viewBox="0 0 24 24">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
              </svg>
              Asking...
            </span>
          ) : 'Ask'}
        </button>
      </form>
    </div>
  );
}

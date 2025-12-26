export default function AnswerBox({ answer, loading }) {
  // Show skeleton while loading
  if (loading) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md p-6 transition-all duration-200">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">Answer</h2>
        
        <div className="mb-6 bg-gradient-to-r from-blue-50 dark:from-blue-900 to-transparent p-4 rounded-lg border-l-4 border-blue-300 dark:border-blue-600">
          <div className="space-y-3">
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-3/4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
          </div>
        </div>
        
        <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
          <div className="h-4 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse mb-3"></div>
          <div className="flex flex-wrap gap-2">
            <div className="h-8 w-32 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
            <div className="h-8 w-40 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
            <div className="h-8 w-36 bg-gray-200 dark:bg-gray-700 rounded-full animate-pulse"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!answer) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md p-6 transition-all duration-200">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">Answer</h2>
        <p className="text-gray-500 dark:text-gray-400 italic">No answer yet. Ask a question to see the answer here.</p>
      </div>
    );
  }
  
  // Calculate confidence percentage and color
  const confidencePercent = Math.min(Math.max(answer.confidence * 100, 0), 100);
  const getConfidenceColor = (percent) => {
    if (percent >= 70) return 'bg-green-500';
    if (percent >= 40) return 'bg-yellow-500';
    return 'bg-red-500';
  };
  const confidenceColor = getConfidenceColor(confidencePercent);

  // Source badge configuration
  const getSourceBadge = () => {
    const source = answer.source?.toLowerCase();
    if (source === 'document') {
      return {
        text: 'From Document',
        bgColor: 'bg-green-100 dark:bg-green-900',
        textColor: 'text-green-700 dark:text-green-300',
        icon: '📄'
      };
    } else if (source === 'web' || answer.used_web) {
      return {
        text: 'From Web',
        bgColor: 'bg-blue-100 dark:bg-blue-900',
        textColor: 'text-blue-700 dark:text-blue-300',
        icon: '🌐'
      };
    } else {
      return {
        text: 'Low Confidence',
        bgColor: 'bg-gray-100 dark:bg-gray-700',
        textColor: 'text-gray-700 dark:text-gray-300',
        icon: '⚠️'
      };
    }
  };
  const sourceBadge = getSourceBadge();

  return (
    <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md p-4 sm:p-6 transition-all duration-200 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0 mb-3 sm:mb-4">
        <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200">Answer</h2>
        <span className={`inline-flex items-center px-3 py-1 rounded-full text-xs font-semibold ${sourceBadge.bgColor} ${sourceBadge.textColor} self-start sm:self-auto`}>
          <span className="mr-1">{sourceBadge.icon}</span>
          {sourceBadge.text}
        </span>
      </div>
      
      <div className="mb-4 sm:mb-6 bg-gradient-to-r from-blue-50 dark:from-blue-900 to-transparent p-3 sm:p-4 rounded-lg border-l-4 border-blue-500 dark:border-blue-400">
        <p className="text-sm sm:text-base lg:text-lg text-gray-800 dark:text-gray-200 leading-relaxed">{answer.answer}</p>
      </div>

      {/* Confidence Progress Bar */}
      <div className="mb-6">
        <div className="flex items-center justify-between mb-2">
          <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Confidence</span>
          <span className="text-sm font-semibold text-gray-900 dark:text-gray-100">{confidencePercent.toFixed(1)}%</span>
        </div>
        <div className="w-full bg-gray-200 dark:bg-gray-700 rounded-full h-3 overflow-hidden">
          <div 
            className={`h-full ${confidenceColor} rounded-full transition-all duration-300 ease-out`}
            style={{ width: `${confidencePercent}%` }}
          ></div>
        </div>
      </div>
      
      <div className="border-t border-gray-200 dark:border-gray-700 pt-3 sm:pt-4">
        <h3 className="text-xs sm:text-sm font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2 sm:mb-3">Metadata</h3>
        <div className="flex flex-wrap gap-2">
          <span className="inline-flex items-center px-2 sm:px-3 py-1 sm:py-1.5 rounded-full text-xs sm:text-sm font-medium bg-blue-100 dark:bg-blue-900 text-blue-800 dark:text-blue-200 hover:bg-blue-200 dark:hover:bg-blue-800 transition-colors duration-200">
            <svg className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Source: {answer.source}
          </span>
          <span className="inline-flex items-center px-2 sm:px-3 py-1 sm:py-1.5 rounded-full text-xs sm:text-sm font-medium bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200 hover:bg-green-200 dark:hover:bg-green-800 transition-colors duration-200">
            <svg className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Confidence: {(answer.confidence * 100).toFixed(2)}%
          </span>
          <span className={`inline-flex items-center px-2 sm:px-3 py-1 sm:py-1.5 rounded-full text-xs sm:text-sm font-medium transition-colors duration-200 ${
            answer.used_web ? 'bg-purple-100 dark:bg-purple-900 text-purple-800 dark:text-purple-200 hover:bg-purple-200 dark:hover:bg-purple-800' : 'bg-gray-100 dark:bg-gray-700 text-gray-800 dark:text-gray-200 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}>
            <svg className="w-3 h-3 sm:w-4 sm:h-4 mr-1 sm:mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9" />
            </svg>
            Web Search: {answer.used_web ? 'Yes' : 'No'}
          </span>
        </div>
      </div>
    </div>
  );
}

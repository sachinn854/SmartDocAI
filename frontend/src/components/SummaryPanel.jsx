import { useState } from 'react';

export default function SummaryPanel({ summary, loading }) {
  const [activeSummaryType, setActiveSummaryType] = useState('detailed');
  const [isExpanded, setIsExpanded] = useState(false);

  // Download summary as TXT file
  const handleDownloadSummary = () => {
    if (!summary?.detailed_summary) return;

    // Prepare text content
    let summaryText = 'Document Summary\n';
    summaryText += '='.repeat(50) + '\n\n';

    if (Array.isArray(summary.detailed_summary)) {
      summaryText += summary.detailed_summary.join('\n\n');
    } else {
      summaryText += summary.detailed_summary;
    }

    // Create blob and download
    const blob = new Blob([summaryText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `summary_${Date.now()}.txt`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Show skeleton while loading
  if (loading) {
    return (
      <div className="space-y-4">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-2">Document Summary</h2>
        
        {/* Tab Navigation Skeleton */}
        <div className="flex space-x-2 border-b border-gray-200 dark:border-gray-700 pb-2">
          <div className="h-10 w-40 bg-gray-200 dark:bg-gray-700 rounded-t-lg animate-pulse"></div>
          <div className="h-10 w-40 bg-gray-200 dark:bg-gray-700 rounded-t-lg animate-pulse"></div>
          <div className="h-10 w-32 bg-gray-200 dark:bg-gray-700 rounded-t-lg animate-pulse"></div>
        </div>

        {/* Content Skeleton */}
        <div className="bg-gradient-to-r from-purple-50 dark:from-purple-900 to-white dark:to-gray-800 rounded-xl shadow-sm p-6 border-l-4 border-purple-300 dark:border-purple-600">
          <div className="mb-3">
            <div className="h-6 w-32 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
          </div>
          <div className="space-y-3">
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
            <div className="h-4 w-3/4 bg-gray-200 dark:bg-gray-700 rounded animate-pulse"></div>
          </div>
        </div>
      </div>
    );
  }

  if (!summary) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-sm hover:shadow-md p-6 transition-all duration-200">
        <h2 className="text-2xl font-semibold text-gray-800 dark:text-gray-200 mb-4">Summary</h2>
        <p className="text-gray-500 dark:text-gray-400 italic">No summary available. Please upload a document first.</p>
      </div>
    );
  }
  
  return (
    <div className="space-y-3 sm:space-y-4 animate-fade-in">
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 sm:gap-0">
        <h2 className="text-xl sm:text-2xl font-semibold text-gray-800 dark:text-gray-200">Document Summary</h2>
        {summary?.detailed_summary && (
          <button
            onClick={handleDownloadSummary}
            className="self-start sm:self-auto inline-flex items-center px-3 sm:px-4 py-2 bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg font-medium text-xs sm:text-sm transition-colors focus:outline-none focus:ring-2 focus:ring-purple-500 focus:ring-offset-2 shadow-sm hover:shadow-md"
          >
            <svg className="w-4 h-4 mr-1.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
            </svg>
            Download Summary
          </button>
        )}
      </div>
      
      {/* Tab Navigation */}
      <div className="flex flex-wrap gap-2 border-b border-gray-200 dark:border-gray-700 pb-2">
        <button
          onClick={() => setActiveSummaryType('detailed')}
          className={`px-3 sm:px-4 py-2 font-semibold rounded-t-lg transition-all duration-200 text-xs sm:text-sm ${
            activeSummaryType === 'detailed'
              ? 'bg-purple-500 dark:bg-purple-600 text-white shadow-md'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          Detailed
        </button>
        <button
          onClick={() => setActiveSummaryType('medium')}
          className={`px-3 sm:px-4 py-2 font-semibold rounded-t-lg transition-all duration-200 text-xs sm:text-sm ${
            activeSummaryType === 'medium'
              ? 'bg-green-500 dark:bg-green-600 text-white shadow-md'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          Medium
        </button>
        <button
          onClick={() => setActiveSummaryType('short')}
          className={`px-3 sm:px-4 py-2 font-semibold rounded-t-lg transition-all duration-200 text-xs sm:text-sm ${
            activeSummaryType === 'short'
              ? 'bg-blue-500 dark:bg-blue-600 text-white shadow-md'
              : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600'
          }`}
        >
          Short
        </button>
      </div>

      {/* Detailed Summary */}
      {activeSummaryType === 'detailed' && (
        <div className="bg-gradient-to-r from-purple-50 dark:from-purple-900 to-white dark:to-gray-800 rounded-xl shadow-sm hover:shadow-md p-4 sm:p-6 border-l-4 border-purple-500 dark:border-purple-400 transition-all duration-200">
          <h3 className="text-base sm:text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3 flex items-center flex-wrap">
            <span className="bg-purple-100 dark:bg-purple-800 text-purple-700 dark:text-purple-200 text-xs font-bold px-2 py-1 rounded mr-2 mb-1 sm:mb-0">DETAILED</span>
            <span className="text-sm sm:text-base">Key Points</span>
          </h3>
          {Array.isArray(summary.detailed_summary) ? (
            <>
              <ul className={`space-y-2 sm:space-y-3 transition-all duration-300 overflow-hidden ${
                isExpanded ? 'max-h-none' : 'max-h-24'
              }`}>
                {summary.detailed_summary.map((item, index) => (
                  <li key={index} className="text-sm sm:text-base text-gray-600 dark:text-gray-300 leading-relaxed flex items-start hover:translate-x-1 transition-transform duration-200">
                    <span className="text-purple-500 dark:text-purple-400 mr-2 sm:mr-3 mt-1 font-bold">•</span>
                    <span>{item}</span>
                  </li>
                ))}
              </ul>
              {summary.detailed_summary.length > 3 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="mt-3 sm:mt-4 text-xs sm:text-sm font-medium text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors flex items-center min-h-[44px] sm:min-h-0"
                >
                  {isExpanded ? (
                    <>
                      Show less
                      <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                    </>
                  ) : (
                    <>
                      Show more
                      <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </>
                  )}
                </button>
              )}
            </>
          ) : (
            <>
              <p className={`text-gray-600 dark:text-gray-300 leading-relaxed transition-all duration-300 overflow-hidden ${
                isExpanded ? 'max-h-none' : 'max-h-24'
              }`}>
                {summary.detailed_summary}
              </p>
              {summary.detailed_summary?.length > 300 && (
                <button
                  onClick={() => setIsExpanded(!isExpanded)}
                  className="mt-4 text-sm font-medium text-purple-600 dark:text-purple-400 hover:text-purple-700 dark:hover:text-purple-300 transition-colors flex items-center"
                >
                  {isExpanded ? (
                    <>
                      Show less
                      <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 15l7-7 7 7" />
                      </svg>
                    </>
                  ) : (
                    <>
                      Show more
                      <svg className="w-4 h-4 ml-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                      </svg>
                    </>
                  )}
                </button>
              )}
            </>
          )}
        </div>
      )}

      {/* Medium Summary */}
      {activeSummaryType === 'medium' && (
        <div className="bg-gradient-to-r from-green-50 dark:from-green-900 to-white dark:to-gray-800 rounded-xl shadow-sm hover:shadow-md p-6 border-l-4 border-green-500 dark:border-green-400 transition-all duration-200">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3 flex items-center">
            <span className="bg-green-100 dark:bg-green-800 text-green-700 dark:text-green-200 text-xs font-bold px-2 py-1 rounded mr-2">MEDIUM</span>
            Comprehensive Overview
          </h3>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">{summary.medium_summary}</p>
        </div>
      )}

      {/* Short Summary */}
      {activeSummaryType === 'short' && (
        <div className="bg-gradient-to-r from-blue-50 dark:from-blue-900 to-white dark:to-gray-800 rounded-xl shadow-sm hover:shadow-md p-6 border-l-4 border-blue-500 dark:border-blue-400 transition-all duration-200">
          <h3 className="text-lg font-semibold text-gray-700 dark:text-gray-200 mb-3 flex items-center">
            <span className="bg-blue-100 dark:bg-blue-800 text-blue-700 dark:text-blue-200 text-xs font-bold px-2 py-1 rounded mr-2">SHORT</span>
            Quick Overview
          </h3>
          <p className="text-gray-600 dark:text-gray-300 leading-relaxed">{summary.short_summary}</p>
        </div>
      )}
    </div>
  );
}

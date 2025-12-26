import { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { useDocuments } from '../context/DocumentContext';
import SummaryPanel from '../components/SummaryPanel';
import AskQuestion from '../components/AskQuestion';
import AnswerBox from '../components/AnswerBox';

export default function DocumentPage() {
  const { id } = useParams();
  const { setActiveDocId, getActiveDocument } = useDocuments();
  const [answer, setAnswer] = useState(null);
  const [isAnswerLoading, setIsAnswerLoading] = useState(false);

  // Sync activeDocId with URL
  useEffect(() => {
    if (id) {
      setActiveDocId(parseInt(id));
    }
  }, [id, setActiveDocId]);

  const activeDoc = getActiveDocument();
  const summary = activeDoc?.summary || null;
  const docId = activeDoc?.docId || parseInt(id);
  const isSummaryLoading = !summary && id;

  const handleAskQuestion = (answerData) => {
    setAnswer(answerData);
    setIsAnswerLoading(false);
  };

  const handleAskStart = () => {
    setIsAnswerLoading(true);
  };

  return (
    <div className="space-y-8">
      {/* Header Section */}
      <div className="relative">
        <div className="absolute inset-0 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 rounded-2xl blur opacity-30"></div>
        <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8">
          <div className="flex items-center justify-between">
            <div className="min-w-0 flex-1">
              <Link 
                to="/dashboard" 
                className="inline-flex items-center text-purple-400 hover:text-purple-300 font-medium mb-4 transition-colors duration-300 group"
              >
                <svg className="w-4 h-4 mr-2 transform group-hover:-translate-x-1 transition-transform duration-300" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
                Back to Dashboard
              </Link>
              <h1 className="text-4xl font-bold bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent mb-3">
                Document Analysis
              </h1>
              {activeDoc && (
                <div className="flex items-center text-gray-400 text-lg">
                  <span className="text-2xl mr-3">📄</span>
                  <span className="truncate">{activeDoc.fileName}</span>
                </div>
              )}
            </div>
            <div className="hidden md:block">
              <div className="text-6xl opacity-50">🔍</div>
            </div>
          </div>
        </div>
      </div>

      {/* Content Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        {/* Summary Section */}
        <div className="space-y-6">
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
            <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-purple-500/50 transition-all duration-300">
              <div className="flex items-center mb-6">
                <div className="text-3xl mr-4">📋</div>
                <div>
                  <h2 className="text-2xl font-bold text-white">AI Summary</h2>
                  <p className="text-gray-400 text-sm">Intelligent document analysis</p>
                </div>
              </div>
              <SummaryPanel summary={summary} loading={isSummaryLoading} />
            </div>
          </div>
        </div>

        {/* Q&A Section */}
        <div className="space-y-6">
          {/* Ask Question */}
          <div className="relative group">
            <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
            <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-cyan-500/50 transition-all duration-300">
              <div className="flex items-center mb-6">
                <div className="text-3xl mr-4">💬</div>
                <div>
                  <h2 className="text-2xl font-bold text-white">Ask Questions</h2>
                  <p className="text-gray-400 text-sm">Chat with your document</p>
                </div>
              </div>
              <AskQuestion 
                docId={docId}
                onAskQuestion={handleAskQuestion}
                onAskStart={handleAskStart}
                loading={false}
              />
            </div>
          </div>

          {/* Answer Box */}
          {(answer || isAnswerLoading) && (
            <div className="relative group">
              <div className="absolute inset-0 bg-gradient-to-r from-pink-600 to-purple-600 rounded-xl blur opacity-20 group-hover:opacity-30 transition duration-500"></div>
              <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-xl p-6 hover:border-pink-500/50 transition-all duration-300">
                <div className="flex items-center mb-6">
                  <div className="text-3xl mr-4">🤖</div>
                  <div>
                    <h2 className="text-2xl font-bold text-white">AI Response</h2>
                    <p className="text-gray-400 text-sm">Smart answers from your content</p>
                  </div>
                </div>
                <AnswerBox answer={answer} loading={isAnswerLoading} />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

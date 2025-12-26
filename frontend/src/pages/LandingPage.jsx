import { Link } from 'react-router-dom';

export default function LandingPage() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 relative overflow-hidden">
      {/* Animated background elements */}
      <div className="absolute inset-0">
        <div className="absolute top-20 left-20 w-72 h-72 bg-purple-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse"></div>
        <div className="absolute top-40 right-20 w-72 h-72 bg-cyan-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-2000"></div>
        <div className="absolute -bottom-8 left-40 w-72 h-72 bg-pink-500 rounded-full mix-blend-multiply filter blur-xl opacity-20 animate-pulse animation-delay-4000"></div>
      </div>

      <div className="relative z-10 flex items-center justify-center min-h-screen px-4">
        <div className="max-w-6xl w-full text-center">
          {/* Hero Section */}
          <div className="mb-12">
            <div className="inline-flex items-center justify-center p-2 bg-gradient-to-r from-purple-600/20 to-cyan-600/20 rounded-full backdrop-blur-sm border border-purple-500/30 mb-8">
              <span className="text-purple-300 text-sm font-medium px-4 py-1">🚀 AI-Powered Document Intelligence</span>
            </div>
            
            <h1 className="text-7xl md:text-8xl font-bold mb-6 bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent">
              Smart<span className="text-purple-400">Doc</span>AI
            </h1>
            
            <p className="text-2xl text-gray-300 mb-4 max-w-3xl mx-auto leading-relaxed">
              Transform your documents into intelligent conversations with cutting-edge AI
            </p>
            <p className="text-lg text-gray-400 max-w-2xl mx-auto">
              Upload, analyze, and interact with your documents like never before
            </p>
          </div>

          {/* Features Grid */}
          <div className="grid md:grid-cols-3 gap-8 mb-12 max-w-5xl mx-auto">
            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:border-purple-500/50 transition-all duration-300">
                <div className="text-6xl mb-6 transform group-hover:scale-110 transition-transform duration-300">📄</div>
                <h3 className="text-2xl font-bold text-white mb-4">Smart Upload</h3>
                <p className="text-gray-400 leading-relaxed">
                  Drag & drop PDFs, DOCX, and text files with intelligent preprocessing
                </p>
              </div>
            </div>

            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-r from-cyan-600 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:border-cyan-500/50 transition-all duration-300">
                <div className="text-6xl mb-6 transform group-hover:scale-110 transition-transform duration-300">🧠</div>
                <h3 className="text-2xl font-bold text-white mb-4">AI Analysis</h3>
                <p className="text-gray-400 leading-relaxed">
                  Advanced ML models extract insights and generate comprehensive summaries
                </p>
              </div>
            </div>

            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-r from-pink-600 to-purple-600 rounded-2xl blur opacity-25 group-hover:opacity-40 transition duration-1000"></div>
              <div className="relative bg-slate-800/50 backdrop-blur-sm border border-slate-700/50 rounded-2xl p-8 hover:border-pink-500/50 transition-all duration-300">
                <div className="text-6xl mb-6 transform group-hover:scale-110 transition-transform duration-300">💬</div>
                <h3 className="text-2xl font-bold text-white mb-4">Chat Interface</h3>
                <p className="text-gray-400 leading-relaxed">
                  Natural language Q&A with context-aware responses
                </p>
              </div>
            </div>
          </div>

          {/* CTA Buttons */}
          <div className="flex flex-col sm:flex-row gap-6 justify-center items-center mb-12">
            <Link
              to="/signup"
              className="group relative inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-white transition-all duration-300 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl hover:from-purple-500 hover:to-cyan-500 transform hover:scale-105 shadow-lg hover:shadow-purple-500/25"
            >
              <span className="relative z-10">Start Free Trial</span>
              <div className="absolute inset-0 bg-gradient-to-r from-purple-600 to-cyan-600 rounded-xl blur opacity-0 group-hover:opacity-50 transition-opacity duration-300"></div>
            </Link>
            
            <Link
              to="/login"
              className="inline-flex items-center justify-center px-8 py-4 text-lg font-semibold text-purple-300 border-2 border-purple-500/50 rounded-xl hover:border-purple-400 hover:bg-purple-500/10 transition-all duration-300 backdrop-blur-sm"
            >
              Sign In
            </Link>
          </div>

          {/* Stats */}
          <div className="flex flex-wrap justify-center gap-8 text-center">
            <div className="text-gray-400">
              <div className="text-2xl font-bold text-white">99.9%</div>
              <div className="text-sm">Accuracy</div>
            </div>
            <div className="text-gray-400">
              <div className="text-2xl font-bold text-white">&lt; 2s</div>
              <div className="text-sm">Processing</div>
            </div>
            <div className="text-gray-400">
              <div className="text-2xl font-bold text-white">24/7</div>
              <div className="text-sm">Available</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

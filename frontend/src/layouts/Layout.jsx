import { Link, useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import DocumentList from '../components/DocumentList';

export default function Layout({ children }) {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout } = useAuth();
  
  const showSidebar = location.pathname.startsWith('/dashboard') || location.pathname.startsWith('/document');

  const handleLogout = () => {
    logout();
    navigate('/login');
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex flex-col">
      {/* Header */}
      <header className="relative">
        <div className="absolute inset-0 bg-slate-800/50 backdrop-blur-sm border-b border-slate-700/50"></div>
        <div className="relative z-10 max-w-7xl mx-auto px-4 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <Link to="/dashboard" className="flex items-center group">
              <h1 className="text-2xl font-bold bg-gradient-to-r from-white via-purple-200 to-cyan-200 bg-clip-text text-transparent group-hover:scale-105 transition-transform duration-300">
                Smart<span className="text-purple-400">Doc</span>AI
              </h1>
            </Link>

            <nav className="flex items-center space-x-4">
              <Link
                to="/dashboard"
                className="text-gray-300 hover:text-purple-300 font-medium transition-colors duration-300 px-3 py-2 rounded-lg hover:bg-purple-500/10"
              >
                Dashboard
              </Link>
              <button
                onClick={handleLogout}
                className="bg-gradient-to-r from-red-600/20 to-pink-600/20 text-red-300 hover:from-red-600/30 hover:to-pink-600/30 font-medium px-4 py-2 rounded-lg transition-all duration-300 backdrop-blur-sm border border-red-500/30 hover:border-red-400/50"
              >
                Logout
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="flex flex-1 overflow-hidden">
        {showSidebar && (
          <aside className="hidden md:block flex-shrink-0 w-80">
            <div className="h-full bg-slate-800/30 backdrop-blur-sm border-r border-slate-700/50">
              <DocumentList />
            </div>
          </aside>
        )}
        <div className="flex-1 overflow-y-auto">
          <div className="max-w-7xl mx-auto px-4 lg:px-8 py-8">
            {children}
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="relative">
        <div className="absolute inset-0 bg-slate-800/30 backdrop-blur-sm border-t border-slate-700/50"></div>
        <div className="relative z-10 max-w-7xl mx-auto px-4 lg:px-8 py-6">
          <p className="text-center text-gray-400 text-sm">
            © 2025 SmartDocAI. Powered by AI • Built with ❤️
          </p>
        </div>
      </footer>
    </div>
  );
}

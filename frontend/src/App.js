import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './context/AuthContext';
import { DocumentProvider } from './context/DocumentContext';
import { ThemeProvider } from './context/ThemeContext';
import ProtectedRoute from './components/routes/ProtectedRoute';
import Layout from './layouts/Layout';
import LandingPage from './pages/LandingPage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import DashboardPage from './pages/DashboardPage';
import DocumentPage from './pages/DocumentPage';

function App() {
  return (
    <ThemeProvider>
      <AuthProvider>
        <DocumentProvider>
          <Router>
            <Routes>
              <Route path="/" element={<LandingPage />} />
              <Route path="/login" element={<LoginPage />} />
              <Route path="/signup" element={<SignupPage />} />
              
              <Route path="/dashboard" element={
                <ProtectedRoute>
                  <Layout>
                    <DashboardPage />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/document/:id" element={
                <ProtectedRoute>
                  <Layout>
                    <DocumentPage />
                  </Layout>
                </ProtectedRoute>
              } />
            </Routes>
          </Router>
        </DocumentProvider>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;

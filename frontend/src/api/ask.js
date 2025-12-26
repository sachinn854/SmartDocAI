const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const getAuthHeaders = () => {
  const token = localStorage.getItem('access_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export const askQuestion = async (docId, question) => {
  const response = await fetch(`${BASE_URL}/ask/${docId}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...getAuthHeaders()
    },
    body: JSON.stringify({ question }),
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to get answer');
  }
  
  return response.json();
};

const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const getAuthHeaders = () => {
  const token = localStorage.getItem('access_token');
  return token ? { 'Authorization': `Bearer ${token}` } : {};
};

export const uploadDocument = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  
  const response = await fetch(`${BASE_URL}/documents/upload`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders()
    },
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to upload document');
  }
  
  return response.json();
};

export const summarizeDocument = async (docId) => {
  const response = await fetch(`${BASE_URL}/summarize/${docId}`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders()
    }
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to generate summary');
  }
  
  return response.json();
};

export const indexDocument = async (docId) => {
  const response = await fetch(`${BASE_URL}/documents/embed/${docId}`, {
    method: 'POST',
    headers: {
      ...getAuthHeaders()
    }
  });
  
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || 'Failed to index document');
  }
  
  return response.json();
};

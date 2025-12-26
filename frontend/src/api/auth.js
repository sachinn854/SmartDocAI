const BASE_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

export const login = async (email, password) => {
  const formData = new URLSearchParams();
  formData.append('username', email);
  formData.append('password', password);
  
  const response = await fetch(`${BASE_URL}/auth/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded',
    },
    body: formData,
  });
  
  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || 'Login failed');
  }
  
  return response.json();
};

export const signup = async (email, password) => {
  console.log('Signup attempt:', { email, password: '***' }); // Debug log
  console.log('API URL:', `${BASE_URL}/auth/signup`); // Debug log
  
  const response = await fetch(`${BASE_URL}/auth/signup`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ email, password }),
  });
  
  console.log('Signup response status:', response.status); // Debug log
  console.log('Signup response headers:', response.headers); // Debug log
  
  if (!response.ok) {
    const errorText = await response.text();
    console.error('Signup error text:', errorText); // Debug log
    
    let error;
    try {
      error = JSON.parse(errorText);
    } catch {
      error = { detail: `HTTP ${response.status}: ${errorText}` };
    }
    
    console.error('Signup error parsed:', error); // Debug log
    throw new Error(error.detail || 'Signup failed');
  }
  
  return response.json();
};

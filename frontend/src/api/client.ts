import axios from 'axios';

const client = axios.create({
  baseURL: 'http://localhost:8000',
  headers: { 'Content-Type': 'application/json' },
});

client.interceptors.request.use((config) => {
  const raw = localStorage.getItem('auth');
  if (raw) {
    try {
      const auth = JSON.parse(raw);
      if (auth?.token) {
        config.headers.Authorization = `Bearer ${auth.token}`;
      }
    } catch {
      // ignore malformed storage
    }
  }
  return config;
});

export default client;

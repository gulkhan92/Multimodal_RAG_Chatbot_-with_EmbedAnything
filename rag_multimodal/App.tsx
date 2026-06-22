import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import './App.css';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: string[];
}

interface LogEntry {
  timestamp: string;
  level: string;
  message: string;
}

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));
  const [isMonitoringOpen, setIsMonitoringOpen] = useState(false);
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [isLoginView, setIsLoginView] = useState(true);
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [email, setEmail] = useState('');

  const getToken = () => localStorage.getItem('token');

  const handleSend = async () => {
    const freshToken = getToken();

    if (!input.trim() || !freshToken) {
      console.debug('[FE] handleSend blocked:', { hasInput: !!input.trim(), hasToken: !!freshToken });
      return;
    }

    console.debug('[FE] Sending /chat with token:', {
      tokenLen: freshToken ? String(freshToken).length : 0,
      tokenPrefix: freshToken ? String(freshToken).slice(0, 18) : '',
      questionLen: input.length,
    });

    const userMsg: Message = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const response = await axios.post(
        'http://localhost:8000/chat',
        { question: input },
        { headers: { Authorization: `Bearer ${freshToken}` } }
      );

      console.debug('[FE] /chat response status/data keys:', {
        status: response.status,
        keys: Object.keys(response.data || {}),
      });

      const botMsg: Message = {
        sender: 'bot',
        text: response.data.answer,
        sources: response.data.sources
      };
      setMessages(prev => [...prev, botMsg]);
    } catch (error: any) {
      console.error('Chat error:', error);

      const status = error?.response?.status;
      if (status === 401) {
        // Token mismatch/expired; force re-login to avoid infinite 401 loop
        localStorage.removeItem('token');
        setToken(null);
        setMessages(prev => [
          ...prev,
          { sender: 'bot', text: 'Session expired. Please login again.' }
        ]);
      } else {
        setMessages(prev => [...prev, { sender: 'bot', text: 'Error communicating with server.' }]);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (token) {
      const ws = new WebSocket('ws://localhost:8000/ws/logs');
      ws.onmessage = (event) => {
        try {
          const logData = JSON.parse(event.data);
          setLogs((prevLogs) => [logData, ...prevLogs].slice(0, 200)); // Keep last 200 logs
        } catch (e) {
          console.error("Failed to parse log message:", event.data);
        }
      };
      ws.onclose = () => console.log('WebSocket disconnected');

      return () => {
        ws.close();
      };
    }
  }, [token]);

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const form = new URLSearchParams();
      form.append('username', username);
      form.append('password', password);

      const resp = await axios.post('http://localhost:8000/token', form, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });

      const accessToken = resp.data?.access_token;
      if (!accessToken) throw new Error('No access_token in /token response');

      localStorage.setItem('token', accessToken);
      setToken(accessToken);
    } catch (err) {
      console.error('Login error:', err);
      alert('Login failed. Please check your credentials.');
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:8000/users/signup', {
        username,
        email,
        password,
      });
      alert('Signup successful! Please log in.');
      setIsLoginView(true); // Switch to login view
    } catch (err: any) {
      console.error('Signup error:', err);
      const detail = err.response?.data?.detail || 'An unknown error occurred.';
      alert(`Signup failed: ${detail}`);
    }
  };

  if (!token) {
    return (
      <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div
          style={{
            background: '#fff',
            border: '1px solid var(--border-color)',
            borderRadius: 12,
            padding: 32,
            boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
            width: 360,
          }}
        >
          <h2 style={{ marginTop: 0 }}>{isLoginView ? 'Login' : 'Sign Up'}</h2>
          <form onSubmit={isLoginView ? handleLogin : handleSignup}>
            <div style={{ marginBottom: 16 }}>
              <label style={{ display: 'block', marginBottom: 4, fontSize: 14, fontWeight: 500 }}>Username</label>
              <input type="text" value={username} onChange={(e) => setUsername(e.target.value)} required style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)' }} />
            </div>
            {!isLoginView && (
              <div style={{ marginBottom: 16 }}>
                <label style={{ display: 'block', marginBottom: 4, fontSize: 14, fontWeight: 500 }}>Email</label>
                <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)' }} />
              </div>
            )}
            <div style={{ marginBottom: 24 }}>
              <label style={{ display: 'block', marginBottom: 4, fontSize: 14, fontWeight: 500 }}>Password</label>
              <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required style={{ width: '100%', padding: '8px 12px', borderRadius: 6, border: '1px solid var(--border-color)' }} />
            </div>
            <button type="submit" className="send-btn" style={{ width: '100%' }}>
              {isLoginView ? 'Login' : 'Create Account'}
            </button>
          </form>
          <p style={{ textAlign: 'center', fontSize: 14, marginTop: 20 }}>
            {isLoginView ? "Don't have an account? " : "Already have an account? "}
            <a
              href="#"
              onClick={(e) => {
                e.preventDefault();
                setIsLoginView(!isLoginView);
              }}
              style={{ color: 'var(--secondary-color)', textDecoration: 'none', fontWeight: 500 }}
            >
              {isLoginView ? 'Sign Up' : 'Login'}
            </a>
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="logo">
          <span style={{ width: 10, height: 10, borderRadius: 999, background: 'var(--primary-color)' }} />
          <span>Multimodal RAG based Chatbot</span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.5 }}>
          Ask questions over everything in <b>data/</b>.
        </div>
        <hr style={{ border: '0', borderTop: '1px solid var(--border-color)', margin: '20px 0' }} />
        <nav>
            <div style={{ padding: '8px 0', fontWeight: 600, color: 'var(--primary-color)', cursor: 'pointer' }}>Chat</div>
            <div 
              onClick={() => setIsMonitoringOpen(!isMonitoringOpen)} 
              style={{ padding: '8px 0', fontWeight: 600, color: 'var(--primary-color)', cursor: 'pointer', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
            >
              <span>Monitoring</span>
              <span style={{ transform: isMonitoringOpen ? 'rotate(90deg)' : 'rotate(0deg)', transition: 'transform 0.2s' }}>▶</span>
            </div>
            <div style={{ padding: '8px 0', cursor: 'pointer' }}>Settings</div>
        </nav>
        {isMonitoringOpen && (
          <div className="monitoring-panel">
            {logs.length === 0 && <div className="log-entry" style={{color: 'var(--text-muted)'}}>Waiting for logs...</div>}
            {logs.map((log, i) => (
              <div key={i} className={`log-entry ${log.level.toLowerCase()}`}>
                <span className="log-timestamp">{new Date(log.timestamp).toLocaleTimeString()}</span>
                <span className="log-level">[{log.level}]</span>
                <span className="log-message">{log.message}</span>
              </div>
            ))}
          </div>
        )}
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div style={{ fontWeight: 700 }}>Chat</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Grounded responses via Qdrant</div>
        </header>

        <>
          <div className="message-list">
            {messages.map((m, i) => (
              <div key={i} className={`message ${m.sender}`}>
                <p style={{ margin: 0, whiteSpace: 'pre-wrap' }}>{m.text}</p>

                {m.sources && m.sources.length > 0 && (
                  <div className="sources-container">Sources: {m.sources.join(', ')}</div>
                )}
              </div>
            ))}
            {loading && <div className="loading">Assistant is thinking...</div>}
          </div>

          <section className="input-wrapper">
            <div className="input-container">
              <input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                placeholder="Ask about your data..."
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !loading) handleSend();
                }}
              />
              <button className="send-btn" onClick={handleSend} disabled={loading}>
                Send
              </button>
            </div>
          </section>
        </>
      </main>
    </div>
  );
};

export default App;

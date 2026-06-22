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
  const [activeTab, setActiveTab] = useState<'chat' | 'monitoring'>('chat');
  const [logs, setLogs] = useState<LogEntry[]>([]);

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
    if (token && activeTab === 'monitoring') {
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
  }, [token, activeTab]);


  if (!token) {
    return (
      <div className="app-container" style={{ justifyContent: 'center', alignItems: 'center' }}>
        <div
          style={{
            background: '#fff',
            border: '1px solid var(--border-color)',
            borderRadius: 12,
            padding: 28,
            boxShadow: '0 4px 6px -1px rgba(0,0,0,0.1)',
            maxWidth: 520,
            width: '100%',
          }}
        >
          <h2 style={{ marginTop: 0 }}>Login to Multimodal RAG</h2>
          <button
            className="send-btn"
            onClick={async () => {
              try {
                const form = new URLSearchParams();
                // Backend /token uses OAuth2PasswordRequestForm and checks username only (MVP)
                form.append('username', 'staff_user');
                form.append('password', 'anything');

                const resp = await axios.post('http://localhost:8000/token', form, {
                  headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
                });

                const accessToken = resp.data?.access_token;
                const tokenType = resp.data?.token_type;

                console.debug('[FE] /token response:', {
                  hasAccessToken: !!accessToken,
                  accessTokenLen: accessToken ? String(accessToken).length : 0,
                  token_type: tokenType,
                });

                if (!accessToken) throw new Error('No access_token in /token response');

                localStorage.setItem('token', accessToken);
                setToken(accessToken);
              } catch (e) {
                console.error('Login error:', e);
                alert('Login failed. Check backend running on http://localhost:8000 and env.');
              }
            }}
          >
            Enter as Viewer
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="app-container">
      <aside className="sidebar">
        <div className="logo">
          <span style={{ width: 10, height: 10, borderRadius: 999, background: 'var(--primary-color)' }} />
          <span>Multimodal RAG</span>
        </div>
        <div style={{ color: 'var(--text-muted)', fontSize: 13, lineHeight: 1.5 }}>
          Ask questions over everything in <b>data/</b>.
        </div>
        <hr style={{ border: '0', borderTop: '1px solid var(--border-color)', margin: '20px 0' }} />
        <nav>
            <div onClick={() => setActiveTab('chat')} style={{ padding: '8px 0', fontWeight: activeTab === 'chat' ? 600 : 400, color: activeTab === 'chat' ? 'var(--primary-color)' : 'inherit', cursor: 'pointer' }}>Chat</div>
            <div onClick={() => setActiveTab('monitoring')} style={{ padding: '8px 0', fontWeight: activeTab === 'monitoring' ? 600 : 400, color: activeTab === 'monitoring' ? 'var(--primary-color)' : 'inherit', cursor: 'pointer' }}>Monitoring</div>
            <div style={{ padding: '8px 0', cursor: 'pointer' }}>Settings</div>
        </nav>
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div style={{ fontWeight: 700 }}>{activeTab === 'chat' ? 'Chat' : 'Live Monitoring'}</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Grounded responses via Qdrant</div>
        </header>

        {activeTab === 'chat' ? (
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
        ) : (
          <div className="message-list" style={{ fontFamily: 'monospace', fontSize: '12px' }}>
            {logs.map((log, i) => (
              <div key={i} style={{ color: log.level === 'INFO' ? 'inherit' : 'red' }}>
                <span style={{ color: 'var(--text-muted)'}}>{log.timestamp}</span> [{log.level}] {log.message}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
};

export default App;

import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

interface Message {
  sender: 'user' | 'bot';
  text: string;
  sources?: string[];
}

const App: React.FC = () => {
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState<string | null>(localStorage.getItem('token'));

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
      </aside>

      <main className="chat-main">
        <header className="chat-header">
          <div style={{ fontWeight: 700 }}>Chat</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 12 }}>Grounded responses via Qdrant</div>
        </header>

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
      </main>
    </div>
  );
};

export default App;

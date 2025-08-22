

import { useState } from 'react';
import './App.css';

function App() {
  const agentName = 'Abigail';
  const [messages, setMessages] = useState([
    { sender: 'agent', text: `Hello! I'm ${agentName}, your personal AGI assistant. How can I help you today?` }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);

  const handleSend = async () => {
    if (!input.trim()) return;
    const userMsg = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      // Send input to backend agent (FastAPI)
      const res = await fetch('http://localhost:8000/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: input })
      });
      const data = await res.json();
  setMessages(prev => [...prev, { sender: 'agent', text: data.response ? `${agentName}: ${data.response}` : `${agentName}: ...` }]);
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'agent', text: 'Error connecting to agent backend.' }]);
    }
    setLoading(false);
  };

  return (
    <div className="agi-ui">
      <header className="agent-header">
        <h1>{agentName} <span className="agent-role">(AGI Agent)</span></h1>
        <p className="agent-persona">Abigail is your friendly, knowledgeable, and context-aware digital assistant.</p>
      </header>
      <main className="main-panel">
        <section className="chat-panel">
          <div className="messages">
            {messages.map((msg, idx) => (
              <div key={idx} className={`msg ${msg.sender}`}>
                <span>{msg.text}</span>
              </div>
            ))}
            {loading && <div className="msg agent"><span>{agentName}: ...</span></div>}
          </div>
          <div className="chat-input">
            <input
              type="text"
              value={input}
              onChange={e => setInput(e.target.value)}
              placeholder="Type your message..."
              disabled={loading}
            />
            <button onClick={handleSend} disabled={loading}>Send</button>
          </div>
        </section>
      </main>
    </div>
  );
}

export default App;

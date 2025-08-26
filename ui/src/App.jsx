
import React from 'react';
import { useState } from 'react';
import './App.css';

function App() {
  const agentName = 'Abigail';
  const [messages, setMessages] = useState([
    { sender: 'agent', text: `Hello! I'm ${agentName}, your personal AGI assistant. How can I help you today?` }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat'); // chat | journal | facebook | memory
  const [fbUploadStatus, setFbUploadStatus] = useState('');

  // Memory Viewer State
  const [memoryView, setMemoryView] = useState('');
  const [memoryLoading, setMemoryLoading] = useState(false);

  const handleViewMemory = async () => {
    setMemoryLoading(true);
    try {
      const res = await fetch('http://localhost:8000/abigail/memory', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'default' })
      });
      const data = await res.json();
      setMemoryView(data.user_context || 'No memory found.');
    } catch (err) {
      setMemoryView('Error fetching memory.');
    }
    setMemoryLoading(false);
  };

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

  // Facebook Data Upload
  const handleFbFileUpload = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    setFbUploadStatus('Uploading...');
    const formData = new FormData();
    formData.append('file', file);
    try {
      const res = await fetch('http://localhost:8000/import-facebook', {
        method: 'POST',
        body: formData
      });
      if (res.ok) {
        setFbUploadStatus('Upload successful! Abigail will now learn from your Facebook history.');
      } else {
        setFbUploadStatus('Upload failed. Please check your file and try again.');
      }
    } catch (err) {
      setFbUploadStatus('Error uploading file.');
    }
  };

  return (
    <div className="agi-ui">
      <header className="agent-header">
        <h1>{agentName} <span className="agent-role">(AGI Agent)</span></h1>
        <p className="agent-persona">Abigail is your friendly, knowledgeable, and context-aware digital assistant.</p>
        <nav className="tab-nav">
          <button onClick={() => setActiveTab('chat')} className={activeTab === 'chat' ? 'active' : ''}>Chat</button>
          <button onClick={() => setActiveTab('journal')} className={activeTab === 'journal' ? 'active' : ''}>Journal</button>
          <button onClick={() => setActiveTab('facebook')} className={activeTab === 'facebook' ? 'active' : ''}>Import Facebook Data</button>
          <button onClick={() => setActiveTab('memory')} className={activeTab === 'memory' ? 'active' : ''}>View Memory</button>
        </nav>
      </header>
      <main className="main-panel">
        {activeTab === 'chat' && (
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
        )}
        {activeTab === 'facebook' && (
          <section className="fb-import-panel">
            <h2>Import Your Facebook Data</h2>
            <ol>
              <li>Go to Facebook Settings &gt; Your Facebook Information &gt; Download Your Information.</li>
              <li>Choose <b>JSON</b> format for best results.</li>
              <li>Download and unzip the file, then upload the relevant JSON file below.</li>
            </ol>
            <input type="file" accept=".json,.zip" onChange={handleFbFileUpload} />
            {fbUploadStatus && <p className="fb-status">{fbUploadStatus}</p>}
            <p className="fb-privacy">Your data is processed locally and never shared without your consent.</p>
          </section>
        )}
        {activeTab === 'journal' && (
          <section className="journal-panel">
            <h2>Journal Entry (Coming Soon)</h2>
            <p>You'll be able to add personal journal entries for Abigail to learn from.</p>
          </section>
        )}
        {activeTab === 'memory' && (
          <section className="memory-panel">
            <h2>Abigail's Learned Memory</h2>
            <button onClick={handleViewMemory} disabled={memoryLoading} style={{marginBottom: '1em'}}>
              {memoryLoading ? 'Loading...' : 'Refresh Memory'}
            </button>
            <pre className="memory-view" style={{background:'#f9f9f9',padding:'1em',borderRadius:'6px',maxHeight:'400px',overflow:'auto'}}>
              {memoryView}
            </pre>
            <p className="memory-privacy">This is what Abigail has learned and stored in her memory. Only you can view this.</p>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;

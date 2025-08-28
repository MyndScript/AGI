import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const agentName = 'Abigail';
  const [messages, setMessages] = useState([
    { sender: 'agent', text: `Hello! I'm ${agentName}, your personal AGI assistant. How can I help you today?` }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [activeTab, setActiveTab] = useState('chat');
  const [fbUploadStatus, setFbUploadStatus] = useState('');
  const [fbUploadSummary, setFbUploadSummary] = useState(null);
  const [fbPublicApprove, setFbPublicApprove] = useState(false);
  const [memoryView, setMemoryView] = useState('');
  const [memoryLoading, setMemoryLoading] = useState(false);
  const messagesEndRef = useRef(null);

  // Scroll to bottom when messages update
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  // Automated agent decision logic for backend endpoint selection
  const handleSend = async () => {
    if (!input.trim() || loading) return;
    const userMsg = { sender: 'user', text: input };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);
    try {
      // Decision logic: choose endpoint(s) based on input
      let endpoint = 'generate';
      let payload = { prompt: input, user_id: 'default' };
      let agentResponse = '';

      // Example: if user asks about personality, mood, or traits, use personality endpoints
      if (/personality|traits|archetype|mood|context/i.test(input)) {
        endpoint = 'get-personality-context';
        payload = { user_id: 'default' };
        const res = await fetch('http://localhost:8002/get-personality-context', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.context || 'No personality context found.';
      } else if (/memory|remember|recall|learned/i.test(input)) {
        endpoint = 'abigail/memory';
        payload = { user_id: 'default' };
        const res = await fetch('http://localhost:8000/abigail/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.user_context || 'No memory found.';
      } else if (/observe|observation|journal|note/i.test(input)) {
        endpoint = 'add-observation';
        payload = { user_id: 'default', observation: input };
        const res = await fetch('http://localhost:8002/add-observation', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.success ? 'Observation added.' : (data.error || 'Failed to add observation.');
      } else {
        // Default: use chat/generate endpoint
        const res = await fetch('http://localhost:8000/generate', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.response ? `${agentName}: ${data.response}` : `${agentName}: ...`;
      }

      setMessages(prev => [
        ...prev,
        { sender: 'agent', text: agentResponse }
      ]);
    } catch {
      setMessages(prev => [
        ...prev,
        { sender: 'agent', text: 'Error connecting to agent backend.' }
      ]);
    }
    setLoading(false);
  };

  const handleInputKeyDown = (e) => {
    if (e.key === 'Enter') handleSend();
  };

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
    } catch {
      setMemoryView('Error fetching memory.');
    }
    setMemoryLoading(false);
  };

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
      const data = await res.json();
      if (res.ok && data) {
        // Expect backend to return: {success, posts_stored, private_count, public_count, public_candidates}
        setFbUploadStatus('Upload successful! Abigail will now learn from your Facebook history.');
        setFbUploadSummary({
          private: data.private_count || 0,
          public: data.public_count || 0,
          publicCandidates: data.public_candidates || []
        });
      } else {
        setFbUploadStatus('Upload failed. Please check your file and try again.');
        setFbUploadSummary(null);
      }
    } catch {
      setFbUploadStatus('Error uploading file.');
      setFbUploadSummary(null);
    }
    setFbPublicApprove(false);
  };

  // Handler for approving public/global learning
  const handleApprovePublicLearning = async () => {
    if (!fbUploadSummary?.publicCandidates?.length) return;
    setFbUploadStatus('Submitting scrubbed posts for public learning...');
    try {
      const res = await fetch('http://localhost:8000/import-facebook-public', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'default', posts: fbUploadSummary.publicCandidates })
      });
      setFbUploadStatus(res.ok ? 'Scrubbed posts contributed to public/global learning!' : 'Failed to submit for public learning.');
      setFbPublicApprove(true);
    } catch {
      setFbUploadStatus('Error submitting for public learning.');
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
            <div className="messages-container">
              <div className="messages-list">
                {messages.map((msg, idx) => (
                  <div key={idx} className={`msg ${msg.sender}`}>
                    <span>{msg.text}</span>
                  </div>
                ))}
                {loading && <div className="msg agent"><span>{agentName}: ...</span></div>}
                <div ref={messagesEndRef} />
              </div>
            </div>
            <div className="chat-input-bar">
              <input
                type="text"
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleInputKeyDown}
                placeholder="Type your message..."
                disabled={loading}
                aria-label="Message input"
                autoFocus
              />
              <button onClick={handleSend} disabled={loading || !input.trim()}>Send</button>
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
            {fbUploadSummary && (
              <div className="fb-summary">
                <p>
                  <b>Privacy Analysis:</b><br />
                  {fbUploadSummary.private} posts detected as <span style={{color:'red'}}>private</span>.<br />
                  {fbUploadSummary.public} posts can be scrubbed and contributed to <span style={{color:'green'}}>public/global learning</span>.<br />
                </p>
                {!fbPublicApprove && fbUploadSummary.public > 0 && (
                  <button onClick={handleApprovePublicLearning} style={{marginTop:'1em'}}>
                    Yes, contribute scrubbed posts to public/global learning
                  </button>
                )}
                {fbPublicApprove && (
                  <p style={{color:'green'}}>Thank you for contributing to public/global learning!</p>
                )}
              </div>
            )}
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
            <pre className="memory-view">
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
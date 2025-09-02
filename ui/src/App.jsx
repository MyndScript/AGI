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
  const [journalEntry, setJournalEntry] = useState('');
  const [journalEntries, setJournalEntries] = useState([]);
  const [journalLoading, setJournalLoading] = useState(false);
  const [aiOverview, setAiOverview] = useState('');
  const [aiOverviewLoading, setAiOverviewLoading] = useState(false);
  const [memoryLoading, setMemoryLoading] = useState(false);
  const [memoryView, setMemoryView] = useState('Click "Refresh Memory" to view Abigail\'s learned memory.');
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
  const res = await fetch('http://localhost:8010/get-personality-context', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.context || 'No personality context found.';
      } else if (/memory|remember|recall|learned/i.test(input)) {
        endpoint = 'abigail/memory';
        payload = { user_id: 'default' };
  const res = await fetch('http://localhost:8010/abigail/memory', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        agentResponse = data.user_context || 'No memory found.';
      } else if (/observe|observation|journal|note/i.test(input.trim())) {
        // Only run observation logic if input is non-empty and trimmed
        if (!input.trim()) {
          agentResponse = 'Observation summary cannot be empty.';
        } else {
          endpoint = 'add-observation';
          const now = Date.now();
          // Generate a truly unique id using timestamp and random string
          const uniqueId = `obs_${now}_${Math.random().toString(36).substr(2, 9)}`;
          payload = {
            moment: {
              id: uniqueId,
              user_id: 'default',
              summary: input.trim(),
              emotion: 'neutral', // Default to neutral if not analyzed
              glyph: '📝',   // Default glyph for observation
              tags: input.trim().split(' ').filter(w => w.length > 2), // crude tag extraction
              timestamp: now,
              embedding: '' // Optionally set
            }
          };
          // Debug: print payload before sending
          console.log('Sending observation payload:', payload);
          const res = await fetch('http://localhost:8010/add-observation', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          agentResponse = data.success ? 'Observation added.' : (data.error || 'Failed to add observation.');
        }
      } else {
        // Default: use chat/generate endpoint
  const res = await fetch('http://localhost:8010/generate', {
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
  const res = await fetch('http://localhost:8010/abigail/memory', {
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
  const res = await fetch('http://localhost:8010/import-facebook', {
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

  const handleApprovePublicLearning = () => {
    setFbPublicApprove(true);
  };

  const handleSaveJournalEntry = async () => {
    if (!journalEntry.trim()) return;
    
    setJournalLoading(true);
    try {
      const now = new Date();
      const entryData = {
        id: `journal_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        content: journalEntry.trim(),
        timestamp: now.toISOString(),
        date: now.toLocaleDateString(),
        time: now.toLocaleTimeString(),
        user_id: 'default'
      };

      const res = await fetch('http://localhost:8010/add-journal-entry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(entryData)
      });

      if (res.ok) {
        setJournalEntries(prev => [entryData, ...prev]);
        setJournalEntry('');
        setMessages(prev => [
          ...prev,
          { sender: 'agent', text: `${agentName}: Journal entry saved successfully! 📝` }
        ]);
      } else {
        throw new Error('Failed to save journal entry');
      }
    } catch (error) {
      setMessages(prev => [
        ...prev,
        { sender: 'agent', text: `${agentName}: Sorry, I couldn't save your journal entry. Please try again.` }
      ]);
    }
    setJournalLoading(false);
  };

  const handleGetAiOverview = async () => {
    if (!journalEntry.trim()) {
      setAiOverview('Please write something in your journal first before requesting an AI overview.');
      return;
    }

    setAiOverviewLoading(true);
    setAiOverview('');
    
    try {
      const res = await fetch('http://localhost:8010/analyze-journal-entry', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          current_entry: journalEntry.trim(),
          previous_entries: journalEntries.slice(0, 5), // Last 5 entries for context
          user_id: 'default'
        })
      });

      if (res.ok) {
        const data = await res.json();
        setAiOverview(data.analysis || data.overview || 'Analysis complete! Your journal entry has been processed.');
      } else {
        throw new Error('Failed to analyze journal entry');
      }
    } catch (error) {
      setAiOverview('I apologize, but I couldn\'t analyze your journal entry right now. Please try again later.');
    }
    setAiOverviewLoading(false);
  };

  const loadJournalEntries = async () => {
    try {
      const res = await fetch('http://localhost:8010/get-journal-entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ user_id: 'default', limit: 10 })
      });

      if (res.ok) {
        const data = await res.json();
        setJournalEntries(data.entries || []);
      }
    } catch (error) {
      console.error('Failed to load journal entries:', error);
    }
  };

  // Load journal entries on component mount
  useEffect(() => {
    loadJournalEntries();
  }, []);

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
            <div className="journal-header">
              <h2>Personal Journal</h2>
              <p>Write your thoughts and reflections. Abigail will learn from your entries to better understand you.</p>
            </div>
            
            <div className="journal-composer">
              <div className="journal-input-section">
                <textarea
                  value={journalEntry}
                  onChange={(e) => setJournalEntry(e.target.value)}
                  placeholder="What's on your mind today? Write your thoughts, feelings, experiences, or reflections..."
                  className="journal-textarea"
                  rows="8"
                  disabled={journalLoading}
                />
                <div className="journal-actions">
                  <button 
                    onClick={handleSaveJournalEntry} 
                    disabled={journalLoading || !journalEntry.trim()}
                    className="save-journal-btn"
                  >
                    {journalLoading ? 'Saving...' : '💾 Save Entry'}
                  </button>
                  <button 
                    onClick={handleGetAiOverview} 
                    disabled={aiOverviewLoading || !journalEntry.trim()}
                    className="ai-overview-btn"
                  >
                    {aiOverviewLoading ? 'Analyzing...' : '🤖 AI Overview'}
                  </button>
                </div>
              </div>
              
              {aiOverview && (
                <div className="ai-overview-section">
                  <h3>AI Analysis & Insights</h3>
                  <div className="ai-overview-content">
                    {aiOverview}
                  </div>
                </div>
              )}
            </div>
            
            <div className="journal-entries">
              <h3>Recent Entries ({journalEntries.length})</h3>
              {journalEntries.length === 0 ? (
                <p className="no-entries">No journal entries yet. Start writing to see your entries here!</p>
              ) : (
                <div className="entries-list">
                  {journalEntries.map((entry) => (
                    <div key={entry.id} className="journal-entry-card">
                      <div className="entry-header">
                        <span className="entry-date">{entry.date}</span>
                        <span className="entry-time">{entry.time}</span>
                      </div>
                      <div className="entry-content">
                        {entry.content.length > 200 
                          ? `${entry.content.substring(0, 200)}...` 
                          : entry.content
                        }
                      </div>
                      <div className="entry-actions">
                        <button 
                          onClick={() => setJournalEntry(entry.content)}
                          className="edit-entry-btn"
                        >
                          ✏️ Edit
                        </button>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
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
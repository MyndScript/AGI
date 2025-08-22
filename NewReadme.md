# AGI: Experiential, Decentralized, and Personalized AI Best Friend

AGI is an open-source framework for building AI companions that learn and evolve through experience, not just static data ingestion. Instead of scraping the internet, AGI remembers each user's mood, intent, and character—growing with them over time. It fuses external knowledge (Wikipedia, Wikidata, FDA drug info) with a dynamic personality and memory system, making every conversation unique and meaningful.

---

## Architecture Overview

### 1. Agent Layer (`agents/base_agent.py`)

- **Role:** Orchestrator for all AI interactions.
- **Features:**
  - Handles conversational AI using models like GPT-2 and FLAN.
  - Interfaces with external APIs for factual knowledge (Wikidata, Wikipedia, openFDA).
  - Loads and fuses user context: personality traits, mood, and memory.
  - Builds comprehensive prompts for model generation, integrating real-time facts, user memory, and emotional state.
  - Updates both memory and personality after every interaction.

#### Example Flow:
1. **User Input**: "Tell me about aspirin side effects."
2. **Agent**:
   - Loads user’s personality/memory.
   - Queries Wikipedia, Wikidata, and (for medical topics) openFDA.
   - Fuses facts, traits, mood, and memory into a single prompt.
   - Generates response with chosen AI model.
   - Post-processes response to match user's personality.
   - Updates personality and memory with new interaction.

---

### 2. Memory Layer (`memory/pb/memory.pb.go`, `memory/pb/memory_grpc.pb.go`, `memory/memory_server.go`)

- **Role:** Persistent, decentralized context store for every user.
- **Features:**
  - Written in Go for concurrency and robustness.
  - Exposes a secure gRPC API for agents to read/write key-value pairs per user.
  - Supports:
    - Individual memory items (Set/Get).
    - Full user context as a readable string (GetUserContext).
  - Easily extensible for graph-based memory (semantic relationships, emotional arc, etc).

#### Example Usage:
- Agent stores `"last_topic":"aspirin"` for user `user123`.
- Agent retrieves recent conversations and emotional context for prompt-building.
- Memory server can be scaled or distributed for heavy workloads.

---

### 3. Personality Layer (`personality/personality.py`)

- **Role:** Dynamic personality engine, evolving with each user.
- **Features:**
  - Stores and adapts:
    - Personality traits.
    - Mood vector (joy, curiosity, frustration, etc).
    - Triggers and archetype (e.g., muse, strategist).
    - Interaction history (emotional arc).
    - Narrative "glyphs" (key moments/themes).
  - Interprets user tone using punctuation, sentiment, metaphor, intensity, empathy, and more.
  - Tracks emotional arc over time and generates context-aware responses (e.g., empathy if user is sad).
  - Evolves via symbolic triggers, extracted intent, and narrative glyphs.
  - Persistent storage per user (JSON-based).

#### Example Usage:
- User’s mood changes, personality adapts in real-time.
- Agent tailors responses based on emotional history and current state.
- Archetype and glyphs allow thematic/narrative evolution of the relationship.

---

## How It All Fits Together

```
User Input
   ↓
BaseAgent (Python)
   ↳ Loads user memory (Go gRPC)
   ↳ Loads user personality (Python, JSON)
   ↳ Fetches factual context (APIs)
   ↳ Fuses context, traits, mood, memory, facts
   ↳ Generates AI response (GPT-2/FLAN)
   ↳ Personalizes response via personality engine
   ↳ Updates memory and personality with new data
   ↓
User Output
```

#### Key Innovations

- **Experiential Learning:** Each agent learns from interactions, not static datasets.
- **Decentralized Memory:** Go-based memory server for robust, scalable, per-user context.
- **Personality Evolution:** Mood, traits, and narrative arcs evolve—making the AI a true companion.
- **API-based Knowledge:** External APIs provide up-to-date facts, keeping models light and focused on experience.

---

## Suggestions for Further Development

- **Deeper Memory-Personality Integration:** Let personality engine query and update semantic memory graphs for richer context.
- **Shared Memories:** Agents can reference past conversations meaningfully (“Remember last week’s talk on aspirin?”).
- **Growth Feedback:** Track and reflect user emotional development (“You seem happier lately!”).
- **Adaptive Archetypes:** Allow users to “train” or influence their agent’s archetype/personality.
- **Scalable Deployment:** Dockerize and orchestrate memory server for distributed, multi-user environments.
- **Graph-Based Memory:** Extend memory to support relationships between topics, emotions, and experiences.

---

## Getting Started

1. **Run Go Memory Server:**  
   See `memory/memory_server.go`. Requires TLS certs.
2. **Configure Agent:**  
   Set up environment variables for API keys.
3. **Launch Agent:**  
   Instantiate `BaseAgent`, connect to memory and personality modules.
4. **Interact:**  
   Watch the agent grow with you!

---

## Credits

- **Agents:** Orchestration, prompt fusion, external API queries.
- **Memory:** Decentralized, secure, per-user context.
- **Personality:** Dynamic evolution, emotional tracking, narrative glyphs.

---

AGI: Not just an assistant. A best friend that learns, adapts, and grows—just like you.
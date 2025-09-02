# --- Workflow Testing & Documentation ---
# 1. Validate user interaction, learning, audit, and evolution workflows
# 2. Document all new endpoints and UI controls in README/NewReadme.md
# 3. Add onboarding steps for new users and developers

import requests
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

import wikipediaapi
import time
import threading
import hashlib


def query_wikidata(entity: str, limit: int = 3):
    """Query Wikidata for facts about an entity."""
    url = f"https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": entity,
        "language": "en",
        "format": "json",
        "limit": limit
    }
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        return resp.json().get("search", [])
    except Exception as e:
        return {"error": str(e)}

def query_wikipedia(topic: str, sentences: int = 2):
    """Query Wikipedia for a detailed summary, sections, and links about a topic using wikipedia-api."""
    wiki = wikipediaapi.Wikipedia('en')
    page = wiki.page(topic)
    if not page.exists():
        return {"error": f"No Wikipedia page found for '{topic}'"}
    # Get summary
    summary = page.summary
    # Get sections
    sections = {}
    def extract_sections(sections_list):
        for s in sections_list:
            sections[s.title] = s.text[:500]  # Truncate for brevity
            extract_sections(s.sections)
    extract_sections(page.sections)
    # Get links
    links = list(page.links.keys())[:10]
    # Get categories
    categories = list(page.categories.keys())
    return {
        "summary": summary,
        "sections": sections,
        "links": links,
        "categories": categories
    }

def query_openfda_drug_event(term: str, limit: int = 5, api_key: Optional[str] = None):
    """
    Query openFDA Drug Adverse Event endpoint for a reaction term.
    Paste your API key below or pass it as an argument.
    """
    url = "https://api.fda.gov/drug/event.json"
    params = {
        "search": f"reactionmeddrapt:\"{term}\"",
        "limit": limit
    }
    if api_key is not None:
        params["api_key"] = api_key
    try:
        resp = requests.get(url, params=params, timeout=5)
        resp.raise_for_status()
        return resp.json().get("results", [])
    except Exception as e:
        return {"error": str(e)}


class BaseAgent:
    def _init_lightweight_models(self):
        """
        Initialize lightweight transformer models for local inference.
        Prioritize small, fast models for laptop use.
        """
        self.available_models = ["distilgpt2", "flan-t5-small"]
        self.model_pipelines = {}
        if pipeline is not None:
            for m in self.available_models:
                try:
                    if m.startswith("flan-t5"):
                        self.model_pipelines[m] = pipeline("text2text-generation", model=m)
                    else:
                        self.model_pipelines[m] = pipeline("text-generation", model=m)
                except Exception:
                    self.model_pipelines[m] = None
    def fetch_external_facts(self, query):
        """Aggregate facts from Wikidata, Wikipedia, news, weather, books, movies, and openFDA if relevant."""
        facts = {
            "wikidata": self.fetch_wikidata(query),
            "wikipedia": self.fetch_wikipedia(query),
            "news": self.fetch_news(query),
            "weather": self.fetch_weather(query),
            "books": self.fetch_books(query),
            "movies": self.fetch_movies(query)
        }
        medical_keywords = ["medication", "drug", "medicine", "side effect", "sickness", "illness", "prescription", "adverse event", "symptom"]
        if any(word in query.lower() for word in medical_keywords):
            facts["openfda_drug_event"] = self.fetch_openfda(query)
        return facts

    def _cache_key(self, prefix, query):
        """Generate a cache key for knowledge cache."""
        return f"{prefix}:{hashlib.sha256(str(query).encode()).hexdigest()}"

    def __init__(self, model_name="distilgpt2", use_http=True, available_models=None):
        """
        Initialize BaseAgent with model ensemble and context/threading scaffolding.
        """
        self.model_name = model_name
        self.use_http = use_http
        self.model_pipelines = {}
        self._init_lightweight_models()
        # Production-ready memory and personality modules
    # Removed obsolete imports and local class instantiation
        self.memory = None  # Local memory API removed; use REST endpoints
        self.personality = None  # Local personality removed; use REST endpoints
    # Always initialize agentic attributes
        self.global_knowledge = {}  # Shared, de-identified
        self.knowledge_cache = {}
        self.cache_expiry = 600  # seconds
        self.audit_log = []
        self.topic_stack = []
        self.thread_context = {}
    # Autonomous learning loop
        self.learning_thread = None
        self.learning_active = False

    # --- External API helpers ---
    def fetch_wikidata(self, query):
        return query_wikidata(query)

    def fetch_wikipedia(self, query):
        return query_wikipedia(query)

    def fetch_openfda(self, query):
        api_key = os.getenv("OPENFDA_API_KEY")
        return query_openfda_drug_event(query, api_key=api_key)

    def fetch_news(self, query):
        news_api_key = os.getenv("NEWS_API_KEY")
        if news_api_key:
            news_url = f"https://newsapi.org/v2/everything?q={query}&apiKey={news_api_key}&pageSize=2"
            try:
                news_resp = requests.get(news_url, timeout=5)
                if news_resp.ok:
                    return news_resp.json().get("articles", [])
            except Exception:
                pass
        return []
    # --- User-Facing Controls for Memory Audit/Pruning ---
    def user_memory_audit(self, user_id):
        """Return a detailed audit of user's memory actions for UI display."""
        return [log for log in self.audit_log if log.get("user_id") == user_id]

    def user_memory_prune(self, user_id, keys):
        """Allow user to prune multiple memory keys at once using agentic memory API."""
        for key in keys:
            if self.memory:
                try:
                    mem = self.memory.get(user_id)
                    if isinstance(mem, dict) and key in mem:
                        mem.pop(key)
                        self.memory.store(user_id, mem)
                except Exception as e:
                    self.audit_log.append({"action": "prune_error", "user_id": user_id, "key": key, "error": str(e), "ts": time.time()})
            self.audit_log.append({"action": "user_prune", "user_id": user_id, "key": key, "ts": time.time()})

    def user_feedback(self, user_id, feedback):
        """Store user feedback for agent learning and audit."""
        self.audit_log.append({"action": "user_feedback", "user_id": user_id, "feedback": feedback, "ts": time.time()})
        # Optionally update personality or memory
        if self.personality:
            self.personality.update_traits({"last_feedback": feedback})



    def fetch_weather(self, query):
        weather_api_key = os.getenv("WEATHER_API_KEY")
        if weather_api_key and any(w in query.lower() for w in ["weather", "forecast", "temperature"]):
            weather_url = f"https://api.openweathermap.org/data/2.5/weather?q={query}&appid={weather_api_key}&units=metric"
            try:
                weather_resp = requests.get(weather_url, timeout=5)
                if weather_resp.ok:
                    return weather_resp.json()
            except Exception:
                pass
        return {}
    def _apply_style(self, response, style):
        if style == "formal":
            return "[Formal] " + response
        elif style == "informal":
            return "[Casual] " + response
        return response

    def _apply_sentiment(self, response, sentiment):
        sentiment_map = {
            "positive": " 😊🌟",
            "negative": " 😔💔",
            "neutral": " 🤔"
        }
        return response + sentiment_map.get(sentiment, "")

    def fetch_books(self, query):
        if any(w in query.lower() for w in ["book", "novel", "author"]):
            books_url = f"https://www.googleapis.com/books/v1/volumes?q={query}&maxResults=2"
            try:
                books_resp = requests.get(books_url, timeout=5)
                if books_resp.ok:
                    return books_resp.json().get("items", [])
            except Exception:
                pass
        return []
    def _get_context_embedding_similarity(self, query, context_keys, context):
        """Helper to compute embedding similarity for context keys."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            def embed(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return emb
            query_emb = embed(str(query))
            sims = {}
            for key in context_keys:
                value = context.get(key)
                value_text = str(value)
                ctx_emb = embed(value_text)
                sim = float(torch.nn.functional.cosine_similarity(torch.tensor(query_emb), torch.tensor(ctx_emb), dim=0))
                sims[key] = sim
            return sims
        except Exception:
            return {key: 0.0 for key in context_keys}
    # --- Graph-Based Memory Relationships ---
    def add_graph_relationship(self, user_id, source, target, relation_type, tags=None):
        """Add a semantic relationship between memory items/topics/emotions using agentic memory API."""
    # NOTE: MemoryAPI does not support add_relationship; not implemented
        self.audit_log.append({"action": "add_graph_relationship", "user_id": user_id, "source": source, "target": target, "type": relation_type, "tags": tags, "ts": time.time()})
    pass

    def get_graph_relationships(self, user_id, filter_type=None):
        """Query semantic relationships for a user, optionally filtered by type using agentic memory API."""
    # NOTE: MemoryAPI does not support get_relationships; not implemented
        return []

    def add_emotional_arc(self, user_id, arc_name, value):
        """Add/update an emotional arc for a user using agentic memory API."""
    # NOTE: MemoryAPI does not support add_emotional_arc; not implemented
        self.audit_log.append({"action": "add_emotional_arc", "user_id": user_id, "arc": arc_name, "value": value, "ts": time.time()})
    pass

    def link_topics(self, user_id, topic_a, topic_b):
        """Link two topics in the user's memory graph."""
        self.add_graph_relationship(user_id, topic_a, topic_b, relation_type="topic_link")

    # --- Dockerization/Orchestration Steps (for memory server) ---
    # See memory/Dockerfile and docker-compose.yml for actual implementation
    # 1. Ensure memory server exposes REST/gRPC endpoints for graph ops
    # 2. Add Dockerfile for Go/Python memory server (see memory/Dockerfile)
    # 3. Add docker-compose.yml for multi-service orchestration
    # 4. Document scaling steps in README (see NewReadme.md)
    # --- Personality Engine: Semantic Memory Graph & Archetype Evolution ---
    def query_semantic_memory_graph(self, user_id, query=None):
        """
        Query semantic memory graph for user insights, relationships, or context using agentic memory API.
        """
    # NOTE: MemoryAPI does not support query_graph; not implemented
        return {}

    def update_semantic_memory_graph(self, user_id, observation):
        """
        Update semantic memory graph with new observation/glyph using agentic memory API.
        """
    # NOTE: MemoryAPI does not support update_graph; not implemented
        return {}
    def _extract_text_blobs(self, facts):
        """Helper to extract all text from facts dict."""
        def handle_dict(d):
            blobs = []
            for key, value in d.items():
                blobs.extend(handle_value(value))
            return blobs

        def handle_list(l):
            blobs = []
            for item in l:
                blobs.extend(handle_value(item))
            return blobs

        def handle_value(val):
            if isinstance(val, str):
                return [val.strip()] if val.strip() else []
            elif isinstance(val, dict):
                return handle_dict(val)
            elif isinstance(val, list):
                return handle_list(val)
            return []

        return handle_value(facts)

    def _map_sentiment_to_emotions(self, results):
        """Helper to map sentiment results to emotions."""
        emotions = {"hope": 0.0, "anxiety": 0.0, "joy": 0.0, "sadness": 0.0}
        for r in results:
            label = r.get("label", "").lower()
            score = r.get("score", 0.0)
            if "positive" in label:
                emotions["hope"] += score
                emotions["joy"] += score
            elif "negative" in label:
                emotions["anxiety"] += score
                emotions["sadness"] += score
        total = sum(emotions.values())
        if total > 0:
            for k in emotions:
                emotions[k] = round(emotions[k] / total, 2)
        return emotions

    def evolve_archetype(self, user_id, archetype_update):
        """
        Evolve user's archetype/personality based on new insights.
        Persists update, logs change, and triggers downstream effects.
        """
        if self.personality:
            self.personality.archetype = archetype_update
            self.audit_log.append({"action": "archetype_evolve", "user_id": user_id, "archetype": archetype_update, "ts": time.time()})
            self.personality.recalculate_mood()

    # --- Privacy/Audit Controls for UI/User ---
    def get_audit_log(self, user_id=None, limit=50):
        """Expose audit log for UI/user control."""
        logs = self.audit_log
        if user_id:
            logs = [log for log in logs if log.get("user_id") == user_id]
        return logs[-limit:]

    def get_memory_snapshot(self, user_id):
        """Expose current memory snapshot for UI/user control, using MemoryAPI if available."""
        if self.memory is not None:
            try:
                return self.memory.get(user_id)
            except Exception as e:
                self.audit_log.append({"action": "memory_api_error", "user_id": user_id, "error": str(e), "ts": time.time()})
        return {}

    def prune_memory_key(self, user_id, key):
        """Allow user to prune a specific memory key using agentic memory API."""
        if self.memory:
            try:
                mem = self.memory.get(user_id)
                if isinstance(mem, dict) and key in mem:
                    mem.pop(key)
                    self.memory.store(user_id, mem)
                self.audit_log.append({"action": "prune_key", "user_id": user_id, "key": key, "ts": time.time()})
            except Exception as e:
                self.audit_log.append({"action": "prune_error", "user_id": user_id, "key": key, "error": str(e), "ts": time.time()})

    def fetch_movies(self, query):
        omdb_api_key = os.getenv("OMDB_API_KEY")
        if omdb_api_key and any(w in query.lower() for w in ["movie", "film", "director"]):
            omdb_url = f"https://www.omdbapi.com/?apikey={omdb_api_key}&t={query}"
            try:
                omdb_resp = requests.get(omdb_url, timeout=5)
                if omdb_resp.ok:
                    return omdb_resp.json()
            except Exception:
                pass
        return {}

    def start_autonomous_learning(self, interval=3600):
        """Start background thread for autonomous learning loop."""
        if self.learning_thread and hasattr(self.learning_thread, 'is_alive') and self.learning_thread.is_alive():
            return
        self.learning_active = True
        def learning_loop():
            while self.learning_active:
                self.run_learning_cycle()
                time.sleep(interval)
        self.learning_thread = threading.Thread(target=learning_loop, daemon=True)
        self.learning_thread.start()

    def stop_autonomous_learning(self):
        """Stop the autonomous learning loop."""
        self.learning_active = False

    def run_learning_cycle(self):
        """Query web, compare sources, distill insights, update global knowledge."""
        trending_topics = ["adolescent psychology", "grief rituals", "empathy", "cultural values"]
        for topic in trending_topics:
            facts = self.fetch_external_facts(topic)
            emotional_tone = self.analyze_emotional_tone(facts)
            distilled = self.distill_insights(facts)
            self.store_global_learning(topic, distilled, emotional_tone)
            self.audit_log.append({"action": "learned", "topic": topic, "emotional_tone": emotional_tone, "ts": time.time()})


    def analyze_emotional_tone(self, facts):
        """Analyze emotional tone from facts using NLP models."""
        text_blobs = self._extract_text_blobs(facts)
        full_text = " ".join(text_blobs)
        if pipeline is not None:
            try:
                from transformers.pipelines import pipeline as hf_pipeline  # type: ignore
                sentiment_pipe = hf_pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")  # type: ignore
                results = sentiment_pipe(full_text[:512])
                return self._map_sentiment_to_emotions(results)
            except Exception:
                pass
        return {"hope": 0.5, "anxiety": 0.2}

    def distill_insights(self, facts):
        """Distill key insights from facts using summarization and basic de-biasing."""
        text_blobs = self._collect_text_blobs(facts)
        full_text = " ".join(text_blobs)
        summary = self._summarize_text(full_text)
        if summary is not None:
            return {"summary": self._debias_summary(summary)}
        return facts

    def _collect_text_blobs(self, facts):
        """Helper to collect all text from facts."""
        def extract_text(val):
            if isinstance(val, str):
                return [val]
            elif isinstance(val, dict):
                blobs = []
                for v in val.values():
                    blobs.extend(extract_text(v))
                return blobs
            elif isinstance(val, list):
                blobs = []
                for item in val:
                    blobs.extend(extract_text(item))
                return blobs
            return []
        text_blobs = []
        for v in facts.values():
            text_blobs.extend(extract_text(v))
        return text_blobs

    def _summarize_text(self, full_text):
        """Helper to summarize text using transformers pipeline."""
        if pipeline is not None:
            try:
                from transformers.pipelines import pipeline as hf_pipeline  # type: ignore
                summarizer = hf_pipeline("summarization", model="facebook/bart-large-cnn")  # type: ignore
                summary = summarizer(full_text[:1024], max_length=80, min_length=20, do_sample=False)
                return summary[0].get("summary_text", "")
            except Exception:
                pass
        return None

    def _debias_summary(self, insight):
        """Helper to remove common bias phrases from summary."""
        for phrase in ["in my opinion", "it is believed", "some say", "many think"]:
            insight = insight.replace(phrase, "")
        return insight.strip()

    def store_global_learning(self, topic, insights, emotional_tone):
        """Store de-identified insights in global knowledge graph."""
        self.global_knowledge[topic] = {
            "insights": insights,
            "emotional_tone": emotional_tone,
            "origin": "autonomous_learning",
            "ts": time.time()
        }

    def prune_local_memory(self, user_id, key=None):
        """Allow user to prune (forget) memory using agentic API."""
        # NOTE: MemoryAPI does not support prune; fallback to store empty value for key
        if self.memory and key is not None:
            try:
                mem = self.memory.get(user_id)
                if isinstance(mem, dict) and key in mem:
                    mem.pop(key)
                    self.memory.store(user_id, mem)
                self.audit_log.append({"action": "prune", "user_id": user_id, "key": key, "ts": time.time()})
            except Exception as e:
                self.audit_log.append({"action": "prune_error", "user_id": user_id, "key": key, "error": str(e), "ts": time.time()})

    def audit_user_memory(self, user_id):
        """Return audit log for user's memory actions."""
        return [log for log in self.audit_log if log.get("user_id") == user_id]

    def store_local_memory(self, user_id, data):
        """Store/update user memory using agentic MemoryAPI."""
        if self.memory is not None:
            try:
                self.memory.store(user_id, data)
                self.audit_log.append({"action": "store_memory_api", "user_id": user_id, "data": data, "ts": time.time()})
            except Exception as e:
                self.audit_log.append({"action": "memory_api_error", "user_id": user_id, "error": str(e), "ts": time.time()})

    def aggregate_global_insight(self, insight, emotional_tag):
        """Aggregate de-identified user insight into global knowledge."""
        # Differential privacy: anonymize and add noise to user insight before aggregation
        import hashlib, random
        anonymized_id = hashlib.sha256(insight.encode()).hexdigest()[:16]
        noisy_tag = emotional_tag.copy() if isinstance(emotional_tag, dict) else emotional_tag
        if isinstance(noisy_tag, dict):
            for k in noisy_tag:
                noise = random.uniform(-0.05, 0.05)
                noisy_tag[k] = round(max(0.0, min(1.0, noisy_tag[k] + noise)), 2)
        self.global_knowledge.setdefault("user_insights", []).append({
            "anonymized_id": anonymized_id,
            "emotional_tag": noisy_tag,
            "ts": time.time()
        })
        self.audit_log.append({"action": "aggregate_global", "anonymized_id": anonymized_id, "emotional_tag": noisy_tag, "ts": time.time()})

    # Personality engine expansion

    def update_personality_traits(self, traits, user_id=None):
        """
        Update personality traits both locally and via global REST endpoint.
        """
        # Update local personality
        if self.personality:
            self.personality.update_traits(traits)
        # Update global trait scores via REST API
        if user_id:
            try:
                url = "http://localhost:8002/personality/update-trait"
                payload = {"user_id": user_id, "traits": traits}
                resp = requests.post(url, json=payload, timeout=3)
                if resp.ok:
                    self.audit_log.append({"action": "update_global_traits", "user_id": user_id, "traits": traits, "ts": time.time()})
                else:
                    self.audit_log.append({"action": "update_global_traits_error", "user_id": user_id, "traits": traits, "error": resp.text, "ts": time.time()})
            except Exception as e:
                self.audit_log.append({"action": "update_global_traits_error", "user_id": user_id, "traits": traits, "error": str(e), "ts": time.time()})

    def fetch_global_personality_traits(self, user_id):
        """
        Fetch global trait scores for a user from REST API.
        """
        try:
            url = "http://localhost:8002/personality/get-traits"
            payload = {"user_id": user_id}
            resp = requests.post(url, json=payload, timeout=3)
            if resp.ok:
                data = resp.json()
                traits = data.get("traits", {})
                self.audit_log.append({"action": "fetch_global_traits", "user_id": user_id, "traits": traits, "ts": time.time()})
                return traits
            else:
                self.audit_log.append({"action": "fetch_global_traits_error", "user_id": user_id, "error": resp.text, "ts": time.time()})
        except Exception as e:
            self.audit_log.append({"action": "fetch_global_traits_error", "user_id": user_id, "error": str(e), "ts": time.time()})
        return {}

    def update_mood_vector(self, mood):
        if self.personality:
            self.personality.mood_vector.update(mood)

    def add_interaction_history(self, interaction):
        if self.personality:
            self.personality.add_interaction(interaction)

    # Semantic fusion reply logic
    def synthesize_reply(self, context):
        """Production-ready reply: surface top relevant memory and facts only."""
        memory = context.get("memory", [])
        facts = context.get("facts", {})
        query = context.get("query", "")
        # Prioritize relevant memory
        if memory:
            return memory[0] if isinstance(memory, list) and memory[0] else ""  # Most relevant snippet
        # If no memory, use external facts
        if facts:
            # Return concise fact summary
            for v in facts.values():
                if isinstance(v, str) and v:
                    return v
                if isinstance(v, list) and v and isinstance(v[0], str):
                    return v[0]
        # Fallback: direct answer or empty
        return "I'm here to help!"

    def _weight_context(self, context):
        """
        Weight context sources by recency, emotional importance, and optionally embeddings.
        Returns a dict of weighted context.
        """
        weights = {key: 1.0 for key in ["traits", "mood", "memory", "facts", "query"]}
        # Boost recent memory, emotional spikes
        if context.get("mood") and max(context["mood"].values(), default=0) > 0.7:
            weights["mood"] = 1.5
        if context.get("memory") and isinstance(context["memory"], dict) and context["memory"].get("recent"):
            weights["memory"] = 1.3
        # Embedding similarity
        sims = self._get_context_embedding_similarity(context.get("query", ""), ["traits", "mood", "memory", "facts"], context)
        for key, sim in sims.items():
            weights[key] += sim
        return weights

    def build_prompt(self, context):
        weights = self._weight_context(context)
        # Simple weighted prompt fusion
        prompt = (
            f"Traits: {context['traits']} (w={weights['traits']})\n"
            f"Mood: {context['mood']} (w={weights['mood']})\n"
            f"Memory: {context['memory']} (w={weights['memory']})\n"
            f"Facts: {context['facts']} (w={weights['facts']})\n"
            f"User: {context['query']} (w={weights['query']})"
        )
        # Embedding-based fusion for richer context
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            def embed(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return emb
            # Embed each context piece
            context_texts = [str(context.get(k, "")) for k in ["traits", "mood", "memory", "facts", "query"]]
            embeddings = [embed(txt) for txt in context_texts]
            # Compute similarity matrix
            fusion_weights = []
            for i, emb_i in enumerate(embeddings):
                sim_sum = 0.0
                for j, emb_j in enumerate(embeddings):
                    if i != j:
                        sim = float(torch.nn.functional.cosine_similarity(torch.tensor(emb_i), torch.tensor(emb_j), dim=0))
                        sim_sum += sim
                fusion_weights.append(round(sim_sum / (len(embeddings)-1), 2) if len(embeddings) > 1 else 1.0)
            # Build fused prompt
            fused_prompt = ""
            for k, txt, w in zip(["traits", "mood", "memory", "facts", "query"], context_texts, fusion_weights):
                fused_prompt += f"{k.capitalize()}: {txt} (emb_w={w})\n"
            return fused_prompt.strip()
        except Exception:
            return prompt

    def personalize_response(self, response, personality, style=None, sentiment=None):
        # Style
        response = self._apply_style(response, style)
        # Sentiment
        response = self._apply_sentiment(response, sentiment)
        # Mimicry
        if personality and personality.interaction_history:
            last_interaction = personality.interaction_history[-1]
            if last_interaction.get("tone", {}).get("mimicry"):
                response = f"[Mimicry] {response}"
        # Humor
        if personality and personality.mood_vector.get("joy", 0) > 0.7:
            response += " 😄 Here's a fun fact: Did you know honey never spoils?"
        return response
    def push_topic(self, topic):
        """Push a topic onto the conversational stack."""
        self.topic_stack.append({"topic": topic, "ts": time.time()})
        if len(self.topic_stack) > 20:
            self.topic_stack = self.topic_stack[-20:]

    def pop_topic(self):
        """Pop the most recent topic from the stack."""
        if self.topic_stack:
            return self.topic_stack.pop()
        return None

    def get_current_topic(self):
        """Get the current topic from the stack."""
        if self.topic_stack:
            return self.topic_stack[-1]["topic"]
        return None

    def add_feedback(self, user_id, feedback):
        """
        Accept feedback (rating/correction) and update memory/personality.
        """
        # Store feedback in memory
        if self.memory is not None:
            try:
                mem = self.memory.get(user_id)
                feedback_list = mem.get("feedback", []) if isinstance(mem, dict) else []
                feedback_list.append(feedback)
                self.memory.store(user_id, {"feedback": feedback_list})
            except Exception as e:
                self.audit_log.append({"action": "memory_api_error", "user_id": user_id, "error": str(e), "ts": time.time()})
        # Optionally update personality
        if self.personality:
            self.personality.update_traits({"last_feedback": feedback})

    def select_model(self, prompt, traits=None, mood=None):
        """
        Select model based on task, user preference, or detected mood using a decision table.
        """
        def prefers_reasoning(traits):
            return traits and traits.get("prefers_reasoning")
        def prefers_formal(traits):
            return traits and traits.get("prefers_formal")
        def is_creative(mood):
            return mood and mood.get("creativity", 0) > 0.7
        def is_joyful(mood):
            return mood and mood.get("joy", 0) > 0.8
        def is_news_topic(prompt):
            return "news" in prompt.lower()
        def is_story_topic(prompt):
            return any(word in prompt.lower() for word in ["story", "creative"])

        decision_table = [
            (prefers_reasoning(traits), "flan-t5-base"),
            (is_creative(mood), "gpt2"),
            (is_news_topic(prompt), "flan-t5-base"),
            (is_story_topic(prompt), "gpt2"),
            (prefers_formal(traits), "flan-t5-base"),
            (is_joyful(mood), "gpt2"),
        ]
        for condition, model in decision_table:
            if condition and model in self.model_pipelines:
                return model
        return self.model_name

    def _load_user_context(self, user_id):
        # Try to fetch global traits first
        traits = self.fetch_global_personality_traits(user_id) if user_id else {}
        # Fallback to local traits if global not available
        if not traits and self.personality:
            traits = self.personality.traits
        mood = self.personality.mood_vector if self.personality else {}
        local_mem = self.memory.get_local(user_id) if self.memory else {}
        global_mem = self.global_knowledge
        return traits, mood, local_mem, global_mem

    def _fetch_external_knowledge(self, prompt):
        return self.fetch_external_facts(prompt)

    def _retrieve_relevant_memory(self, user_id, query, top_k=5):
        """
        Agentic: Retrieve relevant memory snippets using MemoryAPI, semantic search, and keyword matching.
        """
        if not self.memory:
            return []
        # Try semantic search first (if embedding available)
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            def embed(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return ",".join(str(x) for x in emb)
            query_emb = embed(query)
            moments = self.memory.semantic_search_moments(user_id, query_emb, top_k=top_k)
            import logging
            # Robust type check for moments
            if moments is None:
                logging.warning(f"semantic_search_moments: moments is None")
                return []
            if isinstance(moments, type):
                logging.warning(f"semantic_search_moments: moments is a type, type={type(moments)}, value={moments}")
                return []
            if not isinstance(moments, (list, tuple)):
                logging.warning(f"semantic_search_moments: moments is not a list/tuple, type={type(moments)}, value={moments}")
                return []
            # Debug: print moments for diagnostics
            print(f"DEBUG: semantic_search_moments returned: {moments}")
            from typing import List, Dict
            moments = moments if isinstance(moments, (list, tuple)) else []  # type: List[Dict]
            return [m['summary'] for m in moments if isinstance(m, dict) and 'summary' in m]
        except Exception as e:
            import logging
            logging.error(f"semantic_search_moments: error iterating moments: {e}")
            return []
        # Fallback: keyword match in posts/memory
        memory = self.memory.get_local(user_id)
        snippets = self._collect_memory_snippets(memory)
        keyword_hits = self._keyword_match_snippets(snippets, query)
        if len(keyword_hits) >= top_k:
            return [v for k, v in keyword_hits[:top_k]]
        embedding_hits = self._embedding_similarity_snippets(snippets, query, top_k)
        if embedding_hits is not None:
            return embedding_hits
        return [v for k, v in snippets[:top_k]]

    def _collect_memory_snippets(self, memory):
        """Helper to collect all text-based memory items."""
        snippets = []
        for key, value in memory.items():
            if isinstance(value, str):
                snippets.append((key, value))
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, str):
                        snippets.append((key, item))
            elif isinstance(value, dict):
                for v in value.values():
                    if isinstance(v, str):
                        snippets.append((key, v))
        return snippets

    def _keyword_match_snippets(self, snippets, query):
        """Helper to perform keyword matching on snippets."""
        query_lower = query.lower()
        return [(k, v) for k, v in snippets if query_lower in v.lower() or query_lower in k.lower()]

    def _embedding_similarity_snippets(self, snippets, query, top_k):
        """Helper to score snippets by embedding similarity."""
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
            def embed(text):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
                with torch.no_grad():
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                return emb
            query_emb = embed(query)
            scored = []
            for k, v in snippets:
                ctx_emb = embed(v)
                sim = float(torch.nn.functional.cosine_similarity(torch.tensor(query_emb), torch.tensor(ctx_emb), dim=0))
                scored.append((sim, v))
            scored.sort(reverse=True)
            return [v for sim, v in scored[:top_k]]
        except Exception:
            return None

    def _fuse_context(self, traits, mood, local_mem, global_mem, facts, prompt, user_id=None):
        """
        Fuse context for the model, including only the most relevant memory snippets for the query.
        """
        relevant_memory = self._retrieve_relevant_memory(user_id, prompt, top_k=5) if user_id else []
        return {
            "traits": traits,
            "mood": mood,
            "memory": relevant_memory,
            "facts": facts,
            "global": global_mem,
            "query": prompt
        }

    def _generate_response(self, context, selected_model):
        # Only use model if memory and facts are empty
        memory = context.get("memory", [])
        facts = context.get("facts", {})
        if memory or facts:
            return self.synthesize_reply(context)
        pipeline_obj = self.model_pipelines.get(selected_model)
        prompt = self.build_prompt(context)
        response = None
        if pipeline_obj:
            try:
                if selected_model.startswith("flan-t5"):
                    result = pipeline_obj(prompt, max_length=80)
                    response = result[0]["generated_text"] if "generated_text" in result[0] else result[0]["generated_text"]
                else:
                    result = pipeline_obj(prompt, max_length=80, num_return_sequences=1)
                    response = result[0]["generated_text"]
            except Exception:
                pass
        if not response:
            response = "I'm here to help!"
        return response

    def _post_process_response(self, response, style, sentiment):
        return self.personalize_response(response, self.personality, style=style, sentiment=sentiment)

    def _learn_and_update(self, prompt, response, user_id):
        """
        Store each interaction (prompt, response, timestamp) in agentic memory for future improvement.
        This enables personalization and learning, but keeps the model general.
        """
        if user_id is not None and self.memory:
            try:
                mem = self.memory.get(user_id)
                history = mem.get("interaction_history", []) if isinstance(mem, dict) else []
                history.append({
                    "prompt": prompt,
                    "response": response,
                    "timestamp": time.time()
                })
                self.memory.store(user_id, {"interaction_history": history[-100:]})
            except Exception:
                pass
        if self.personality:
            self.add_interaction_history({"prompt": prompt, "response": response, "timestamp": time.time()})

    def _update_topic_stack(self, prompt, response):
        self.topic_stack.append({"prompt": prompt, "response": response, "ts": time.time()})
        if len(self.topic_stack) > 20:
            self.topic_stack = self.topic_stack[-20:]

    def respond(self, prompt, user_id=None, style=None, sentiment=None):
        traits, mood, local_mem, global_mem = self._load_user_context(user_id)
        facts = self._fetch_external_knowledge(prompt)
        context = self._fuse_context(traits, mood, local_mem, global_mem, facts, prompt)
        selected_model = self.select_model(prompt, traits, mood)
        response = self._generate_response(context, selected_model)
        response = self._post_process_response(response, style, sentiment)
        self._learn_and_update(prompt, response, user_id)
        self._update_topic_stack(prompt, response)
        return response

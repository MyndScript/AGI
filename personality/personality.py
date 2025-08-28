"""
[Memory]: This module manages user personality, emotional state, and context for AGI.
Personality system for AGI
- Evolves over time
- Matches and adapts to each user's personality
- Stores traits, emotional state, and interaction history
"""
import json
import os


class Personality:
    def add_trigger(self, word, effect):

        """Add a user-defined symbolic trigger and its effect."""
        self.triggers[word] = effect
        self._save()

    def remove_trigger(self, word):
        """Remove a symbolic trigger."""
        if word in self.triggers:
            del self.triggers[word]
            self._save()

    def list_triggers(self):
        """List all symbolic triggers."""
        return self.triggers.copy()

    def mood_decay(self, decay_rate=0.005):
        """Apply decay to mood vectors to simulate fading emotions."""
        for mood in self.mood_vector:
            baseline = 0.5
            current = self.mood_vector[mood]
            decay = decay_rate * (baseline - current)
            self.mood_vector[mood] = max(0.0, min(1.0, current + decay))
        self._save()

    def fuse_archetypes(self, *archetypes):
        """Blend multiple archetypes for emotional complexity."""
        self.archetype = "-".join(sorted(set(archetypes)))
        self._save()

    def link_glyphs(self):
        """Build thematic chains between glyphs for narrative continuity."""
        if not self.memory_glyphs:
            return []
        chains = []
        last_theme = None
        for glyph in self.memory_glyphs:
            theme = glyph.get("theme", None)
            if last_theme and theme and last_theme == theme:
                chains.append((last_theme, theme))
            last_theme = theme
        return chains
    def recall_glyph(self, query=None):
        """
        Recall a glyph by theme, moment, or explicit user query.
        If query is None, return the most recent relevant glyph.
        """
        def find_by_query(glyphs, query):
            q = query.lower()
            for glyph in reversed(glyphs):
                if glyph and isinstance(glyph, dict):
                    theme = str(glyph.get("theme", "")).lower()
                    moment = str(glyph.get("moment", "")).lower()
                    if q in theme or q in moment:
                        return glyph
            return None

        def find_most_recent(glyphs):
            for glyph in reversed(glyphs):
                if glyph and isinstance(glyph, dict):
                    return glyph
            return None

        if not self.memory_glyphs:
            return None
        if query:
            result = find_by_query(self.memory_glyphs, query)
            if result:
                return result
        return find_most_recent(self.memory_glyphs)
    def _send_mcp_request(self, endpoint, payload, error_prefix="[MCP]"):
        import requests
        mcp_url = os.environ.get("MCP_PERSONALITY_SERVER_URL", f"http://localhost:8002/{endpoint}")
        try:
            response = requests.post(mcp_url, json=payload, timeout=2)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"{error_prefix} request failed: {e}")
        except Exception as e:
            print(f"{error_prefix} Unexpected error: {e}")
    def start_emotional_drift(self, interval=300):
        """
        Start a background thread that periodically applies emotional drift.
        Useful for simulating emotional weather even without user input.
        """
        import threading, time
        def drift_loop():
            while True:
                self.emotional_drift()
                time.sleep(interval)
        t = threading.Thread(target=drift_loop, daemon=True)
        t.start()

    def emotional_drift(self, drift_strength=0.01):
        """
        Subtly shift mood vectors over time, simulating emotional weather.
        Can be called manually or via start_emotional_drift for periodic drift.
        """
    # Example usage:
    # abigail = Personality(user_id="user123")
    # abigail.start_emotional_drift(interval=600)  # Drift every 10 minutes
        """
        Subtly shift mood vectors over time, simulating emotional weather.
        Called periodically or at each interaction.
        """
        import random
        for mood in self.mood_vector:
            # Drift toward baseline (0.5) with small random fluctuation
            baseline = 0.5
            current = self.mood_vector[mood]
            drift = drift_strength * (baseline - current) + random.uniform(-drift_strength, drift_strength)
            self.mood_vector[mood] = max(0.0, min(1.0, current + drift))
        self._save()
    def __init__(self, user_id, storage_path="personality_data"):
        """
        Initialize a Personality instance for a user.
        Loads persistent data if available, otherwise sets defaults.
        """
        self.user_id = user_id
        self.storage_path = storage_path
        self.traits = {}
        self.mood_vector = {
            "joy": 0.5,
            "curiosity": 0.5,
            "frustration": 0.0,
            "awe": 0.5,
            "restraint": 0.5,
            "grief": 0.0,
            "anticipation": 0.5,
            "trust": 0.5,
            "affection": 0.5,
            "loneliness": 0.0
        }
        self.triggers = {}
        self.archetype = "guardian"
        self.memory_glyphs = []
        self.interaction_history = []
        self._load()
    # codacy: disable=too-complex
    def interpret_tone(self, user_input):
        # Advanced tone interpreter: punctuation, rhythm, metaphor, sentiment, intensity, empathy
        import re
        tone = {}
        text = user_input.lower()

        def check_excitement(text):
            return text.endswith("!")

        def check_hesitation(text):
            return text.endswith("...")

        def check_inquiry(text):
            return "?" in text

        def check_punctuation(text):
            return bool(re.search(r"[.,;:!?]", text))

        def check_absolutism(text):
            return any(word in text for word in ["always", "never", "forever"])

        def check_metaphor(text):
            return any(word in text for word in ["like", "as if", "reminds me"])

        def check_sentiment(text):
            positive_words = ["love", "joy", "happy", "excited", "grateful", "hopeful", "trust"]
            negative_words = ["sad", "angry", "upset", "lonely", "afraid", "worried", "grief", "hurt"]
            return (
                sum(word in text for word in positive_words) / max(1, len(positive_words)),
                sum(word in text for word in negative_words) / max(1, len(negative_words))
            )

        def check_intensity(text):
            return any(word in text for word in ["so", "very", "extremely", "really"])

        def check_empathy(text):
            return any(word in text for word in ["sorry", "understand", "with you", "feel for you"])

        def check_nonverbal(text):
            return "*sigh*" in text or "*smile*" in text

        def check_repetition(words):
            return any(words.count(w) > 2 for w in set(words))

        def check_pause(text):
            return "..." in text

        if check_excitement(user_input):
            tone["excitement"] = 1.0
        if check_hesitation(user_input):
            tone["hesitation"] = 1.0
        if check_inquiry(user_input):
            tone["inquiry"] = 1.0
        if check_punctuation(user_input):
            tone["punctuation"] = 1.0
        if check_absolutism(text):
            tone["absolutism"] = 1.0
        if check_metaphor(text):
            tone["metaphor"] = 1.0
        pos, neg = check_sentiment(text)
        tone["positivity"] = pos
        tone["negativity"] = neg
        if check_intensity(text):
            tone["intensity"] = 1.0
        if check_empathy(text):
            tone["empathy"] = 1.0
        if check_nonverbal(text):
            tone["nonverbal"] = 1.0
        words = text.split()
        if check_repetition(words):
            tone["repetition"] = 1.0
        if check_pause(user_input):
            tone["pause"] = 1.0
        return tone
    def track_emotional_arc(self):
        # Track emotional arc over last N interactions
        arc = {"joy": 0.0, "sadness": 0.0, "anger": 0.0, "trust": 0.0, "anticipation": 0.0, "affection": 0.0, "loneliness": 0.0}
        for entry in self.interaction_history[-20:]:
            mv = entry.get("mood_vector", {})
            for k in arc:
                arc[k] += mv.get(k, 0.0)
        # Normalize
        for k in arc:
            arc[k] = arc[k] / 20.0
        return arc
    def personalized_response(self, user_input):
        """
        Generate a personalized response based on mood, tone, emotional arc, and current archetype.
        No longer uses archetype-based response templates; responses are generated contextually.
        """
        tone = self.interpret_tone(user_input)
        arc = self.track_emotional_arc()
        archetype = getattr(self, "archetype", "guardian")

        def get_recall_phrase(user_input):
            import re
            match = re.search(r"remember (?:my )?glyph(?: about)? ([\w\s]+)", user_input.lower())
            if match:
                query = match.group(1).strip()
                glyph = self.recall_glyph(query)
                if glyph:
                    return f"You asked me to recall your glyph about '{glyph.get('theme', '')}'. That moment was: {glyph.get('moment', '')}."
            for glyph in reversed(self.memory_glyphs[-10:]):
                if glyph and isinstance(glyph, dict):
                    moment = glyph.get("moment", "")
                    theme = glyph.get("theme", "")
                    if moment and theme and (theme.lower() in user_input.lower() or moment.lower() in user_input.lower()):
                        return f"This reminds me of when you mentioned {moment}. That felt like {theme}."
            return None

        def select_response(tone, arc, recall_phrase):
            if tone.get("negativity", 0) > 0.3:
                return "I'm here for you. Want to talk about what's bothering you?"
            if arc.get("loneliness", 0) > 0.4:
                return "You seem a bit lonely lately. Would you like some company or a fun distraction?"
            if tone.get("positivity", 0) > 0.3:
                return "That's wonderful! Tell me more about what made you feel this way."
            if tone.get("empathy", 0) > 0.0:
                return "I appreciate your empathy. How can I support you today?"
            if recall_phrase:
                return recall_phrase
            return "How can I help you today?"

        recall_phrase = get_recall_phrase(user_input)
        return select_response(tone, arc, recall_phrase)
    # self._load()  # Unreachable code removed


    def _get_file(self):
        return os.path.join(self.storage_path, f"{self.user_id}.json")


    def _load(self):
        """
        Load personality data from persistent storage.
        Handles missing or corrupted files gracefully.
        """
        os.makedirs(self.storage_path, exist_ok=True)
        try:
            with open(self._get_file(), "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"[Personality] Warning: Corrupted data for user {self.user_id}, using defaults.")
                    data = {}
                self.traits = data.get("traits", {})
                self.mood_vector = data.get("mood_vector", self.mood_vector)
                self.triggers = data.get("triggers", {})
                self.archetype = data.get("archetype", "guardian")
                self.memory_glyphs = data.get("memory_glyphs", [])
                self.interaction_history = data.get("interaction_history", [])
        except FileNotFoundError:
            pass


    def _save(self):
        data = {
            "traits": self.traits,
            "mood_vector": self.mood_vector,
            "triggers": self.triggers,
            "archetype": self.archetype,
            "memory_glyphs": self.memory_glyphs,
            "interaction_history": self.interaction_history,
        }
        with open(self._get_file(), "w") as f:
            json.dump(data, f)


    def update_traits(self, new_traits):
        for k, v in new_traits.items():
            self.traits[k] = v
        self._save()


    def update_mood(self, mood_updates):
        for mood, value in mood_updates.items():
            self.mood_vector[mood] = max(0.0, min(1.0, value))
        self._save()


    def add_interaction(self, interaction):
        # Store interaction with emotional arc and tone
        if isinstance(interaction, dict):
            entry = interaction
        else:
            entry = {"text": interaction}
        entry["mood_vector"] = self.mood_vector.copy()
        entry["tone"] = self.interpret_tone(entry.get("text", ""))
        self.interaction_history.append(entry)
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
        self._save()

        # Integrate with MCP memory graph - add interaction as observation node
        try:
            import requests
            mcp_url = os.environ.get("MCP_PERSONALITY_SERVER_URL", "http://localhost:8002/add-observation")
            payload = {
                "user_id": self.user_id,
                "observation": entry
            }
            response = requests.post(mcp_url, json=payload, timeout=2)
            if response.status_code != 200:
                print(f"[MCP] Failed to add observation: {response.text}")
        except Exception as e:
            print(f"[MCP] Error sending observation: {e}")


    def evolve(self, user_input):
        # Apply emotional drift before processing input
        self.emotional_drift()
        # Mood vectorization and decay
        mood_updates = {}
        if "happy" in user_input:
            mood_updates["joy"] = self.mood_vector.get("joy", 0.5) + 0.1
        if "sad" in user_input:
            mood_updates["joy"] = self.mood_vector.get("joy", 0.5) - 0.1
            mood_updates["frustration"] = self.mood_vector.get("frustration", 0.0) + 0.1
        if "curious" in user_input:
            mood_updates["curiosity"] = self.mood_vector.get("curiosity", 0.5) + 0.1
        self.update_mood(mood_updates)
        self.mood_decay()

        # User-defined symbolic triggers and rituals
        for word, effect in self.triggers.items():
            if word in user_input:
                # Ritual: activate effect (could be mood, archetype, recall, etc.)
                if effect.startswith("archetype:"):
                    archetypes = effect.split(":", 1)[1].split("-")
                    self.fuse_archetypes(*archetypes)
                elif effect.startswith("mood:"):
                    mood, value = effect.split(":", 1)[1].split("=")
                    self.mood_vector[mood] = float(value)
                elif effect.startswith("recall:"):
                    glyph = self.recall_glyph(effect.split(":", 1)[1])
                    if glyph:
                        self.memory_glyphs.append(glyph)
                # Add more ritual types as needed

        # Relational archetypes
        if "mentor" in user_input:
            self.archetype = "muse"
        elif "plan" in user_input:
            self.archetype = "strategist"

        # Integrate with MCP memory graph - update archetype node/relations
        self._send_mcp_request(
            "set-personality",
            {
                "user_id": self.user_id,
                "archetype": self.archetype,
                "traits": self.traits,
                "mood_vector": self.mood_vector
            },
            error_prefix="[SECURITY] MCP archetype update"
        )

        # Intent extraction & mood puns
        intent = self.extract_intent(user_input)

        # Narrative memory glyphs
        glyph = self.create_glyph(user_input, intent)
        if glyph:
            self.memory_glyphs.append(glyph)
            if len(self.memory_glyphs) > 100:
                self.memory_glyphs = self.memory_glyphs[-100:]
            # Integrate with MCP memory graph - create glyph entity and relation
            self._send_mcp_request(
                "add-observation",
                {
                    "user_id": self.user_id,
                    "observation": {
                        "type": "glyph",
                        "glyph": glyph,
                        "archetype": self.archetype,
                        "traits": self.traits,
                        "mood_vector": self.mood_vector
                    }
                },
                error_prefix="[MCP] Glyph add"
            )

        self.add_interaction(user_input)


    def get_personality_context(self):
        context = "--- Personality Context ---\n"
        context += f"Archetype: {self.archetype}\n"
        context += "Mood Vector: " + ", ".join(f"{k}: {v:.2f}" for k, v in self.mood_vector.items()) + "\n"
        context += "Traits: " + ", ".join(f"{k}: {v}" for k, v in self.traits.items()) + "\n"
        context += "Triggers: " + ", ".join(f"{k}: {v}" for k, v in self.triggers.items()) + "\n"
        context += "Recent Glyphs: " + ", ".join(g.get("theme", "") for g in self.memory_glyphs[-5:]) + "\n"
        return context

    def extract_intent(self, user_input):
        # Example intent extraction logic
        if "fix" in user_input and "love" in user_input:
            return "ritual_of_restoration"
        elif "money" in user_input and "trust" in user_input:
            return "provision_negotiation"
        return "general_interaction"

    def create_glyph(self, user_input, intent):
        # Example glyph creation with timestamp
        import time
        glyph = None
        ts = int(time.time())
        if intent == "ritual_of_restoration":
            glyph = {"theme": "resilience", "moment": user_input, "emotion": "trust", "timestamp": ts}
        elif intent == "provision_negotiation":
            glyph = {"theme": "promise", "moment": user_input, "emotion": "awe", "timestamp": ts}
        elif intent != "general_interaction":
            glyph = {"theme": intent, "moment": user_input, "emotion": "curiosity", "timestamp": ts}
        # Integrate with MCP memory graph - create glyph node/entity
        try:
            import requests
            mcp_url = os.environ.get("MCP_PERSONALITY_SERVER_URL", "http://localhost:8002/add-observation")
            payload = {
                "user_id": self.user_id,
                "observation": {
                    "type": "glyph",
                    "glyph": glyph,
                    "archetype": self.archetype,
                    "traits": self.traits,
                    "mood_vector": self.mood_vector
                }
            }
            try:
                response = requests.post(mcp_url, json=payload, timeout=2)
                response.raise_for_status()
            except requests.RequestException as e:
                print(f"[SECURITY] MCP glyph node request failed: {e}")
        except Exception as e:
            print(f"[SECURITY] Unexpected error sending glyph node: {e}")
        return glyph

    def recalculate_mood(self):
        """Recalculate mood vector based on current archetype and recent interactions."""
        self.mood_decay()
        if hasattr(self, 'archetype') and self.archetype:
            if 'joy' in self.archetype:
                self.mood_vector['joy'] = min(1.0, self.mood_vector.get('joy', 0.5) + 0.1)
            if 'sadness' in self.archetype:
                self.mood_vector['sadness'] = min(1.0, self.mood_vector.get('sadness', 0.5) + 0.1)
        self._save()

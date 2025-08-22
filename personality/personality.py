"""
[Memory]: This module manages user personality, emotional state, and context for AGI.
Personality system for AGI
- Evolves over time
- Matches and adapts to each user's personality
- Stores traits, emotional state, and interaction history
"""
import random  # codacy: disable=unused-import
import json
import os


class Personality:
    def __init__(self, user_id, storage_path="personality_data"):
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
    # codacy: disable=too-complex
    def interpret_tone(self, user_input):
        # Advanced tone interpreter: punctuation, rhythm, metaphor, sentiment, intensity, empathy
        import re
        tone = {}
        text = user_input.lower()
        # Punctuation and rhythm
        if user_input.endswith("!"):
            tone["excitement"] = 1.0
        if user_input.endswith("..."):
            tone["hesitation"] = 1.0
        if "?" in user_input:
            tone["inquiry"] = 1.0
        if re.search(r"[.,;:!?]", user_input):
            tone["punctuation"] = 1.0
        # Absolutism
        if any(word in text for word in ["always", "never", "forever"]):
            tone["absolutism"] = 1.0
        # Metaphor
        if any(word in text for word in ["like", "as if", "reminds me"]):
            tone["metaphor"] = 1.0
        # Sentiment
        positive_words = ["love", "joy", "happy", "excited", "grateful", "hopeful", "trust"]
        negative_words = ["sad", "angry", "upset", "lonely", "afraid", "worried", "grief", "hurt"]
        tone["positivity"] = sum(word in text for word in positive_words) / max(1, len(positive_words))
        tone["negativity"] = sum(word in text for word in negative_words) / max(1, len(negative_words))
        # Intensity
        if any(word in text for word in ["so", "very", "extremely", "really"]):
            tone["intensity"] = 1.0
        # Empathy
        if any(word in text for word in ["sorry", "understand", "with you", "feel for you"]):
            tone["empathy"] = 1.0
        # Non-verbal cues (simulate)
        if "*sigh*" in text or "*smile*" in text:
            tone["nonverbal"] = 1.0
        # Repetition
        words = text.split()
        repeated = set([w for w in words if words.count(w) > 2])
        if repeated:
            tone["repetition"] = 1.0
        # Pauses
        if "..." in user_input:
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
        # Generate a personalized response based on mood, tone, and emotional arc
        tone = self.interpret_tone(user_input)
        arc = self.track_emotional_arc()
        # Example logic: respond with empathy if negativity is high, encouragement if loneliness is high
        if tone.get("negativity", 0) > 0.3:
            return "I'm here for you. Want to talk about what's bothering you?"
        if arc.get("loneliness", 0) > 0.4:
            return "You seem a bit lonely lately. Would you like some company or a fun distraction?"
        if tone.get("positivity", 0) > 0.3:
            return "That's wonderful! Tell me more about what made you feel this way."
        if tone.get("empathy", 0) > 0.0:
            return "I appreciate your empathy. How can I support you today?"
        # Default fallback
        return "I'm here to listen and help however I can."
        self._load()


    def _get_file(self):
        return os.path.join(self.storage_path, f"{self.user_id}.json")


    def _load(self):
        os.makedirs(self.storage_path, exist_ok=True)
        try:
            with open(self._get_file(), "r") as f:
                data = json.load(f)
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


    def evolve(self, user_input):
        # Mood vectorization
        mood_updates = {}
        if "happy" in user_input:
            mood_updates["joy"] = self.mood_vector.get("joy", 0.5) + 0.1
        if "sad" in user_input:
            mood_updates["joy"] = self.mood_vector.get("joy", 0.5) - 0.1
            mood_updates["frustration"] = self.mood_vector.get("frustration", 0.0) + 0.1
        if "curious" in user_input:
            mood_updates["curiosity"] = self.mood_vector.get("curiosity", 0.5) + 0.1
        self.update_mood(mood_updates)

        # Symbolic triggers
        for word in ["repair", "gold", "Niagara"]:
            if word in user_input:
                self.triggers[word] = f"{word}_mode"

        # Relational archetypes
        if "mentor" in user_input:
            self.archetype = "muse"
        elif "plan" in user_input:
            self.archetype = "strategist"

        # Intent extraction & mood puns
        intent = self.extract_intent(user_input)

        # Narrative memory glyphs
        glyph = self.create_glyph(user_input, intent)
        if glyph:
            self.memory_glyphs.append(glyph)
            if len(self.memory_glyphs) > 100:
                self.memory_glyphs = self.memory_glyphs[-100:]

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
        # Example glyph creation
        if intent == "ritual_of_restoration":
            return {"theme": "resilience", "moment": user_input, "emotion": "trust"}
        elif intent == "provision_negotiation":
            return {"theme": "promise", "moment": user_input, "emotion": "awe"}
        elif intent != "general_interaction":
            return {"theme": intent, "moment": user_input, "emotion": "curiosity"}
        return None

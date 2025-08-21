"""
[Memory]: This module manages user personality, emotional state, and context for AGI.
Personality system for AGI
- Evolves over time
- Matches and adapts to each user's personality
- Stores traits, emotional state, and interaction history
"""
import random
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
            "awe": 0.5
        }
        self.triggers = {}
        self.archetype = "guardian"
        self.memory_glyphs = []
        self.interaction_history = []
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
        self.interaction_history.append(interaction)
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

"""
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
        self.emotional_state = "neutral"
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
                self.emotional_state = data.get("emotional_state", "neutral")
                self.interaction_history = data.get("interaction_history", [])
        except FileNotFoundError:
            pass

    def _save(self):
        data = {
            "traits": self.traits,
            "emotional_state": self.emotional_state,
            "interaction_history": self.interaction_history,
        }
        with open(self._get_file(), "w") as f:
            json.dump(data, f)

    def update_traits(self, new_traits):
        for k, v in new_traits.items():
            self.traits[k] = v
        self._save()

    def update_emotional_state(self, state):
        self.emotional_state = state
        self._save()

    def add_interaction(self, interaction):
        self.interaction_history.append(interaction)
        if len(self.interaction_history) > 1000:
            self.interaction_history = self.interaction_history[-1000:]
        self._save()

    def evolve(self, user_input):
        # Example: adjust traits based on user input
        if "happy" in user_input:
            self.update_emotional_state("happy")
            self.update_traits({"positivity": self.traits.get("positivity", 0) + 1})
        elif "sad" in user_input:
            self.update_emotional_state("sad")
            self.update_traits({"positivity": self.traits.get("positivity", 0) - 1})
        # Add more nuanced logic here
        self.add_interaction(user_input)

    def get_personality_context(self):
        context = f"Emotional state: {self.emotional_state}\n"
        context += "Traits: " + ", ".join(f"{k}: {v}" for k, v in self.traits.items()) + "\n"
        return context

import json
import os
import time
import random
from typing import Dict, List, Optional, Any

class Personality:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.data_file = f"personality_data/{user_id}.json"
        self.traits = self._load_traits()
        self.mood_vector = self._load_mood_vector()
        self.interaction_history = self._load_interaction_history()
        self.triggers = self._load_triggers()
        self.glyphs = self._load_glyphs()
        self.archetype = self._load_archetype()

    def _load_traits(self) -> Dict[str, float]:
        """Load personality traits from file or return defaults"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('traits', self._default_traits())
            return self._default_traits()
        except Exception as e:
            print(f"Error loading traits: {e}")
            return self._default_traits()

    def _default_traits(self) -> Dict[str, float]:
        """Default personality traits"""
        return {
            'openness': 0.5,
            'conscientiousness': 0.5,
            'extraversion': 0.5,
            'agreeableness': 0.5,
            'neuroticism': 0.5
        }

    def _load_mood_vector(self) -> Dict[str, float]:
        """Load mood vector from file or return defaults"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('mood_vector', self._default_mood())
            return self._default_mood()
        except Exception as e:
            print(f"Error loading mood vector: {e}")
            return self._default_mood()

    def _default_mood(self) -> Dict[str, float]:
        """Default mood vector"""
        return {
            'happiness': 0.5,
            'energy': 0.5,
            'calmness': 0.5,
            'confidence': 0.5
        }

    def _load_interaction_history(self) -> List[Dict[str, Any]]:
        """Load interaction history from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('interaction_history', [])
            return []
        except Exception as e:
            print(f"Error loading interaction history: {e}")
            return []

    def _load_triggers(self) -> Dict[str, str]:
        """Load triggers from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('triggers', {})
            return {}
        except Exception as e:
            print(f"Error loading triggers: {e}")
            return {}

    def _load_glyphs(self) -> Dict[str, str]:
        """Load glyphs from file"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('glyphs', {})
            return {}
        except Exception as e:
            print(f"Error loading glyphs: {e}")
            return {}

    def _load_archetype(self) -> str:
        """Load archetype from file or return default"""
        try:
            if os.path.exists(self.data_file):
                with open(self.data_file, 'r') as f:
                    data = json.load(f)
                    return data.get('archetype', 'balanced')
            return 'balanced'
        except Exception as e:
            print(f"Error loading archetype: {e}")
            return 'balanced'

    def _save_data(self):
        """Save all personality data to file"""
        data = {
            'traits': self.traits,
            'mood_vector': self.mood_vector,
            'interaction_history': self.interaction_history,
            'triggers': self.triggers,
            'glyphs': self.glyphs,
            'archetype': self.archetype
        }
        os.makedirs(os.path.dirname(self.data_file), exist_ok=True)
        with open(self.data_file, 'w') as f:
            json.dump(data, f, indent=2)

    def add_interaction(self, interaction: Dict[str, Any]):
        """Add an interaction to history"""
        interaction['timestamp'] = time.time()
        self.interaction_history.append(interaction)
        # Keep only last 100 interactions
        if len(self.interaction_history) > 100:
            self.interaction_history = self.interaction_history[-100:]
        self._save_data()

    def personalized_response(self, prompt: str) -> str:
        """Generate a personalized response based on traits and mood"""
        # Simple personality-based response generation
        base_response = self._generate_base_response(prompt)

        # Apply personality modifiers
        response = self._apply_personality_modifiers(base_response, prompt)

        # Update mood based on interaction
        self._update_mood_from_interaction(prompt, response)

        return response

    def _generate_base_response(self, prompt: str) -> str:
        """Generate a basic response (placeholder for more sophisticated generation)"""
        # This is a simple placeholder - in a real implementation you'd use GPT or similar
        responses = [
            "I understand what you're saying.",
            "That's interesting!",
            "Let me think about that.",
            "I appreciate you sharing that with me.",
            "How does that make you feel?"
        ]
        return random.choice(responses)

    def _apply_personality_modifiers(self, response: str, prompt: str) -> str:
        """Apply personality traits to modify the response"""
        modified_response = response

        # Apply trait-based modifications
        if self.traits.get('extraversion', 0.5) > 0.7:
            modified_response += " I'm really excited to chat more about this!"
        elif self.traits.get('extraversion', 0.5) < 0.3:
            modified_response += " I tend to be more introspective about these things."

        if self.traits.get('openness', 0.5) > 0.7:
            modified_response += " I'm always open to exploring new ideas."
        elif self.traits.get('openness', 0.5) < 0.3:
            modified_response += " I prefer sticking to what I know well."

        return modified_response

    def _update_mood_from_interaction(self, prompt: str, response: str):
        """Update mood vector based on interaction"""
        # Simple mood updates based on prompt content
        if any(word in prompt.lower() for word in ['happy', 'great', 'awesome', 'love']):
            self.mood_vector['happiness'] = min(1.0, self.mood_vector['happiness'] + 0.1)
            self.mood_vector['energy'] = min(1.0, self.mood_vector['energy'] + 0.05)
        elif any(word in prompt.lower() for word in ['sad', 'angry', 'frustrated', 'hate']):
            self.mood_vector['happiness'] = max(0.0, self.mood_vector['happiness'] - 0.1)
            self.mood_vector['calmness'] = max(0.0, self.mood_vector['calmness'] - 0.05)

        # Natural mood decay
        self.mood_decay()
        self._save_data()

    def mood_decay(self):
        """Gradually return mood to baseline"""
        for key in self.mood_vector:
            if self.mood_vector[key] > 0.5:
                self.mood_vector[key] = max(0.5, self.mood_vector[key] - 0.01)
            elif self.mood_vector[key] < 0.5:
                self.mood_vector[key] = min(0.5, self.mood_vector[key] + 0.01)
        self._save_data()

    def add_trigger(self, word: str, effect: str):
        """Add a trigger word and its effect"""
        self.triggers[word.lower()] = effect
        self._save_data()

    def remove_trigger(self, word: str):
        """Remove a trigger word"""
        if word.lower() in self.triggers:
            del self.triggers[word.lower()]
            self._save_data()

    def list_triggers(self) -> Dict[str, str]:
        """List all triggers"""
        return self.triggers.copy()

    def recall_glyph(self, query: str) -> str:
        """Recall a glyph based on query"""
        # Simple glyph recall - in practice this would be more sophisticated
        for key, glyph in self.glyphs.items():
            if key.lower() in query.lower():
                return glyph
        return "No specific glyph found for that query."

    def fuse_archetypes(self, *archetypes: str):
        """Fuse multiple archetypes into current personality"""
        # Simple archetype fusion
        archetype_traits = {
            'creative': {'openness': 0.8, 'extraversion': 0.6},
            'analytical': {'conscientiousness': 0.8, 'openness': 0.7},
            'nurturing': {'agreeableness': 0.8, 'extraversion': 0.4},
            'adventurous': {'extraversion': 0.8, 'openness': 0.7},
            'balanced': {'openness': 0.5, 'conscientiousness': 0.5, 'extraversion': 0.5, 'agreeableness': 0.5, 'neuroticism': 0.5}
        }

        # Average traits from selected archetypes
        new_traits = {}
        for archetype in archetypes:
            if archetype in archetype_traits:
                for trait, value in archetype_traits[archetype].items():
                    if trait not in new_traits:
                        new_traits[trait] = []
                    new_traits[trait].append(value)

        # Calculate averages
        for trait, values in new_traits.items():
            self.traits[trait] = sum(values) / len(values)

        self.archetype = ', '.join(archetypes)
        self._save_data()

    def update_trait(self, trait: str, score: float):
        """Update a specific personality trait"""
        if trait in self.traits:
            self.traits[trait] = max(0.0, min(1.0, score))
            self._save_data()

    def get_trait_score(self, trait: str) -> Optional[float]:
        """Get the score for a specific trait"""
        return self.traits.get(trait)
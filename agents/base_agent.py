"""
Base Agent class for AGI system.
Handles conversational AI using GPT-2 and FLAN.
"""
from transformers import pipeline

class BaseAgent:
    def __init__(self, model_name="gpt2"):
        self.model_name = model_name
        self.pipeline = pipeline("text-generation", model=model_name)
        # Placeholder for memory and personality modules
        self.memory = None
        self.personality = None

    def set_memory(self, memory):
        self.memory = memory

    def set_personality(self, personality):
        self.personality = personality

    def respond(self, prompt, user_id=None):
        # Use memory and personality if available
        context = ""
        if self.memory and user_id:
            context += self.memory.get_user_context(user_id)
        if self.personality:
            context += self.personality.get_personality_context()
        full_prompt = context + prompt
        response = self.pipeline(full_prompt, max_length=100, num_return_sequences=1)
        return response[0]["generated_text"]

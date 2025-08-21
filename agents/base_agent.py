"""
[Memory]: This module orchestrates agent context, memory, and personality for AGI responses.
Base Agent class for AGI system.
Handles conversational AI using GPT-2 and FLAN.
"""


import requests
try:
    from transformers import pipeline
except ImportError:
    pipeline = None



class BaseAgent:
    def __init__(self, model_name="gpt2", use_http=True):
        self.model_name = model_name
        self.use_http = use_http
        if not use_http and pipeline is not None:
            self.pipeline = pipeline("text-generation", model=model_name)
        else:
            self.pipeline = None
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
        if self.use_http:
            try:
                r = requests.post("http://localhost:8000/text-generation", json={"prompt": full_prompt})
                if r.ok:
                    return r.json().get("generated_text", "")
                else:
                    return f"[HTTP error: {r.status_code}]"
            except Exception as e:
                return f"[HTTP error: {e}]"
        elif self.pipeline:
            response = self.pipeline(full_prompt, max_length=100, num_return_sequences=1)
            return response[0]["generated_text"]
        else:
            return "[No model available for response]"

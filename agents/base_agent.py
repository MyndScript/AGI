"""
[Memory]: This module orchestrates agent context, memory, and personality for AGI responses.
Base Agent class for AGI system.
Handles conversational AI using GPT-2 and FLAN.
"""


import requests
import os
from typing import Optional
from dotenv import load_dotenv
load_dotenv()
try:
    from transformers import pipeline
except ImportError:
    pipeline = None

import wikipedia


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
    """Query Wikipedia for a summary about a topic."""
    try:
        summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True)
        return {"summary": summary}
    except Exception as e:
        return {"error": str(e)}

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
    def __init__(self, model_name="gpt2", use_http=True):
        # Run Codacy analysis on this file after edits
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




    def fetch_external_facts(self, query):
        """Fuse facts from Wikipedia, Wikidata, and openFDA (only for medical topics)."""
        facts = {}
        facts["wikidata"] = query_wikidata(query)
        facts["wikipedia"] = query_wikipedia(query)
        # Only call openFDA if query is about medication, sickness, or side effects
        medical_keywords = ["medication", "drug", "medicine", "side effect", "sickness", "illness", "prescription", "adverse event", "symptom"]
        if any(word in query.lower() for word in medical_keywords):
            api_key = os.getenv("OPENFDA_API_KEY")
            facts["openfda_drug_event"] = query_openfda_drug_event(query, api_key=api_key)
        return facts

    def build_prompt(self, context):
        # Build a prompt string from context dict
        prompt = f"Traits: {context['traits']}\nMood: {context['mood']}\nMemory: {context['memory']}\nFacts: {context['facts']}\nUser: {context['query']}"
        return prompt

    def personalize_response(self, response, personality):
        # TODO: Adjust response tone/style based on personality
        return response

    def respond(self, prompt, user_id=None):
        # 1. Load user context
        traits = self.personality.traits if self.personality else {}
        mood = self.personality.mood_vector if self.personality else {}
        memory = self.memory.get(user_id) if self.memory else {}

        # 2. Fetch external knowledge
        facts = self.fetch_external_facts(prompt)

        # 3. Fuse context
        context = {
            "traits": traits,
            "mood": mood,
            "memory": memory,
            "facts": facts,
            "query": prompt
        }

        # 4. Generate response
        full_prompt = self.build_prompt(context)
        if self.pipeline:
            result = self.pipeline(full_prompt, max_length=100, num_return_sequences=1)
            response = result[0]["generated_text"]
        else:
            # TODO: Implement HTTP or other model call
            response = "[Model response placeholder]"

        # 5. Post-process
        response = self.personalize_response(response, self.personality)

        # 6. Learn/update
        if self.personality:
            self.personality.add_interaction({"prompt": prompt, "response": response})
        if self.memory is not None and user_id is not None:
            self.memory[user_id] = memory  # Extend with actual updates as needed

        return response

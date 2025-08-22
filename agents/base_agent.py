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
    def __init__(self, model_name="gpt2", use_http=True, available_models=None):
        """
        Initialize BaseAgent with model ensemble and context/threading scaffolding.
        """
        self.model_name = model_name
        self.use_http = use_http
        self.available_models = available_models or ["gpt2", "flan-t5-base"]
        self.model_pipelines = {}
        if pipeline is not None:
            for m in self.available_models:
                try:
                    self.model_pipelines[m] = pipeline("text-generation", model=m)
                except Exception:
                    self.model_pipelines[m] = None
        # Placeholder for memory and personality modules
        self.memory = None
        self.personality = None
        # External knowledge cache
        self.knowledge_cache = {}
        self.cache_expiry = 600  # seconds
        # Conversational threading
        self.topic_stack = []
        self.thread_context = {}


    def set_memory(self, memory):
        self.memory = memory


    def set_personality(self, personality):
        self.personality = personality




    def _cache_key(self, *args):
        return hashlib.sha256("|".join(map(str, args)).encode()).hexdigest()

    def fetch_external_facts(self, query):
        """Fuse facts from Wikipedia, Wikidata, and openFDA (only for medical topics), with caching."""
        facts = {}
        cache_key = self._cache_key("external_facts", query)
        now = time.time()
        cached = self.knowledge_cache.get(cache_key)
        if cached and now - cached["ts"] < self.cache_expiry:
            return cached["facts"]
        facts["wikidata"] = query_wikidata(query)
        facts["wikipedia"] = query_wikipedia(query)
        # Only call openFDA if query is about medication, sickness, or side effects
        medical_keywords = ["medication", "drug", "medicine", "side effect", "sickness", "illness", "prescription", "adverse event", "symptom"]
        if any(word in query.lower() for word in medical_keywords):
            api_key = os.getenv("OPENFDA_API_KEY")
            facts["openfda_drug_event"] = query_openfda_drug_event(query, api_key=api_key)
        # TODO: Add support for news, weather, books, movies APIs
        self.knowledge_cache[cache_key] = {"facts": facts, "ts": now}
        return facts

    def _weight_context(self, context):
        """
        Weight context sources by recency, emotional importance, and optionally embeddings.
        Returns a dict of weighted context.
        """
        weights = {
            "traits": 1.0,
            "mood": 1.0,
            "memory": 1.0,
            "facts": 1.0,
            "query": 1.0
        }
        # Example: boost recent memory, emotional spikes
        if context.get("mood") and max(context["mood"].values(), default=0) > 0.7:
            weights["mood"] = 1.5
        if context.get("memory") and isinstance(context["memory"], dict):
            if context["memory"].get("recent"):
                weights["memory"] = 1.3
        # TODO: Use embeddings for semantic fusion
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
        # TODO: Replace with embedding-based fusion for richer context
        return prompt

    def personalize_response(self, response, personality, style=None, sentiment=None):
        # Expand to adjust response tone/style/humor/mimicry/sentiment
        # Example: adjust for style/sentiment
        if style == "formal":
            response = "[Formal] " + response
        elif style == "informal":
            response = "[Casual] " + response
        if sentiment == "positive":
            response += " 😊"
        elif sentiment == "negative":
            response += " 😔"
        # TODO: Implement mimicry, humor, advanced sentiment
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
            if user_id not in self.memory:
                self.memory[user_id] = {}
            self.memory[user_id]["feedback"] = self.memory[user_id].get("feedback", []) + [feedback]
        # Optionally update personality
        if self.personality:
            self.personality.update_traits({"last_feedback": feedback})

    def select_model(self, prompt, traits=None, mood=None):
        """
        Select model based on task, user preference, or detected mood.
        """
        # Example: Use FLAN for reasoning, GPT-2 for creative chat
        if traits and traits.get("prefers_reasoning"):
            return "flan-t5-base" if "flan-t5-base" in self.model_pipelines else self.model_name
        if mood and mood.get("creativity", 0) > 0.7:
            return "gpt2" if "gpt2" in self.model_pipelines else self.model_name
        # TODO: Add more selection logic
        return self.model_name

    def respond(self, prompt, user_id=None, style=None, sentiment=None):
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

        # 4. Model selection
        selected_model = self.select_model(prompt, traits, mood)
        pipeline_obj = self.model_pipelines.get(selected_model)

        # 5. Generate response
        full_prompt = self.build_prompt(context)
        if pipeline_obj:
            result = pipeline_obj(full_prompt, max_length=100, num_return_sequences=1)
            response = result[0]["generated_text"]
        else:
            # TODO: Implement HTTP or other model call
            response = "[Model response placeholder]"

        # 6. Post-process
        response = self.personalize_response(response, self.personality, style=style, sentiment=sentiment)

        # 7. Learn/update
        if self.personality:
            self.personality.add_interaction({"prompt": prompt, "response": response})
        if self.memory is not None and user_id is not None:
            self.memory[user_id] = memory  # Extend with actual updates as needed

        # 8. Threading: update topic stack
        self.topic_stack.append({"prompt": prompt, "response": response, "ts": time.time()})
        if len(self.topic_stack) > 20:
            self.topic_stack = self.topic_stack[-20:]

        return response

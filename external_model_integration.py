#!/usr/bin/env python3
"""
External Model Integration with Personality Adaptation
Demonstrates how to wrap external AI models with our personality engine
"""

import requests
import json
import time
from typing import Dict, Any, Optional

class ExternalModelWrapper:
    def __init__(self, model_config: Dict[str, Any]):
        """
        Initialize wrapper for external model with personality adaptation

        Args:
            model_config: Configuration for the external model
                {
                    "type": "openai" | "anthropic" | "ollama" | "local",
                    "api_url": "https://api.openai.com/v1",
                    "api_key": "your-api-key",
                    "model_name": "gpt-3.5-turbo",
                    "personality_server_url": "http://localhost:8002"
                }
        """
        self.config = model_config
        self.personality_url = model_config.get("personality_server_url", "http://localhost:8002")

    def generate_response(self, message: str, user_id: str = "default",
                         context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate response using external model, then adapt with personality

        Args:
            message: User message
            user_id: User identifier for personality tracking
            context: Additional context information

        Returns:
            Dictionary with response, personality info, and metadata
        """
        # Step 1: Get raw response from external model
        raw_response = self._call_external_model(message, context)

        # Step 2: Adapt response with our personality engine
        adapted_response = self._adapt_with_personality(
            raw_response, message, user_id, context
        )

        return {
            "response": adapted_response["response"],
            "personality_context": adapted_response.get("personality_context", {}),
            "raw_response": raw_response,
            "adaptation_metadata": {
                "external_model": self.config.get("model_name"),
                "personality_adapted": True,
                "timestamp": time.time()
            }
        }

    def _call_external_model(self, message: str, context: Optional[Dict] = None) -> str:
        """Call the external model API"""
        model_type = self.config.get("type", "openai")

        if model_type == "openai":
            return self._call_openai_api(message, context)
        elif model_type == "anthropic":
            return self._call_anthropic_api(message, context)
        elif model_type == "ollama":
            return self._call_ollama_api(message, context)
        elif model_type == "local":
            return self._call_local_model(message, context)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def _call_openai_api(self, message: str, context: Optional[Dict] = None) -> str:
        """Call OpenAI API"""
        headers = {
            "Authorization": f"Bearer {self.config.get('api_key')}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.config.get("model_name", "gpt-3.5-turbo"),
            "messages": [{"role": "user", "content": message}],
            "max_tokens": 150,
            "temperature": 0.7
        }

        try:
            resp = requests.post(
                f"{self.config.get('api_url', 'https://api.openai.com/v1')}/chat/completions",
                headers=headers,
                json=payload,
                timeout=30
            )

            if resp.ok:
                data = resp.json()
                return data.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                return f"Error: {resp.status_code} - {resp.text}"

        except Exception as e:
            return f"Error calling OpenAI API: {str(e)}"

    def _call_anthropic_api(self, message: str, context: Optional[Dict] = None) -> str:
        """Call Anthropic Claude API"""
        headers = {
            "x-api-key": self.config.get("api_key"),
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }

        payload = {
            "model": self.config.get("model_name", "claude-3-sonnet-20240229"),
            "max_tokens": 150,
            "messages": [{"role": "user", "content": message}]
        }

        try:
            resp = requests.post(
                f"{self.config.get('api_url', 'https://api.anthropic.com')}/v1/messages",
                headers=headers,
                json=payload,
                timeout=30
            )

            if resp.ok:
                data = resp.json()
                return data.get("content", [{}])[0].get("text", "")
            else:
                return f"Error: {resp.status_code} - {resp.text}"

        except Exception as e:
            return f"Error calling Anthropic API: {str(e)}"

    def _call_ollama_api(self, message: str, context: Optional[Dict] = None) -> str:
        """Call Ollama API"""
        payload = {
            "model": self.config.get("model_name", "llama3.2:1b"),
            "prompt": message,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "max_tokens": 150
            }
        }

        try:
            resp = requests.post(
                f"{self.config.get('api_url', 'http://localhost:11434')}/api/generate",
                json=payload,
                timeout=30
            )

            if resp.ok:
                data = resp.json()
                return data.get("response", "").strip()
            else:
                return f"Error: {resp.status_code} - {resp.text}"

        except Exception as e:
            return f"Error calling Ollama API: {str(e)}"

    def _call_local_model(self, message: str, context: Optional[Dict] = None) -> str:
        """Call local model (placeholder for custom implementations)"""
        # This could be a local quantized model, Triton server, etc.
        return f"Local model response to: {message[:50]}..."

    def _adapt_with_personality(self, raw_response: str, original_message: str,
                               user_id: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """Adapt external model response with our personality engine"""
        try:
            payload = {
                "user_id": user_id,
                "message": original_message,
                "context": {
                    "external_response": raw_response,
                    "model_type": self.config.get("type"),
                    **(context or {})
                }
            }

            resp = requests.post(
                f"{self.personality_url}/personalized-response",
                json=payload,
                timeout=10
            )

            if resp.ok:
                return resp.json()
            else:
                # Fallback: return raw response if personality server fails
                return {
                    "status": "fallback",
                    "response": raw_response,
                    "personality_context": {
                        "archetype": "adaptive_sage",
                        "confidence": 0.5,
                        "mood": {"trust": 0.5},
                        "response_style": "balanced"
                    }
                }

        except Exception as e:
            # Fallback: return raw response if personality adaptation fails
            return {
                "status": "fallback",
                "response": raw_response,
                "personality_context": {
                    "archetype": "adaptive_sage",
                    "confidence": 0.5,
                    "mood": {"trust": 0.5},
                    "response_style": "balanced"
                },
                "error": str(e)
            }

# Example usage and configuration
def create_openai_wrapper():
    """Create wrapper for OpenAI GPT models"""
    config = {
        "type": "openai",
        "api_url": "https://api.openai.com/v1",
        "api_key": "your-openai-api-key-here",
        "model_name": "gpt-3.5-turbo",
        "personality_server_url": "http://localhost:8002"
    }
    return ExternalModelWrapper(config)

def create_anthropic_wrapper():
    """Create wrapper for Anthropic Claude models"""
    config = {
        "type": "anthropic",
        "api_url": "https://api.anthropic.com",
        "api_key": "your-anthropic-api-key-here",
        "model_name": "claude-3-sonnet-20240229",
        "personality_server_url": "http://localhost:8002"
    }
    return ExternalModelWrapper(config)

def create_ollama_wrapper():
    """Create wrapper for local Ollama models"""
    config = {
        "type": "ollama",
        "api_url": "http://localhost:11434",
        "model_name": "llama3.2:1b",
        "personality_server_url": "http://localhost:8002"
    }
    return ExternalModelWrapper(config)

if __name__ == "__main__":
    # Example: Test with Ollama (no API key needed)
    print("🔧 Testing External Model Integration with Personality Adaptation")
    print("=" * 60)

    # Create Ollama wrapper (works with local models)
    wrapper = create_ollama_wrapper()

    test_messages = [
        "Hello, how are you today?",
        "What is the meaning of life?",
        "Tell me about artificial intelligence"
    ]

    for message in test_messages:
        print(f"\n📝 User: {message}")

        try:
            result = wrapper.generate_response(message, user_id="test_user")

            print(f"🤖 Adapted Response: {result['response']}")
            print(f"🧠 Personality: {result['personality_context'].get('archetype', 'unknown')}")
            print(f"📊 Confidence: {result['personality_context'].get('confidence', 0):.2f}")

        except Exception as e:
            print(f"❌ Error: {str(e)}")

    print("\n✅ Integration test complete!")
    print("\nTo use with OpenAI/Anthropic:")
    print("1. Set your API keys in the config")
    print("2. Change wrapper type to 'openai' or 'anthropic'")
    print("3. Ensure personality server is running on localhost:8002")

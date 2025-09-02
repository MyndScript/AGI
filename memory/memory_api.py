"""
AGI Memory API Client
Handles sophisticated conversation analysis and personality scoring
"""

import httpx
from typing import Dict, List, Optional, Any
from datetime import datetime
import re
from collections import defaultdict

class ConversationAnalyzer:
    """Advanced conversation analysis for personality insights"""

    def __init__(self):
        # Personality dimensions based on Big Five + additional traits
        self.personality_dimensions = {
            'openness': ['curious', 'imaginative', 'creative', 'open-minded', 'adventurous'],
            'conscientiousness': ['organized', 'responsible', 'reliable', 'disciplined', 'thorough'],
            'extraversion': ['outgoing', 'energetic', 'talkative', 'social', 'enthusiastic'],
            'agreeableness': ['kind', 'cooperative', 'empathetic', 'helpful', 'considerate'],
            'neuroticism': ['anxious', 'sensitive', 'emotional', 'worried', 'tense'],
            'intellect': ['analytical', 'logical', 'thoughtful', 'reflective', 'philosophical'],
            'creativity': ['innovative', 'artistic', 'original', 'visionary', 'inspired'],
            'empathy': ['understanding', 'compassionate', 'supportive', 'caring', 'attuned']
        }

        # Emotional patterns
        self.emotion_patterns = {
            'joy': ['happy', 'excited', 'delighted', 'thrilled', 'wonderful', 'amazing'],
            'sadness': ['sad', 'depressed', 'unhappy', 'disappointed', 'heartbroken', 'miserable'],
            'anger': ['angry', 'frustrated', 'irritated', 'furious', 'annoyed', 'outraged'],
            'fear': ['scared', 'afraid', 'anxious', 'worried', 'nervous', 'terrified'],
            'surprise': ['shocked', 'amazed', 'astonished', 'unexpected', 'surprising'],
            'trust': ['confident', 'reliable', 'faithful', 'loyal', 'dependable'],
            'anticipation': ['excited', 'eager', 'hopeful', 'optimistic', 'enthusiastic']
        }

        # Communication styles
        self.communication_styles = {
            'direct': ['clearly', 'directly', 'straightforward', 'honestly', 'bluntly'],
            'diplomatic': ['carefully', 'tactfully', 'politely', 'respectfully', 'considerately'],
            'analytical': ['logically', 'rationally', 'objectively', 'systematically', 'methodically'],
            'emotional': ['feelingly', 'passionately', 'emotionally', 'intensely', 'deeply'],
            'narrative': ['story', 'experience', 'journey', 'tale', 'adventure']
        }

    def analyze_conversation(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Comprehensive conversation analysis for personality insights"""

        if not conversation_history:
            return self._default_personality_profile()

        # Extract text content
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        assistant_messages = [msg for msg in conversation_history if msg.get('sender') == 'assistant']

        user_text = ' '.join([msg.get('text', '') for msg in user_messages])
        assistant_text = ' '.join([msg.get('text', '') for msg in assistant_messages])

        # Analyze personality traits
        personality_scores = self._analyze_personality_traits(user_text)

        # Analyze emotional patterns
        emotional_profile = self._analyze_emotional_patterns(user_text)

        # Analyze communication style
        communication_style = self._analyze_communication_style(user_text)

        # Analyze cognitive patterns
        cognitive_patterns = self._analyze_cognitive_patterns(user_text)

        # Analyze temporal patterns
        temporal_patterns = self._analyze_temporal_patterns(conversation_history)

        # Calculate engagement metrics
        engagement_metrics = self._calculate_engagement_metrics(conversation_history)

        return {
            'personality_scores': personality_scores,
            'emotional_profile': emotional_profile,
            'communication_style': communication_style,
            'cognitive_patterns': cognitive_patterns,
            'temporal_patterns': temporal_patterns,
            'engagement_metrics': engagement_metrics,
            'analysis_timestamp': datetime.now().isoformat(),
            'conversation_count': len(conversation_history),
            'user_message_count': len(user_messages),
            'assistant_message_count': len(assistant_messages)
        }

    def _analyze_personality_traits(self, text: str) -> Dict[str, float]:
        """Analyze Big Five personality traits + additional dimensions"""
        text_lower = text.lower()
        scores = {}

        for trait, keywords in self.personality_dimensions.items():
            # Count keyword matches
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            # Normalize by text length (words per 1000 words)
            word_count = len(text.split())
            if word_count > 0:
                scores[trait] = min(1.0, (matches / word_count) * 1000)
            else:
                scores[trait] = 0.5  # Default neutral score

        return scores

    def _analyze_emotional_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze emotional expression patterns"""
        text_lower = text.lower()
        emotion_scores = {}

        for emotion, keywords in self.emotion_patterns.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            word_count = len(text.split())
            if word_count > 0:
                emotion_scores[emotion] = min(1.0, (matches / word_count) * 1000)
            else:
                emotion_scores[emotion] = 0.0

        # Calculate emotional stability (inverse of emotional variance)
        emotion_values = list(emotion_scores.values())
        if emotion_values:
            emotional_variance = sum((x - sum(emotion_values)/len(emotion_values))**2 for x in emotion_values) / len(emotion_values)
            emotional_stability = 1.0 - min(1.0, emotional_variance)
        else:
            emotional_stability = 0.5

        return {
            'emotion_scores': emotion_scores,
            'emotional_stability': emotional_stability,
            'dominant_emotion': max(emotion_scores.items(), key=lambda x: x[1])[0] if emotion_scores else 'neutral'
        }

    def _analyze_communication_style(self, text: str) -> Dict[str, float]:
        """Analyze communication style preferences"""
        text_lower = text.lower()
        style_scores = {}

        for style, keywords in self.communication_styles.items():
            matches = sum(1 for keyword in keywords if keyword in text_lower)
            word_count = len(text.split())
            if word_count > 0:
                style_scores[style] = min(1.0, (matches / word_count) * 1000)
            else:
                style_scores[style] = 0.0

        return style_scores

    def _analyze_cognitive_patterns(self, text: str) -> Dict[str, Any]:
        """Analyze cognitive processing patterns"""
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]

        # Question patterns
        question_count = text.count('?')
        total_sentences = len(sentences)

        # Complexity metrics
        avg_sentence_length = sum(len(s.split()) for s in sentences) / max(1, total_sentences)
        unique_words = len(set(text.lower().split()))
        total_words = len(text.split())
        lexical_diversity = unique_words / max(1, total_words)

        # Reasoning patterns
        reasoning_indicators = ['because', 'therefore', 'however', 'although', 'since', 'due to']
        reasoning_score = sum(1 for word in reasoning_indicators if word in text.lower())

        return {
            'avg_sentence_length': avg_sentence_length,
            'lexical_diversity': lexical_diversity,
            'question_ratio': question_count / max(1, total_sentences),
            'reasoning_score': reasoning_score,
            'sentence_count': total_sentences
        }

    def _analyze_temporal_patterns(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal conversation patterns"""
        if not conversation_history:
            return {'session_duration': 0, 'message_frequency': 0, 'response_times': []}

        timestamps = self._extract_timestamps(conversation_history)
        if len(timestamps) < 2:
            return {'session_duration': 0, 'message_frequency': 0, 'response_times': []}

        session_duration = self._calculate_session_duration(timestamps)
        message_frequency = self._calculate_message_frequency(len(timestamps), session_duration)
        response_times = self._calculate_response_times(conversation_history, timestamps)

        return {
            'session_duration': session_duration,
            'message_frequency': message_frequency,
            'avg_response_time': sum(response_times) / max(1, len(response_times)),
            'response_times': response_times
        }

    def _extract_timestamps(self, conversation_history: List[Dict]) -> List[datetime]:
        """Extract valid timestamps from conversation history"""
        timestamps = []
        for msg in conversation_history:
            if 'timestamp' not in msg:
                continue
            
            try:
                if isinstance(msg['timestamp'], str):
                    timestamps.append(datetime.fromisoformat(msg['timestamp']))
                elif isinstance(msg['timestamp'], (int, float)):
                    timestamps.append(datetime.fromtimestamp(msg['timestamp']))
            except (ValueError, TypeError, OSError):
                continue
        return timestamps

    def _calculate_session_duration(self, timestamps: List[datetime]) -> float:
        """Calculate session duration in seconds"""
        if len(timestamps) < 2:
            return 0
        return (max(timestamps) - min(timestamps)).total_seconds()

    def _calculate_message_frequency(self, message_count: int, session_duration: float) -> float:
        """Calculate message frequency (messages per hour)"""
        return message_count / max(1, session_duration / 3600)

    def _calculate_response_times(self, conversation_history: List[Dict], timestamps: List[datetime]) -> List[float]:
        """Calculate response times for assistant responses"""
        response_times = []
        user_timestamps = [t for msg, t in zip(conversation_history, timestamps) if msg.get('sender') == 'user']
        assistant_timestamps = [t for msg, t in zip(conversation_history, timestamps) if msg.get('sender') == 'assistant']

        for i in range(min(len(user_timestamps), len(assistant_timestamps))):
            if i < len(assistant_timestamps) and i < len(user_timestamps):
                response_time = (assistant_timestamps[i] - user_timestamps[i]).total_seconds()
                if 0 < response_time < 300:  # Reasonable response time
                    response_times.append(response_time)

        return response_times

    def _calculate_engagement_metrics(self, conversation_history: List[Dict]) -> Dict[str, Any]:
        """Calculate user engagement metrics"""
        user_messages = [msg for msg in conversation_history if msg.get('sender') == 'user']
        assistant_messages = [msg for msg in conversation_history if msg.get('sender') == 'assistant']

        # Message length analysis
        user_lengths = [len(msg.get('text', '').split()) for msg in user_messages]
        assistant_lengths = [len(msg.get('text', '').split()) for msg in assistant_messages]

        # Conversation depth (back-and-forth exchanges)
        conversation_depth = min(len(user_messages), len(assistant_messages))

        # Topic persistence (simplified)
        user_texts = [msg.get('text', '') for msg in user_messages]
        topic_persistence = len(set(user_texts)) / max(1, len(user_texts))  # Unique messages ratio

        return {
            'avg_user_message_length': sum(user_lengths) / max(1, len(user_lengths)),
            'avg_assistant_message_length': sum(assistant_lengths) / max(1, len(assistant_lengths)),
            'conversation_depth': conversation_depth,
            'topic_persistence': topic_persistence,
            'total_exchanges': len(conversation_history)
        }

    def _default_personality_profile(self) -> Dict[str, Any]:
        """Return default personality profile"""
        return {
            'personality_scores': {trait: 0.5 for trait in self.personality_dimensions.keys()},
            'emotional_profile': {
                'emotion_scores': {emotion: 0.0 for emotion in self.emotion_patterns.keys()},
                'emotional_stability': 0.5,
                'dominant_emotion': 'neutral'
            },
            'communication_style': {style: 0.0 for style in self.communication_styles.keys()},
            'cognitive_patterns': {
                'avg_sentence_length': 15.0,
                'lexical_diversity': 0.5,
                'question_ratio': 0.1,
                'reasoning_score': 0,
                'sentence_count': 0
            },
            'temporal_patterns': {
                'session_duration': 0,
                'message_frequency': 0,
                'avg_response_time': 0,
                'response_times': []
            },
            'engagement_metrics': {
                'avg_user_message_length': 10,
                'avg_assistant_message_length': 20,
                'conversation_depth': 0,
                'topic_persistence': 0,
                'total_exchanges': 0
            },
            'analysis_timestamp': datetime.now().isoformat(),
            'conversation_count': 0,
            'user_message_count': 0,
            'assistant_message_count': 0
        }


class MemoryAPIClient:
    """Production-ready memory API client with advanced personality analysis"""

    def __init__(self, base_url: str = "http://localhost:8001"):
        self.base_url = base_url
        self.client = httpx.Client(timeout=30.0)
        self.analyzer = ConversationAnalyzer()

        # Batch processing settings
        self.batch_size = 10
        self.batch_timeout = 300  # 5 minutes
        self.pending_updates = defaultdict(list)

    def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context including personality analysis"""
        try:
            # Get basic memory context
            response = self.client.post(f"{self.base_url}/user_context",
                                      json={"user_id": user_id})
            response.raise_for_status()
            basic_context = response.json()

            # Get personality data
            personality_response = self.client.post(f"{self.base_url}/get-personality",
                                                  json={"user_id": user_id})
            personality_data = {}
            if personality_response.status_code == 200:
                personality_data = personality_response.json()

            # Get conversation history for analysis
            conversation_history = self._get_conversation_history(user_id)

            # Perform advanced analysis
            analysis = self.analyzer.analyze_conversation(conversation_history)

            return {
                'basic_context': basic_context.get('context', ''),
                'personality_data': personality_data,
                'advanced_analysis': analysis,
                'user_id': user_id,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            print(f"Error getting user context: {e}")
            return {
                'basic_context': '',
                'personality_data': {},
                'advanced_analysis': self.analyzer._default_personality_profile(),
                'user_id': user_id,
                'error': str(e)
            }

    def _get_conversation_history(self, user_id: str, limit: int = 50) -> List[Dict]:
        """Get recent conversation history for analysis"""
        try:
            # This would need to be implemented based on your storage system
            # For now, return empty list
            return []
        except Exception as e:
            print(f"Error getting conversation history: {e}")
            return []

    def update_personality_batch(self, user_id: str, conversation_data: List[Dict]):
        """Batch update personality analysis to reduce PostgreSQL calls"""
        self.pending_updates[user_id].extend(conversation_data)

        # Process batch if threshold reached
        if len(self.pending_updates[user_id]) >= self.batch_size:
            self._process_personality_batch(user_id)

    def _process_personality_batch(self, user_id: str):
        """Process accumulated personality updates"""
        if not self.pending_updates[user_id]:
            return

        try:
            # Combine all pending updates
            all_conversations = self.pending_updates[user_id]

            # Perform comprehensive analysis
            analysis = self.analyzer.analyze_conversation(all_conversations)

            # Update PostgreSQL with aggregated insights
            self._update_global_personality_matrix(user_id, analysis)

            # Clear processed updates
            self.pending_updates[user_id].clear()

        except Exception as e:
            print(f"Error processing personality batch: {e}")

    def _update_global_personality_matrix(self, user_id: str, analysis: Dict):
        """Update global personality learning matrix"""
        try:
            # Extract key personality scores
            personality_scores = analysis.get('personality_scores', {})

            # Send to PostgreSQL for global learning
            for trait, score in personality_scores.items():
                payload = {
                    'user_id': user_id,
                    'trait': trait,
                    'score': score,
                    'analysis_data': analysis
                }

                response = self.client.post(f"{self.base_url}/update-personality-trait",
                                          json=payload)

                if response.status_code != 200:
                    print(f"Failed to update personality trait {trait}: {response.text}")

        except Exception as e:
            print(f"Error updating global personality matrix: {e}")

    def get_similar_users(self, user_id: str, trait: Optional[str] = None, limit: int = 5) -> List[Dict]:
        """Find users with similar personality profiles for collaborative learning"""
        try:
            payload = {
                'user_id': user_id,
                'trait': trait,
                'limit': limit
            }

            response = self.client.post(f"{self.base_url}/find-similar-users",
                                      json=payload)
            response.raise_for_status()

            return response.json().get('similar_users', [])

        except Exception as e:
            print(f"Error finding similar users: {e}")
            return []

    def get_global_insights(self, trait: Optional[str] = None) -> Dict[str, Any]:
        """Get global personality insights and trends"""
        try:
            payload = {'trait': trait} if trait else {}

            response = self.client.post(f"{self.base_url}/global-personality-insights",
                                      json=payload)
            response.raise_for_status()

            return response.json()

        except Exception as e:
            print(f"Error getting global insights: {e}")
            return {'error': str(e)}

    def store_memory(self, user_id: str, key: str, value: str) -> bool:
        """Store user-specific memory in SQLite"""
        try:
            response = self.client.post(f"{self.base_url}/set-memory",
                                      json={"user_id": user_id, "key": key, "value": value})
            response.raise_for_status()
            return response.json().get('success', False)
        except Exception as e:
            print(f"Error storing memory: {e}")
            return False

    def retrieve_memory(self, user_id: str, key: str) -> str:
        """Retrieve user-specific memory from SQLite"""
        try:
            response = self.client.post(f"{self.base_url}/get-memory",
                                      json={"user_id": user_id, "key": key})
            response.raise_for_status()
            return response.json().get('value', '')
        except Exception as e:
            print(f"Error retrieving memory: {e}")
            return ""

    def add_conversation_context(self, user_id: str, message: str, response: str):
        """Add conversation context for personality analysis"""
        conversation_data = {
            'user_message': message,
            'assistant_response': response,
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id
        }

        # Add to batch for processing
        self.update_personality_batch(user_id, [conversation_data])

    def store_post(self, user_id: str, post_id: str, content: str, timestamp: int, tags: List[str]) -> bool:
        """Store a Facebook post in memory"""
        try:
            payload: Dict[str, Any] = {
                "user_id": user_id,
                "post_id": post_id,
                "content": content,
                "timestamp": timestamp,
                "tags": tags
            }
            print(f"[DEBUG] Storing post payload: {payload}")
            response = self.client.post(f"{self.base_url}/store-post", json=payload)
            print(f"[DEBUG] Store post response status: {response.status_code}")
            if response.status_code != 200:
                print(f"[DEBUG] Store post response text: {response.text}")
            response.raise_for_status()
            return response.json().get('success', False)
        except Exception as e:
            print(f"Error storing post: {e}")
            return False

    def get_posts(self, user_id: str, tags: Optional[List[str]] = None, since: Optional[int] = None, until: Optional[int] = None) -> List[Dict]:
        """Get posts from memory"""
        try:
            payload: Dict[str, Any] = {"user_id": user_id}
            if tags:
                payload["tags"] = tags
            if since:
                payload["since"] = since
            if until:
                payload["until"] = until

            response = self.client.post(f"{self.base_url}/get-posts", json=payload)
            response.raise_for_status()
            return response.json().get('posts', [])
        except Exception as e:
            print(f"Error getting posts: {e}")
            return []

    def store_journal_entry(self, user_id: str, journal_data: Dict[str, Any]) -> bool:
        """Store a journal entry in memory"""
        try:
            payload = {
                "user_id": user_id,
                "journal_data": journal_data
            }
            
            response = self.client.post(f"{self.base_url}/store-journal-entry", json=payload)
            response.raise_for_status()
            return response.json().get('success', False)
        except Exception as e:
            print(f"Error storing journal entry: {e}")
            return False

    def get_journal_entries(self, user_id: str, limit: int = 10) -> List[Dict]:
        """Get journal entries from memory"""
        try:
            payload = {
                "user_id": user_id,
                "limit": limit
            }
            
            response = self.client.post(f"{self.base_url}/get-journal-entries", json=payload)
            response.raise_for_status()
            return response.json().get('entries', [])
        except Exception as e:
            print(f"Error getting journal entries: {e}")
            return []
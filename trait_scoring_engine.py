#!/usr/bin/env python3
"""
🧬 Advanced Trait Scoring Engine & Glyph Response Modulator
Extends the personality analyzer with sophisticated trait inference and response modulation
"""

import numpy as np
import requests
from datetime import datetime
from typing import Dict, List, Any
import re
import logging

logger = logging.getLogger(__name__)

class TraitScoringEngine:
    """Advanced trait scoring based on embedding patterns and text analysis"""
    
    def __init__(self):
        # Trait detection patterns
        self.trait_indicators = self._get_trait_indicators()

    def _get_trait_indicators(self) -> Dict[str, Dict]:
        """Get trait detection patterns"""
        return {
            'curiosity_score': {
                'question_markers': ['?', 'why', 'how', 'what', 'when', 'where', 'wonder', 'curious'],
                'exploration_words': ['explore', 'discover', 'learn', 'understand', 'find out', 'investigate'],
                'weight': 1.0
            },
            'sarcasm_index': {
                'irony_markers': ['obviously', 'clearly', 'sure', 'right', 'great', 'perfect', 'wonderful'],
                'context_indicators': ['...', '!', 'oh really', 'of course', 'wow'],
                'punctuation_patterns': ['!!!', '...', '?!'],
                'weight': 0.8
            },
            'analytical_thinking': {
                'logic_words': ['because', 'therefore', 'however', 'analysis', 'conclusion', 'evidence'],
                'reasoning_patterns': ['first', 'second', 'then', 'consequently', 'thus', 'hence'],
                'structure_indicators': ['step', 'process', 'method', 'approach', 'strategy'],
                'weight': 1.2
            },
            'emotional_expressiveness': {
                'emotion_words': ['feel', 'feeling', 'emotion', 'heart', 'soul', 'passionate'],
                'intensity_markers': ['very', 'extremely', 'incredibly', 'absolutely', 'deeply'],
                'exclamation_usage': ['!', '!!', '!!!'],
                'weight': 1.0
            },
            'learning_preference': {
                'example_seeking': ['example', 'instance', 'case', 'sample', 'demonstration'],
                'theory_seeking': ['theory', 'principle', 'concept', 'framework', 'model'],
                'practice_seeking': ['practice', 'try', 'apply', 'implement', 'hands-on'],
                'weight': 0.9
            },
            'social_engagement': {
                'inclusive_language': ['we', 'us', 'our', 'together', 'collaborate', 'share'],
                'personal_sharing': ['my', 'me', 'myself', 'personal', 'experience', 'story'],
                'community_focus': ['community', 'group', 'team', 'everyone', 'all'],
                'weight': 1.1
            },
            'innovation_drive': {
                'creation_words': ['create', 'build', 'make', 'design', 'innovate', 'invent'],
                'future_focus': ['future', 'tomorrow', 'next', 'upcoming', 'vision', 'goal'],
                'change_orientation': ['change', 'transform', 'improve', 'revolutionize'],
                'weight': 1.0
            },
            'detail_orientation': {
                'precision_words': ['specific', 'exact', 'precise', 'detailed', 'thorough'],
                'qualification_markers': ['however', 'although', 'except', 'unless', 'but'],
                'elaboration_patterns': ['furthermore', 'moreover', 'additionally', 'specifically'],
                'weight': 1.0
            }
        }

    def analyze_text_traits(self, text: str) -> Dict[str, float]:
        """Analyze text for advanced personality traits"""
        if not text:
            return {}
        
        text_lower = text.lower()
        word_count = len(text.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', text) if s.strip()])
        
        trait_scores = {}
        
        for trait, indicators in self.trait_indicators.items():
            score = self._calculate_trait_score(text, text_lower, indicators)
            trait_scores[trait] = self._normalize_trait_score(score, word_count, indicators)
        
        # Special calculations for complex traits
        trait_scores.update(self._calculate_complex_traits(text, text_lower, word_count, sentence_count))
        
        return trait_scores

    def _calculate_trait_score(self, text: str, text_lower: str, indicators: Dict) -> tuple[float, int]:
        """Calculate raw score and total indicators for a trait"""
        score = 0.0
        total_indicators = 0
        
        for indicator_type, words in indicators.items():
            if indicator_type == 'weight':
                continue
                
            if indicator_type == 'punctuation_patterns':
                score, total_indicators = self._count_punctuation_patterns(text, words, score, total_indicators)
            elif indicator_type == 'exclamation_usage':
                score, total_indicators = self._count_exclamation_usage(text, words, score, total_indicators)
            else:
                score, total_indicators = self._count_word_occurrences(text_lower, words, score, total_indicators)
        
        return score, total_indicators

    def _count_punctuation_patterns(self, text: str, patterns: List[str], score: float, total: int) -> tuple[float, int]:
        """Count punctuation patterns in text"""
        for pattern in patterns:
            score += text.count(pattern) * 0.5
            total += 1
        return score, total

    def _count_exclamation_usage(self, text: str, patterns: List[str], score: float, total: int) -> tuple[float, int]:
        """Count exclamation usage patterns"""
        for pattern in patterns:
            score += text.count(pattern) * 0.3
            total += 1
        return score, total

    def _count_word_occurrences(self, text_lower: str, words: List[str], score: float, total: int) -> tuple[float, int]:
        """Count word occurrences in text"""
        for word in words:
            if word in text_lower:
                score += 1.0
                total += 1
        return score, total

    def _normalize_trait_score(self, score_data: tuple[float, int], word_count: int, indicators: Dict) -> float:
        """Normalize trait score based on text length and weight"""
        score, total_indicators = score_data
        
        if word_count > 0 and total_indicators > 0:
            normalized_score = (score / word_count) * 100  # per 100 words
            weighted_score = normalized_score * indicators.get('weight', 1.0)
            return min(1.0, weighted_score)
        return 0.0

    def _calculate_complex_traits(self, text: str, text_lower: str, word_count: int, sentence_count: int) -> Dict[str, float]:
        """Calculate complex trait metrics"""
        return {
            'question_density': text.count('?') / max(1, sentence_count),
            'avg_sentence_length': word_count / max(1, sentence_count),
            'lexical_diversity': len(set(text_lower.split())) / max(1, word_count)
        }

    def analyze_embedding_clusters(self, embeddings: List[Dict], user_id: str) -> Dict[str, Any]:
        """Analyze embedding patterns for deeper personality insights"""
        if len(embeddings) < 3:
            return {'cluster_analysis': 'insufficient_data'}
        
        time_groups = self._group_embeddings_by_time(embeddings)
        analysis = {}
        
        for period, period_embeddings in time_groups.items():
            if len(period_embeddings) < 2:
                continue
                
            period_analysis = self._analyze_period_embeddings(period_embeddings)
            analysis[period] = period_analysis
        
        return analysis

    def _group_embeddings_by_time(self, embeddings: List[Dict]) -> Dict[str, List[Dict]]:
        """Group embeddings by time periods"""
        now = datetime.now().timestamp()
        return {
            'recent': [e for e in embeddings if e['timestamp'] > now - (7 * 24 * 3600)],
            'medium': [e for e in embeddings if now - (14 * 24 * 3600) < e['timestamp'] <= now - (7 * 24 * 3600)],
            'older': [e for e in embeddings if e['timestamp'] <= now - (14 * 24 * 3600)]
        }

    def _analyze_period_embeddings(self, period_embeddings: List[Dict]) -> Dict[str, Any]:
        """Analyze embeddings for a specific time period"""
        # Analyze text traits for this period
        period_texts = [e['text'] for e in period_embeddings]
        combined_text = ' '.join(period_texts)
        period_traits = self.analyze_text_traits(combined_text)
        
        # Calculate embedding coherence
        coherence = self._calculate_embedding_coherence(period_embeddings)
        
        return {
            'trait_scores': period_traits,
            'embedding_coherence': float(coherence),
            'text_count': len(period_embeddings),
            'avg_text_length': np.mean([len(e['text'].split()) for e in period_embeddings])
        }

    def _calculate_embedding_coherence(self, embeddings: List[Dict]) -> float:
        """Calculate coherence score for embeddings"""
        if len(embeddings) <= 1:
            return 1.0
            
        embedding_matrix = np.array([e['embedding'] for e in embeddings])
        similarities = []
        
        for i in range(len(embedding_matrix)):
            for j in range(i + 1, len(embedding_matrix)):
                sim = self._cosine_similarity(embedding_matrix[i], embedding_matrix[j])
                similarities.append(sim)
        
        return float(np.mean(similarities)) if similarities else 0.0

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

class GlyphResponseModulator:
    """Modulates agent responses based on semantic glyph matches and personality traits"""
    
    def __init__(self, embedding_analyzer):
        self.embedding_analyzer = embedding_analyzer
        
        # Response modulation templates based on glyph types
        self.glyph_response_styles = {
            'nature_peaceful': {
                'tone_markers': ['calm', 'gentle', 'serene', 'peaceful'],
                'pacing': 'slow',
                'emotional_intensity': 0.3,
                'response_templates': [
                    "I sense a peaceful energy in what you're sharing. Let me reflect on this with you...",
                    "There's something beautifully grounding about your perspective. Tell me more...",
                    "I feel the tranquility in your words. How can I support this feeling?"
                ]
            },
            'tech_curiosity': {
                'tone_markers': ['energetic', 'exploring', 'discovering', 'fascinated'],
                'pacing': 'quick',
                'emotional_intensity': 0.7,
                'response_templates': [
                    "Your excitement about this is infectious! Let's dive deeper into...",
                    "I love how your mind works - always exploring new possibilities! What if we...",
                    "This fascination you have opens up so many interesting paths. Which direction calls to you?"
                ]
            },
            'analytical_logic': {
                'tone_markers': ['precise', 'systematic', 'thoughtful', 'structured'],
                'pacing': 'measured',
                'emotional_intensity': 0.5,
                'response_templates': [
                    "Let me think through this systematically with you. First, we should consider...",
                    "Your logical approach is valuable here. Building on your reasoning...",
                    "I appreciate the analytical framework you're using. Let's examine..."
                ]
            },
            'emotional_support': {
                'tone_markers': ['warm', 'understanding', 'supportive', 'empathetic'],
                'pacing': 'gentle',
                'emotional_intensity': 0.8,
                'response_templates': [
                    "I hear the emotion in your words, and I want you to know I'm here with you...",
                    "Your feelings are valid and important. Let's explore this together...",
                    "I sense this touches something deep in you. Thank you for trusting me with this..."
                ]
            },
            'creative_vision': {
                'tone_markers': ['inspiring', 'imaginative', 'visionary', 'artistic'],
                'pacing': 'flowing',
                'emotional_intensity': 0.6,
                'response_templates': [
                    "Your imagination sparks new possibilities! What if we could...",
                    "I see the creative vision you're painting. Let's bring it to life by...",
                    "There's something beautifully innovative in your thinking. How might we..."
                ]
            }
        }

    def classify_glyph_type(self, text: str, similarity_score: float) -> str:
        """Classify the type of glyph based on text content and similarity"""
        text_lower = text.lower()
        
        # Keywords that indicate different glyph types
        glyph_keywords = {
            'nature_peaceful': ['nature', 'peace', 'calm', 'hiking', 'mountains', 'trees', 'quiet', 'serene'],
            'tech_curiosity': ['technology', 'learn', 'excited', 'discover', 'programming', 'innovation'],
            'analytical_logic': ['think', 'analyze', 'logical', 'reason', 'systematic', 'method', 'process'],
            'emotional_support': ['feel', 'emotion', 'support', 'care', 'heart', 'difficult', 'struggle'],
            'creative_vision': ['create', 'imagine', 'vision', 'artistic', 'design', 'creative', 'inspire']
        }
        
        # Score each glyph type
        glyph_scores = {}
        for glyph_type, keywords in glyph_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            glyph_scores[glyph_type] = score
        
        # Return the highest scoring type, or default if no clear match
        if max(glyph_scores.values()) > 0:
            return max(glyph_scores.items(), key=lambda x: x[1])[0]
        else:
            return 'analytical_logic'  # Default

    def modulate_response(self, user_message: str, glyph_matches: List[Dict], trait_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate modulated response based on glyph matches and personality traits"""
        
        if not glyph_matches:
            return self._default_response(user_message, trait_scores)
        
        # Get the best glyph match
        best_match = max(glyph_matches, key=lambda x: x['similarity'])
        
        if best_match['similarity'] < 0.4:
            return self._default_response(user_message, trait_scores)
        
        # Classify the glyph type
        glyph_type = self.classify_glyph_type(best_match['text'], best_match['similarity'])
        
        # Get response style for this glyph type
        response_style = self.glyph_response_styles.get(glyph_type, self.glyph_response_styles['analytical_logic'])
        
        # Adjust style based on user's trait scores
        adjusted_style = self._adjust_style_for_traits(response_style, trait_scores)
        
        # Select appropriate response template
        template = self._select_response_template(adjusted_style, trait_scores)
        
        return {
            'glyph_type': glyph_type,
            'similarity_score': best_match['similarity'],
            'matched_content': best_match['text'][:100] + '...',
            'response_style': adjusted_style,
            'suggested_template': template,
            'tone_guidance': {
                'emotional_intensity': adjusted_style['emotional_intensity'],
                'pacing': adjusted_style['pacing'],
                'key_tone_markers': adjusted_style['tone_markers'][:3]
            },
            'personality_influence': {
                'curiosity_adjustment': trait_scores.get('curiosity_score', 0.5),
                'emotional_expressiveness': trait_scores.get('emotional_expressiveness', 0.5),
                'analytical_preference': trait_scores.get('analytical_thinking', 0.5)
            }
        }

    def _adjust_style_for_traits(self, base_style: Dict, trait_scores: Dict[str, float]) -> Dict[str, Any]:
        """Adjust response style based on user's personality traits"""
        adjusted_style = base_style.copy()
        
        # Adjust emotional intensity based on user's emotional expressiveness
        emotional_trait = trait_scores.get('emotional_expressiveness', 0.5)
        adjusted_style['emotional_intensity'] = (
            base_style['emotional_intensity'] * 0.7 + emotional_trait * 0.3
        )
        
        # Adjust pacing based on curiosity and analytical thinking
        curiosity = trait_scores.get('curiosity_score', 0.5)
        analytical = trait_scores.get('analytical_thinking', 0.5)
        
        if curiosity > 0.7:
            adjusted_style['pacing'] = 'quick'
        elif analytical > 0.7:
            adjusted_style['pacing'] = 'measured'
        
        return adjusted_style

    def _select_response_template(self, style: Dict, trait_scores: Dict[str, float]) -> str:
        """Select the most appropriate response template"""
        templates = style.get('response_templates', [])
        
        if not templates:
            return "I understand what you're sharing. Let me think about this with you..."
        
        # Simple selection based on user's social engagement
        social_engagement = trait_scores.get('social_engagement', 0.5)
        
        if social_engagement > 0.7:
            # Choose more collaborative templates
            collaborative_templates = [t for t in templates if 'we' in t or 'us' in t or 'together' in t]
            if collaborative_templates:
                return collaborative_templates[0]
        
        # Default to first template
        return templates[0]

    def _default_response(self, user_message: str, trait_scores: Dict[str, float]) -> Dict[str, Any]:
        """Generate default response when no strong glyph matches"""
        return {
            'glyph_type': 'default',
            'similarity_score': 0.0,
            'matched_content': 'No strong semantic match found',
            'response_style': {
                'tone_markers': ['thoughtful', 'attentive', 'present'],
                'pacing': 'natural',
                'emotional_intensity': 0.5
            },
            'suggested_template': "I'm here and listening. Tell me more about what's on your mind...",
            'tone_guidance': {
                'emotional_intensity': 0.5,
                'pacing': 'natural',
                'key_tone_markers': ['thoughtful', 'attentive', 'present']
            }
        }

class AdvancedPersonalityEngine:
    """Integration of trait scoring and glyph modulation with the existing system"""
    
    def __init__(self, embedding_analyzer, memory_server_url: str = "http://localhost:8001"):
        self.embedding_analyzer = embedding_analyzer
        self.trait_scorer = TraitScoringEngine()
        self.glyph_modulator = GlyphResponseModulator(embedding_analyzer)
        self.memory_server_url = memory_server_url

    def analyze_user_interaction(self, user_id: str, user_message: str) -> Dict[str, Any]:
        """Comprehensive analysis of user interaction with response modulation"""
        
        # Get user's embeddings and semantic context
        embeddings = self.embedding_analyzer.get_user_embeddings(user_id, days_back=30)
        
        # Analyze text traits
        message_traits = self.trait_scorer.analyze_text_traits(user_message)
        
        # Analyze embedding patterns
        embedding_analysis = self.trait_scorer.analyze_embedding_clusters(embeddings, user_id)
        
        # Find semantic glyph matches
        glyph_matches = self.embedding_analyzer.find_semantic_glyphs(user_id, user_message, top_k=3)
        
        # Generate modulated response
        response_modulation = self.glyph_modulator.modulate_response(
            user_message, glyph_matches, message_traits
        )
        
        # Update personality matrix with new trait scores
        self._update_trait_scores(user_id, message_traits)
        
        return {
            'user_id': user_id,
            'message_traits': message_traits,
            'embedding_analysis': embedding_analysis,
            'glyph_matches': glyph_matches,
            'response_modulation': response_modulation,
            'analysis_timestamp': datetime.now().isoformat()
        }

    def _update_trait_scores(self, user_id: str, trait_scores: Dict[str, float]):
        """Update trait scores in the personality matrix"""
        for trait, score in trait_scores.items():
            try:
                response = requests.post(f"{self.memory_server_url}/personality/update-trait", 
                                       json={
                                           'user_id': user_id,
                                           'trait': trait,
                                           'score': score
                                       }, timeout=5)
                if response.status_code != 200:
                    logger.warning(f"Failed to update trait {trait}: {response.text}")
            except Exception as e:
                logger.error(f"Error updating trait {trait}: {e}")

def demo_advanced_analysis():
    """Demonstration of the advanced personality engine"""
    from embedding_personality_analyzer import EmbeddingAnalyzer
    
    # Initialize components
    embedding_analyzer = EmbeddingAnalyzer()
    personality_engine = AdvancedPersonalityEngine(embedding_analyzer)
    
    # Test messages representing different personality types
    test_interactions = [
        {
            'user_id': 'test_user_123',
            'message': 'I love exploring new technologies and understanding how they work. What can you teach me about machine learning?'
        },
        {
            'user_id': 'test_user_123', 
            'message': 'I feel overwhelmed by all the changes happening in my life right now. I need some peace and clarity.'
        },
        {
            'user_id': 'test_user_123',
            'message': 'Let me think about this systematically. First, we need to analyze the problem, then consider the available solutions.'
        }
    ]
    
    print("🧬 Advanced Personality Engine Demo")
    print("=" * 60)
    
    for i, interaction in enumerate(test_interactions, 1):
        print(f"\n🔍 INTERACTION {i}")
        print(f"Message: {interaction['message']}")
        print("-" * 40)
        
        analysis = personality_engine.analyze_user_interaction(
            interaction['user_id'], 
            interaction['message']
        )
        
        # Display trait scores
        traits = analysis['message_traits']
        print("🎯 DETECTED TRAITS:")
        for trait, score in sorted(traits.items(), key=lambda x: x[1], reverse=True)[:5]:
            if score > 0.01:
                print(f"   {trait}: {score:.3f}")
        
        # Display glyph matches
        glyph_matches = analysis['glyph_matches']
        print(f"\n🔮 GLYPH MATCHES ({len(glyph_matches)} found):")
        for match in glyph_matches[:2]:
            print(f"   {match['similarity']:.3f}: {match['text'][:50]}...")
        
        # Display response modulation
        modulation = analysis['response_modulation']
        print(f"\n🎭 RESPONSE MODULATION:")
        print(f"   Glyph Type: {modulation['glyph_type']}")
        print(f"   Emotional Intensity: {modulation['tone_guidance']['emotional_intensity']:.2f}")
        print(f"   Pacing: {modulation['tone_guidance']['pacing']}")
        print(f"   Template: {modulation['suggested_template'][:80]}...")
        
        print("\n" + "=" * 60)

if __name__ == "__main__":
    demo_advanced_analysis()

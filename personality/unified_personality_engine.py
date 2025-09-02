#!/usr/bin/env python3
"""
🧠 Unified AGI Personality Engine
Single comprehensive system combining all personality analysis approaches:
- Traditional Big Five personality traits
- Advanced 28-facet TGA (Taxonomic Graph Analysis) 
- Embedding-based pattern recognition
- Longitudinal trait tracking and archetype evolution
- Real-time mood and emotional state analysis
"""

import numpy as np
import json
import sqlite3
import logging
import hashlib
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import OrderedDict
from dataclasses import dataclass
import threading
import os

logger = logging.getLogger(__name__)

@dataclass
class PersonalitySnapshot:
    """Complete personality snapshot with all dimensions"""
    timestamp: datetime
    user_id: str
    
    # Core trait systems
    big_five_scores: Dict[str, float]
    facet_scores: Dict[str, float]  # 28+ facets
    mood_vector: Dict[str, float]
    meta_traits: Dict[str, float]  # Stability, Plasticity, Disinhibition
    
    # Contextual data
    archetype: str
    confidence: float
    sample_size: int
    
    # Evolution tracking
    trait_drift: Dict[str, float]
    archetype_stability: float

class UnifiedPersonalityEngine:
    """
    Complete personality analysis system combining all approaches
    """
    
    def __init__(self, db_path: str = "memory/user_memory.db"):
        # Make database path relative to AGI root directory
        if not os.path.isabs(db_path):
            # Try to find the AGI root directory
            current_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(current_dir)
            db_path = os.path.join(parent_dir, db_path)
        
        # Ensure the memory directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        self.db_path = db_path
        self.personality_cache = OrderedDict()  # LRU cache
        self.cache_lock = threading.Lock()
        self.cache_size = 100
        
        # Initialize database
        self._initialize_database()
        
        # Define personality archetypes (consolidated from all systems)
        self.archetypes = {
            # Analytical Types
            'strategic_analyst': {
                'traits': {'conscientiousness': 0.8, 'openness': 0.7, 'analytical_thinking': 0.9},
                'description': 'Logical, systematic, plans ahead',
                'response_style': 'structured and methodical',
                'keywords': ['analyze', 'logic', 'systematic', 'rational', 'data']
            },
            
            # Creative Types  
            'creative_visionary': {
                'traits': {'openness': 0.9, 'creativity': 0.8, 'innovation_drive': 0.8},
                'description': 'Imaginative, innovative, artistic',
                'response_style': 'inspiring and creative',
                'keywords': ['imagine', 'create', 'vision', 'innovative', 'artistic']
            },
            
            # Social Types
            'empathetic_nurturer': {
                'traits': {'agreeableness': 0.9, 'emotional_expressiveness': 0.8, 'social_engagement': 0.8},
                'description': 'Caring, supportive, emotionally attuned',
                'response_style': 'warm and empathetic', 
                'keywords': ['understand', 'support', 'feel', 'help', 'care']
            },
            
            # Explorer Types
            'adventurous_explorer': {
                'traits': {'extraversion': 0.8, 'openness': 0.8, 'curiosity_score': 0.9},
                'description': 'Curious, adventurous, seeks new experiences',
                'response_style': 'enthusiastic and exploratory',
                'keywords': ['adventure', 'explore', 'experience', 'travel', 'discover']
            },
            
            # Balanced Types
            'adaptive_sage': {
                'traits': {'conscientiousness': 0.6, 'openness': 0.6, 'agreeableness': 0.7},
                'description': 'Balanced, wise, thoughtful',
                'response_style': 'measured and insightful',
                'keywords': ['balance', 'wisdom', 'thoughtful', 'consider', 'reflect']
            },
            
            # Dynamic Types
            'dynamic_catalyst': {
                'traits': {'extraversion': 0.8, 'innovation_drive': 0.7, 'social_engagement': 0.8},
                'description': 'Energetic, inspiring, brings change',
                'response_style': 'dynamic and motivating',
                'keywords': ['energy', 'inspire', 'change', 'motivate', 'action']
            }
        }
        
        # TGA Facet definitions (enhanced with advanced trait indicators)
        self.facet_definitions = {
            # Openness Facets
            'intellectual_curiosity': {
                'indicators': ['why', 'how', 'understand', 'learn', 'knowledge', 'research', 'study', 'wonder', 'curious', 'explore', 'discover', 'find out', 'investigate'],
                'reverse_indicators': ['boring', 'obvious', 'simple', 'dont care'],
                'question_markers': ['?', 'why', 'how', 'what', 'when', 'where'],
                'weight': 1.0
            },
            'aesthetic_sensitivity': {
                'indicators': ['beautiful', 'art', 'design', 'elegant', 'style', 'creative', 'artistic'],
                'reverse_indicators': ['ugly', 'plain', 'boring design'],
                'weight': 1.0
            },
            'creative_thinking': {
                'indicators': ['creative', 'innovative', 'imagine', 'idea', 'brainstorm', 'invent', 'create', 'build', 'make', 'design', 'innovate'],
                'reverse_indicators': ['conventional', 'traditional', 'standard way'],
                'creation_words': ['create', 'build', 'make', 'design', 'innovate', 'invent'],
                'weight': 1.0
            },
            'cognitive_flexibility': {
                'indicators': ['different', 'alternative', 'various', 'multiple ways', 'perspective'],
                'reverse_indicators': ['one way', 'rigid', 'fixed', 'stubborn'],
                'weight': 1.0
            },
            
            # Conscientiousness Facets  
            'orderliness': {
                'indicators': ['organize', 'plan', 'schedule', 'structure', 'system', 'order', 'tidy'],
                'reverse_indicators': ['messy', 'chaotic', 'disorganized', 'scattered'],
                'weight': 1.0
            },
            'impulse_control': {
                'indicators': ['think before', 'consider', 'careful', 'deliberate', 'patient'],
                'reverse_indicators': ['impulsive', 'hasty', 'rush', 'without thinking'],
                'weight': 1.0
            },
            'behavioral_control': {
                'indicators': ['disciplined', 'self-control', 'restrained', 'measured'],
                'reverse_indicators': ['wild', 'uncontrolled', 'reckless', 'spontaneous'],
                'weight': 1.0
            },
            
            # Extraversion Facets
            'sociability': {
                'indicators': ['social', 'people', 'friends', 'party', 'group', 'together'],
                'reverse_indicators': ['alone', 'solitude', 'private', 'isolated'],
                'weight': 1.0
            },
            'assertiveness': {
                'indicators': ['lead', 'direct', 'confident', 'strong', 'assertive', 'command'],
                'reverse_indicators': ['shy', 'quiet', 'follow', 'passive'],
                'weight': 1.0
            },
            'social_adaptability': {
                'indicators': ['adapt', 'fit in', 'adjust', 'flexible socially'],
                'reverse_indicators': ['awkward', 'misfit', 'dont belong'],
                'weight': 1.0
            },
            
            # Agreeableness Facets
            'compassion': {
                'indicators': ['care', 'empathy', 'compassionate', 'understanding', 'kind'],
                'reverse_indicators': ['cold', 'harsh', 'uncaring', 'selfish'],
                'weight': 1.0
            },
            'trust': {
                'indicators': ['trust', 'believe', 'faith', 'reliable', 'honest'],
                'reverse_indicators': ['suspicious', 'doubt', 'distrust', 'skeptical'],
                'weight': 1.0
            },
            'cooperation': {
                'indicators': ['together', 'teamwork', 'collaborate', 'cooperate', 'help'],
                'reverse_indicators': ['compete', 'against', 'conflict', 'oppose'],
                'weight': 1.0
            },
            
            # Neuroticism Facets (reverse scored)
            'emotional_regulation': {
                'indicators': ['calm', 'stable', 'composed', 'balanced', 'steady'],
                'reverse_indicators': ['anxious', 'stressed', 'worried', 'upset', 'emotional'],
                'weight': 1.0
            },
            'anxiety_management': {
                'indicators': ['relaxed', 'confident', 'secure', 'comfortable'],
                'reverse_indicators': ['anxious', 'nervous', 'worried', 'fearful'],
                'weight': 1.0
            },
            'mood_stability': {
                'indicators': ['consistent', 'stable mood', 'even', 'predictable'],
                'reverse_indicators': ['moody', 'volatile', 'unpredictable', 'ups and downs'],
                'weight': 1.0
            },
            
            # Advanced Facets (enhanced with trait scoring patterns)
            'innovation_drive': {
                'indicators': ['new', 'innovative', 'change', 'improve', 'better way', 'breakthrough'],
                'reverse_indicators': ['traditional', 'same old', 'conventional', 'status quo'],
                'future_focus': ['future', 'tomorrow', 'next', 'upcoming', 'vision', 'goal'],
                'change_orientation': ['change', 'transform', 'improve', 'revolutionize'],
                'weight': 1.0
            },
            'detail_orientation': {
                'indicators': ['detail', 'precise', 'exact', 'thorough', 'careful', 'specific'],
                'reverse_indicators': ['general', 'rough', 'approximate', 'overview'],
                'precision_words': ['specific', 'exact', 'precise', 'detailed', 'thorough'],
                'qualification_markers': ['however', 'although', 'except', 'unless', 'but'],
                'elaboration_patterns': ['furthermore', 'moreover', 'additionally', 'specifically'],
                'weight': 1.0
            },
            'analytical_thinking': {
                'indicators': ['analyze', 'logic', 'rational', 'reason', 'systematic', 'data'],
                'reverse_indicators': ['gut feeling', 'intuitive', 'emotional decision'],
                'logic_words': ['because', 'therefore', 'however', 'analysis', 'conclusion', 'evidence'],
                'reasoning_patterns': ['first', 'second', 'then', 'consequently', 'thus', 'hence'],
                'structure_indicators': ['step', 'process', 'method', 'approach', 'strategy'],
                'weight': 1.2
            },
            'emotional_expressiveness': {
                'indicators': ['feel', 'emotion', 'heart', 'passionate', 'expressive'],
                'reverse_indicators': ['logical only', 'unemotional', 'cold', 'detached'],
                'emotion_words': ['feel', 'feeling', 'emotion', 'heart', 'soul', 'passionate'],
                'intensity_markers': ['very', 'extremely', 'incredibly', 'absolutely', 'deeply'],
                'exclamation_usage': ['!', '!!', '!!!'],
                'weight': 1.0
            },
            'social_engagement': {
                'indicators': ['social', 'community', 'people', 'interact', 'connect'],
                'reverse_indicators': ['antisocial', 'isolated', 'withdrawn', 'loner'],
                'inclusive_language': ['we', 'us', 'our', 'together', 'collaborate', 'share'],
                'personal_sharing': ['my', 'me', 'myself', 'personal', 'experience', 'story'],
                'community_focus': ['community', 'group', 'team', 'everyone', 'all'],
                'weight': 1.1
            },
            'curiosity_score': {
                'indicators': ['curious', 'wonder', 'explore', 'discover', 'investigate'],
                'reverse_indicators': ['uninterested', 'bored', 'dont care', 'obvious'],
                'question_markers': ['?', 'why', 'how', 'what', 'when', 'where', 'wonder', 'curious'],
                'exploration_words': ['explore', 'discover', 'learn', 'understand', 'find out', 'investigate'],
                'weight': 1.0
            },
            'learning_preference': {
                'indicators': ['learn', 'study', 'understand', 'knowledge', 'education'],
                'reverse_indicators': ['dont want to learn', 'waste of time'],
                'example_seeking': ['example', 'instance', 'case', 'sample', 'demonstration'],
                'theory_seeking': ['theory', 'principle', 'concept', 'framework', 'model'],
                'practice_seeking': ['practice', 'try', 'apply', 'implement', 'hands-on'],
                'weight': 0.9
            },
            'sarcasm_index': {
                'indicators': ['obviously', 'clearly', 'sure', 'right', 'great', 'perfect'],
                'irony_markers': ['obviously', 'clearly', 'sure', 'right', 'great', 'perfect', 'wonderful'],
                'context_indicators': ['...', '!', 'oh really', 'of course', 'wow'],
                'punctuation_patterns': ['!!!', '...', '?!'],
                'weight': 0.8
            }
        }
        
        # Trait evolution thresholds
        self.evolution_thresholds = {
            'minor_drift': 0.15,
            'moderate_shift': 0.3,
            'major_evolution': 0.5
        }
    
    def _initialize_database(self):
        """Initialize comprehensive personality database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Main personality snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    big_five_json TEXT NOT NULL,
                    facets_json TEXT NOT NULL,
                    mood_vector_json TEXT NOT NULL,
                    meta_traits_json TEXT NOT NULL,
                    archetype TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    snapshot_hash TEXT UNIQUE
                )
            """)
            
            # Trait timeline for longitudinal tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trait_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trait_name TEXT NOT NULL,
                    trait_value REAL NOT NULL,
                    interaction_context TEXT,
                    timestamp BIGINT NOT NULL,
                    session_id TEXT
                )
            """)
            
            # Archetype evolution tracking
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archetype_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    previous_archetype TEXT,
                    new_archetype TEXT NOT NULL,
                    transition_reason TEXT,
                    confidence_score REAL,
                    trait_snapshot_json TEXT,
                    timestamp BIGINT NOT NULL
                )
            """)
            
            # Trait drift events
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trait_drift_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    trait_name TEXT NOT NULL,
                    drift_magnitude REAL NOT NULL,
                    drift_direction TEXT NOT NULL,
                    retraining_triggered BOOLEAN DEFAULT FALSE
                )
            """)
            
            # Create performance indices
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_user_time ON personality_snapshots(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_timeline_user_time ON trait_timeline(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_evolution_user_time ON archetype_evolution(user_id, timestamp)")
            
            conn.commit()
    
    def analyze_user_interaction(self, user_id: str, message: str, session_id: str = "") -> PersonalitySnapshot:
        """
        Complete personality analysis of user interaction
        """
        # Calculate facet scores using TGA methodology
        facet_scores = self._calculate_facet_scores([message])
        
        # Calculate Big Five from facets
        big_five_scores = self._calculate_big_five_from_facets(facet_scores)
        
        # Calculate mood vector
        mood_vector = self._analyze_emotional_state(message)
        
        # Calculate meta-traits
        meta_traits = self._calculate_meta_traits(facet_scores)
        
        # Determine archetype
        archetype = self._determine_archetype(big_five_scores, facet_scores, meta_traits)
        
        # Calculate confidence
        confidence = self._calculate_confidence([message], facet_scores)
        
        # Get historical context for drift analysis
        trait_drift = self._calculate_trait_drift(user_id, facet_scores)
        archetype_stability = self._calculate_archetype_stability(user_id, archetype)
        
        # Create snapshot
        snapshot = PersonalitySnapshot(
            timestamp=datetime.now(),
            user_id=user_id,
            big_five_scores=big_five_scores,
            facet_scores=facet_scores,
            mood_vector=mood_vector,
            meta_traits=meta_traits,
            archetype=archetype,
            confidence=confidence,
            sample_size=1,
            trait_drift=trait_drift,
            archetype_stability=archetype_stability
        )
        
        # Store snapshot and update tracking
        self._store_snapshot(snapshot)
        self._update_trait_timeline(user_id, facet_scores, message[:100], session_id)
        self._check_archetype_evolution(user_id, snapshot)
        
        # Update cache
        self._update_cache(user_id, snapshot)
        
        return snapshot
    
    def _calculate_facet_scores(self, texts: List[str]) -> Dict[str, float]:
        """Calculate enhanced facet scores using advanced trait indicators"""
        combined_text = " ".join(texts).lower()
        original_text = " ".join(texts)  # Keep original for punctuation analysis
        facet_scores = {}
        
        word_count = len(combined_text.split())
        sentence_count = len([s for s in re.split(r'[.!?]+', original_text) if s.strip()])
        
        for facet_name, facet_def in self.facet_definitions.items():
            score = self._calculate_enhanced_facet_score(original_text, combined_text, facet_def)
            facet_scores[facet_name] = self._normalize_enhanced_score(score, word_count, facet_def)
        
        # Add complex trait calculations
        facet_scores.update(self._calculate_complex_traits(original_text, combined_text, word_count, sentence_count))
        
        return facet_scores
    
    def _calculate_enhanced_facet_score(self, original_text: str, text_lower: str, facet_def: Dict) -> float:
        """Calculate enhanced facet score using multiple indicator types"""
        total_score = 0.0
        total_weight = 0.0
        
        # Process all indicator types
        for indicator_type, indicators in facet_def.items():
            if indicator_type == 'weight':
                continue
                
            if isinstance(indicators, list):
                if indicator_type == 'indicators':
                    # Positive indicators (standard weight)
                    score = sum(1.0 for indicator in indicators if indicator in text_lower)
                    total_score += score
                    total_weight += len(indicators)
                    
                elif indicator_type == 'reverse_indicators':
                    # Negative indicators (subtract)
                    score = sum(1.0 for indicator in indicators if indicator in text_lower)
                    total_score -= score * 0.8  # Slightly less negative weight
                    total_weight += len(indicators) * 0.8
                    
                elif indicator_type == 'question_markers':
                    # Question patterns
                    score = sum(0.8 for indicator in indicators if indicator in text_lower)
                    total_score += score
                    total_weight += len(indicators) * 0.8
                    
                elif indicator_type == 'exclamation_usage':
                    # Exclamation patterns
                    score = sum(original_text.count(pattern) * 0.3 for pattern in indicators)
                    total_score += score
                    total_weight += len(indicators) * 0.3
                    
                elif indicator_type == 'punctuation_patterns':
                    # Complex punctuation patterns
                    score = sum(original_text.count(pattern) * 0.5 for pattern in indicators)
                    total_score += score
                    total_weight += len(indicators) * 0.5
                    
                else:
                    # Other specialized indicators (medium weight)
                    score = sum(0.6 for indicator in indicators if indicator in text_lower)
                    total_score += score
                    total_weight += len(indicators) * 0.6
        
        # Return weighted average
        return total_score / max(1.0, total_weight)
    
    def _normalize_enhanced_score(self, raw_score: float, word_count: int, facet_def: Dict) -> float:
        """Normalize enhanced score with word count and facet weight"""
        if word_count == 0:
            return 0.5
            
        # Normalize by word count (per 100 words)
        normalized = (raw_score / word_count) * 100
        
        # Apply facet-specific weight
        weight = facet_def.get('weight', 1.0)
        weighted_score = normalized * weight
        
        # Ensure score stays in valid range with neutral baseline
        return float(max(0.0, min(1.0, 0.5 + (weighted_score - 0.5))))
    
    def _calculate_complex_traits(self, original_text: str, text_lower: str, word_count: int, sentence_count: int) -> Dict[str, float]:
        """Calculate complex trait metrics from text patterns"""
        complex_traits = {}
        
        # Question density (curiosity indicator)
        if sentence_count > 0:
            complex_traits['question_density'] = min(1.0, original_text.count('?') / sentence_count)
        else:
            complex_traits['question_density'] = 0.0
            
        # Average sentence length (detail orientation indicator)
        if sentence_count > 0:
            avg_length = word_count / sentence_count
            # Normalize to 0-1 scale (assuming 5-25 words per sentence is normal range)
            complex_traits['sentence_complexity'] = min(1.0, max(0.0, (avg_length - 5) / 20))
        else:
            complex_traits['sentence_complexity'] = 0.5
            
        # Lexical diversity (intellectual sophistication indicator)
        if word_count > 0:
            unique_words = len(set(text_lower.split()))
            diversity = unique_words / word_count
            complex_traits['lexical_diversity'] = float(diversity)
        else:
            complex_traits['lexical_diversity'] = 0.5
            
        # Emotional intensity (based on exclamation usage)
        exclamation_count = original_text.count('!') + original_text.count('!!') * 2 + original_text.count('!!!') * 3
        if word_count > 0:
            complex_traits['emotional_intensity'] = min(1.0, exclamation_count / max(1, word_count / 20))
        else:
            complex_traits['emotional_intensity'] = 0.1
        
        return complex_traits
    
    def _calculate_big_five_from_facets(self, facet_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate Big Five scores from facet scores"""
        big_five = {}
        
        # Openness
        openness_facets = ['intellectual_curiosity', 'aesthetic_sensitivity', 'creative_thinking', 'cognitive_flexibility']
        big_five['openness'] = float(np.mean([facet_scores.get(f, 0.5) for f in openness_facets]))
        
        # Conscientiousness  
        consc_facets = ['orderliness', 'impulse_control', 'behavioral_control', 'detail_orientation']
        big_five['conscientiousness'] = float(np.mean([facet_scores.get(f, 0.5) for f in consc_facets]))
        
        # Extraversion
        extra_facets = ['sociability', 'assertiveness', 'social_adaptability', 'social_engagement']
        big_five['extraversion'] = float(np.mean([facet_scores.get(f, 0.5) for f in extra_facets]))
        
        # Agreeableness
        agree_facets = ['compassion', 'trust', 'cooperation']
        big_five['agreeableness'] = float(np.mean([facet_scores.get(f, 0.5) for f in agree_facets]))
        
        # Neuroticism (reverse scored)
        neuro_facets = ['emotional_regulation', 'anxiety_management', 'mood_stability']
        big_five['neuroticism'] = float(1.0 - np.mean([facet_scores.get(f, 0.5) for f in neuro_facets]))
        
        return {k: float(v) for k, v in big_five.items()}
    
    def _analyze_emotional_state(self, text: str) -> Dict[str, float]:
        """Analyze current emotional state from text"""
        text_lower = text.lower()
        
        # Emotional indicators
        emotions = {
            'joy': ['happy', 'excited', 'great', 'awesome', 'love', 'wonderful', 'amazing'],
            'curiosity': ['why', 'how', 'what', 'wonder', 'curious', 'interesting', 'explore'],
            'frustration': ['frustrated', 'annoyed', 'irritated', 'stuck', 'difficult', 'problems'],
            'awe': ['incredible', 'amazing', 'wow', 'impressive', 'remarkable'],
            'trust': ['trust', 'confident', 'sure', 'reliable', 'believe'],
            'anticipation': ['excited', 'looking forward', 'cant wait', 'anticipate'],
            'affection': ['care', 'love', 'appreciate', 'fond', 'cherish'],
            'loneliness': ['lonely', 'alone', 'isolated', 'nobody', 'miss'],
            'grief': ['sad', 'loss', 'miss', 'gone', 'sorry', 'regret'],
            'restraint': ['careful', 'cautious', 'hold back', 'restrained', 'conservative']
        }
        
        mood_vector = {}
        for emotion, indicators in emotions.items():
            score = sum(1 for indicator in indicators if indicator in text_lower)
            # Normalize with baseline
            mood_vector[emotion] = min(1.0, 0.1 + (score * 0.2))
        
        return mood_vector
    
    def _calculate_meta_traits(self, facet_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate higher-order meta-traits"""
        
        # Stability (emotional regulation + behavioral control)
        stability_facets = ['emotional_regulation', 'behavioral_control', 'mood_stability', 'impulse_control']
        stability = float(np.mean([facet_scores.get(f, 0.5) for f in stability_facets]))
        
        # Plasticity (openness + extraversion facets)
        plasticity_facets = ['intellectual_curiosity', 'creative_thinking', 'social_engagement', 'innovation_drive']
        plasticity = float(np.mean([facet_scores.get(f, 0.5) for f in plasticity_facets]))
        
        # Disinhibition (impulsivity + risk-taking)
        disinhibition = float(1.0 - np.mean([
            facet_scores.get('impulse_control', 0.5),
            facet_scores.get('behavioral_control', 0.5)
        ]))
        
        return {
            'stability': stability,
            'plasticity': plasticity, 
            'disinhibition': disinhibition
        }
    
    def _determine_archetype(self, big_five: Dict[str, float], facets: Dict[str, float], 
                           meta_traits: Dict[str, float]) -> str:
        """Determine personality archetype from trait combination"""
        
        # Calculate archetype scores
        archetype_scores = {}
        
        for archetype_name, archetype_def in self.archetypes.items():
            score = 0.0
            trait_count = 0
            
            for trait_name, required_value in archetype_def['traits'].items():
                # Check in all trait sources
                current_value = (
                    big_five.get(trait_name) or 
                    facets.get(trait_name) or 
                    meta_traits.get(trait_name) or 
                    0.5
                )
                
                # Calculate match score
                if required_value >= 0.7:  # High trait requirement
                    match = min(current_value / required_value, 1.0) if required_value > 0 else 0.0
                else:  # Low trait requirement  
                    match = 1.0 - abs(current_value - required_value)
                
                score += max(0.0, match)
                trait_count += 1
            
            archetype_scores[archetype_name] = score / trait_count if trait_count > 0 else 0.0
        
        # Return best match
        best_archetype = max(archetype_scores.items(), key=lambda x: x[1])
        return best_archetype[0] if best_archetype[1] > 0.4 else 'adaptive_sage'
    
    def _calculate_confidence(self, texts: List[str], facet_scores: Dict[str, float]) -> float:
        """Calculate confidence in personality assessment"""
        
        # Base confidence on sample size
        sample_confidence = min(len(texts) / 10.0, 1.0)  # Max confidence at 10+ texts
        
        # Confidence based on trait clarity (distance from neutral 0.5)
        trait_clarity = float(np.mean([abs(score - 0.5) * 2 for score in facet_scores.values()]))
        
        # Combined confidence
        overall_confidence = float((sample_confidence * 0.6) + (trait_clarity * 0.4))
        
        return float(min(1.0, max(0.1, overall_confidence)))
    
    def _calculate_trait_drift(self, user_id: str, current_facets: Dict[str, float]) -> Dict[str, float]:
        """Calculate trait drift from historical baseline"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get historical averages (30 days back)
                cutoff_time = int((datetime.now() - timedelta(days=30)).timestamp())
                
                drift_scores = {}
                
                for trait_name in current_facets.keys():
                    cursor.execute("""
                        SELECT AVG(trait_value) FROM trait_timeline 
                        WHERE user_id = ? AND trait_name = ? AND timestamp > ?
                    """, (user_id, trait_name, cutoff_time))
                    
                    result = cursor.fetchone()
                    if result and result[0] is not None:
                        historical_avg = result[0]
                        current_value = current_facets[trait_name]
                        drift_scores[trait_name] = abs(current_value - historical_avg)
                    else:
                        drift_scores[trait_name] = 0.0
                
                return drift_scores
                
        except Exception as e:
            logger.error(f"Error calculating trait drift: {e}")
            return {trait: 0.0 for trait in current_facets.keys()}
    
    def _calculate_archetype_stability(self, user_id: str, current_archetype: str) -> float:
        """Calculate how stable the current archetype is"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get recent archetype history (7 days)
                cutoff_time = int((datetime.now() - timedelta(days=7)).timestamp())
                
                cursor.execute("""
                    SELECT new_archetype FROM archetype_evolution 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (user_id, cutoff_time))
                
                recent_archetypes = [row[0] for row in cursor.fetchall()]
                
                if not recent_archetypes:
                    return 1.0  # Stable if no recent changes
                
                # Calculate stability as proportion of matching archetypes
                matches = sum(1 for arch in recent_archetypes if arch == current_archetype)
                stability = matches / len(recent_archetypes)
                
                return float(stability)
                
        except Exception as e:
            logger.error(f"Error calculating archetype stability: {e}")
            return 1.0
    
    def _store_snapshot(self, snapshot: PersonalitySnapshot):
        """Store personality snapshot in database"""
        
        try:
            # Create unique hash for deduplication
            snapshot_data = {
                'user_id': snapshot.user_id,
                'big_five': snapshot.big_five_scores,
                'facets': snapshot.facet_scores,
                'archetype': snapshot.archetype
            }
            snapshot_hash = hashlib.sha256(json.dumps(snapshot_data, sort_keys=True).encode()).hexdigest()
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR IGNORE INTO personality_snapshots 
                    (user_id, timestamp, big_five_json, facets_json, mood_vector_json, 
                     meta_traits_json, archetype, confidence, sample_size, snapshot_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    snapshot.user_id,
                    snapshot.timestamp.isoformat(),
                    json.dumps(snapshot.big_five_scores),
                    json.dumps(snapshot.facet_scores),
                    json.dumps(snapshot.mood_vector),
                    json.dumps(snapshot.meta_traits),
                    snapshot.archetype,
                    snapshot.confidence,
                    snapshot.sample_size,
                    snapshot_hash
                ))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error storing personality snapshot: {e}")
    
    def _update_trait_timeline(self, user_id: str, facet_scores: Dict[str, float], 
                             context: str, session_id: str):
        """Update trait timeline for longitudinal tracking"""
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                timestamp = int(datetime.now().timestamp())
                
                for trait_name, trait_value in facet_scores.items():
                    cursor.execute("""
                        INSERT INTO trait_timeline 
                        (user_id, trait_name, trait_value, interaction_context, timestamp, session_id)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (user_id, trait_name, trait_value, context, timestamp, session_id))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating trait timeline: {e}")
    
    def _check_archetype_evolution(self, user_id: str, current_snapshot: PersonalitySnapshot):
        """Check if archetype should evolve based on trait changes"""
        
        try:
            # Get current archetype from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT new_archetype FROM archetype_evolution 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (user_id,))
                
                result = cursor.fetchone()
                previous_archetype = result[0] if result else 'adaptive_sage'
                
                # Check if archetype has changed significantly
                if (current_snapshot.archetype != previous_archetype and 
                    current_snapshot.confidence > 0.7):
                    
                    # Calculate trait changes that triggered transition
                    major_drifts = [
                        trait for trait, drift in current_snapshot.trait_drift.items() 
                        if drift > self.evolution_thresholds['major_evolution']
                    ]
                    
                    transition_reason = f"Major trait changes in: {', '.join(major_drifts)}" if major_drifts else "Gradual evolution"
                    
                    # Record archetype transition
                    timestamp = int(datetime.now().timestamp())
                    
                    cursor.execute("""
                        INSERT INTO archetype_evolution 
                        (user_id, previous_archetype, new_archetype, transition_reason, 
                         confidence_score, trait_snapshot_json, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        user_id, previous_archetype, current_snapshot.archetype,
                        transition_reason, current_snapshot.confidence,
                        json.dumps(current_snapshot.facet_scores), timestamp
                    ))
                    
                    conn.commit()
                    
                    logger.info(f"Archetype evolution: {user_id} {previous_archetype} -> {current_snapshot.archetype}")
                
        except Exception as e:
            logger.error(f"Error checking archetype evolution: {e}")
    
    def _update_cache(self, user_id: str, snapshot: PersonalitySnapshot):
        """Update LRU cache with latest snapshot"""
        
        with self.cache_lock:
            if user_id in self.personality_cache:
                self.personality_cache.move_to_end(user_id)
            else:
                if len(self.personality_cache) >= self.cache_size:
                    self.personality_cache.popitem(last=False)  # Remove oldest
                
            self.personality_cache[user_id] = snapshot
    
    def get_user_personality(self, user_id: str) -> Optional[PersonalitySnapshot]:
        """Get latest personality snapshot for user"""
        
        # Check cache first
        with self.cache_lock:
            if user_id in self.personality_cache:
                self.personality_cache.move_to_end(user_id)
                return self.personality_cache[user_id]
        
        # Query database
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT timestamp, big_five_json, facets_json, mood_vector_json, 
                           meta_traits_json, archetype, confidence, sample_size
                    FROM personality_snapshots 
                    WHERE user_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                """, (user_id,))
                
                result = cursor.fetchone()
                
                if result:
                    snapshot = PersonalitySnapshot(
                        timestamp=datetime.fromisoformat(result[0]),
                        user_id=user_id,
                        big_five_scores=json.loads(result[1]),
                        facet_scores=json.loads(result[2]),
                        mood_vector=json.loads(result[3]),
                        meta_traits=json.loads(result[4]),
                        archetype=result[5],
                        confidence=result[6],
                        sample_size=result[7],
                        trait_drift={},
                        archetype_stability=1.0
                    )
                    
                    # Update cache
                    self._update_cache(user_id, snapshot)
                    return snapshot
                
        except Exception as e:
            logger.error(f"Error getting user personality: {e}")
        
        return None
    
    def generate_personality_report(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive personality evolution report"""
        
        try:
            current_personality = self.get_user_personality(user_id)
            
            if not current_personality:
                return {'error': 'No personality data found'}
            
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get trait evolution over time
                cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
                
                cursor.execute("""
                    SELECT trait_name, AVG(trait_value) as avg_value,
                           COUNT(*) as sample_count,
                           MIN(trait_value) as min_val,
                           MAX(trait_value) as max_val
                    FROM trait_timeline 
                    WHERE user_id = ? AND timestamp > ?
                    GROUP BY trait_name
                """, (user_id, cutoff_time))
                
                trait_evolution = {}
                for row in cursor.fetchall():
                    trait_evolution[row[0]] = {
                        'average': row[1],
                        'sample_count': row[2],
                        'range': [row[3], row[4]],
                        'stability': 1.0 - (row[4] - row[3])  # Low range = high stability
                    }
                
                # Get archetype history
                cursor.execute("""
                    SELECT previous_archetype, new_archetype, transition_reason, 
                           confidence_score, timestamp
                    FROM archetype_evolution 
                    WHERE user_id = ? AND timestamp > ?
                    ORDER BY timestamp DESC
                """, (user_id, cutoff_time))
                
                archetype_history = []
                for row in cursor.fetchall():
                    archetype_history.append({
                        'from': row[0],
                        'to': row[1],
                        'reason': row[2],
                        'confidence': row[3],
                        'timestamp': datetime.fromtimestamp(row[4]).isoformat()
                    })
            
            # Generate insights
            archetype_info = self.archetypes.get(current_personality.archetype, {})
            
            return {
                'user_id': user_id,
                'report_period_days': days_back,
                'generated_at': datetime.now().isoformat(),
                'current_state': {
                    'archetype': current_personality.archetype,
                    'description': archetype_info.get('description', 'Unknown'),
                    'response_style': archetype_info.get('response_style', 'Adaptive'),
                    'big_five_scores': current_personality.big_five_scores,
                    'top_facets': sorted(current_personality.facet_scores.items(), 
                                       key=lambda x: x[1], reverse=True)[:5],
                    'confidence': current_personality.confidence
                },
                'trait_evolution': trait_evolution,
                'archetype_history': archetype_history,
                'stability_metrics': {
                    'archetype_stability': current_personality.archetype_stability,
                    'avg_trait_stability': float(np.mean([t['stability'] for t in trait_evolution.values()])) if trait_evolution else 1.0,
                    'total_interactions': sum([t['sample_count'] for t in trait_evolution.values()]) if trait_evolution else 0
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating personality report: {e}")
            return {'error': str(e)}

    def generate_response_modulation(self, snapshot: PersonalitySnapshot, user_message: str) -> Dict[str, Any]:
        """Generate response modulation based on personality and message content"""
        
        # Determine response style based on archetype
        archetype_info = self.archetypes.get(snapshot.archetype, {})
        base_style = archetype_info.get('response_style', 'balanced')
        
        # Calculate tone adjustment
        tone_adjustment = self._calculate_tone_adjustment(snapshot)
        
        # Calculate complexity level
        complexity_level = self._calculate_complexity_level(snapshot)
        
        # Calculate emotional resonance
        emotional_resonance = self._calculate_emotional_resonance(snapshot)
        
        # Determine engagement style based on social traits
        engagement_style = self._determine_engagement_style(snapshot, user_message)
        
        return {
            'base_style': base_style,
            'tone_adjustment': tone_adjustment,
            'complexity_level': complexity_level,
            'emotional_resonance': emotional_resonance,
            'engagement_style': engagement_style,
            'personality_influence': {
                'curiosity_level': snapshot.facet_scores.get('curiosity_score', 0.5),
                'analytical_preference': snapshot.facet_scores.get('analytical_thinking', 0.5),
                'emotional_expressiveness': snapshot.facet_scores.get('emotional_expressiveness', 0.5),
                'social_engagement': snapshot.facet_scores.get('social_engagement', 0.5)
            },
            'suggested_approach': self._suggest_response_approach(snapshot, user_message)
        }
    
    def _calculate_tone_adjustment(self, snapshot: PersonalitySnapshot) -> float:
        """Calculate tone adjustment based on personality"""
        extraversion = snapshot.big_five_scores.get('extraversion', 0.5)
        agreeableness = snapshot.big_five_scores.get('agreeableness', 0.5)
        
        # Scale from -1 (more reserved) to +1 (more enthusiastic)
        return float((extraversion + agreeableness) - 1.0)
    
    def _calculate_complexity_level(self, snapshot: PersonalitySnapshot) -> float:
        """Calculate appropriate response complexity"""
        openness = snapshot.big_five_scores.get('openness', 0.5)
        conscientiousness = snapshot.big_five_scores.get('conscientiousness', 0.5)
        analytical = snapshot.facet_scores.get('analytical_thinking', 0.5)
        
        return float((openness + conscientiousness + analytical) / 3.0)
    
    def _calculate_emotional_resonance(self, snapshot: PersonalitySnapshot) -> float:
        """Calculate emotional resonance level"""
        agreeableness = snapshot.big_five_scores.get('agreeableness', 0.5)
        emotional_expr = snapshot.facet_scores.get('emotional_expressiveness', 0.5)
        
        return float((agreeableness + emotional_expr) / 2.0)
    
    def _determine_engagement_style(self, snapshot: PersonalitySnapshot, message: str) -> str:
        """Determine engagement style based on personality and message"""
        social_engagement = snapshot.facet_scores.get('social_engagement', 0.5)
        curiosity = snapshot.facet_scores.get('curiosity_score', 0.5)
        innovation = snapshot.facet_scores.get('innovation_drive', 0.5)
        
        message_lower = message.lower()
        
        # Check for collaborative language
        collaborative_words = ['we', 'us', 'together', 'collaborate', 'team']
        has_collaborative = any(word in message_lower for word in collaborative_words)
        
        # Check for questions
        has_questions = '?' in message
        
        if social_engagement > 0.7 and has_collaborative:
            return 'collaborative'
        elif curiosity > 0.7 and has_questions:
            return 'exploratory'
        elif innovation > 0.7:
            return 'creative'
        elif snapshot.facet_scores.get('analytical_thinking', 0.5) > 0.7:
            return 'analytical'
        else:
            return 'adaptive'
    
    def _suggest_response_approach(self, snapshot: PersonalitySnapshot, message: str) -> str:
        """Suggest specific response approach"""
        approaches = {
            'high_curiosity': "Encourage exploration and ask follow-up questions",
            'high_analytical': "Provide structured, logical responses with clear reasoning",
            'high_emotional': "Show empathy and emotional understanding", 
            'high_social': "Use inclusive language and collaborative framing",
            'high_creative': "Encourage creative thinking and new possibilities",
            'balanced': "Maintain a thoughtful, adaptive approach"
        }
        
        # Find dominant trait
        facet_scores = snapshot.facet_scores
        max_score = 0.0
        dominant_trait = 'balanced'
        
        trait_mappings = {
            'curiosity_score': 'high_curiosity',
            'analytical_thinking': 'high_analytical', 
            'emotional_expressiveness': 'high_emotional',
            'social_engagement': 'high_social',
            'innovation_drive': 'high_creative'
        }
        
        for trait, approach in trait_mappings.items():
            score = facet_scores.get(trait, 0.5)
            if score > max_score and score > 0.6:  # Threshold for dominance
                max_score = score
                dominant_trait = approach
        
        return approaches.get(dominant_trait, approaches['balanced'])

# Export the unified engine
__all__ = ['UnifiedPersonalityEngine', 'PersonalitySnapshot']

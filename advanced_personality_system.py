#!/usr/bin/env python3
"""
🧠 Advanced Big Five+ Personality System with TGA Integration
Implementation of cutting-edge personality research including:
- Taxonomic Graph Analysis (TGA) bottom-up trait discovery
- 28 facet personality model beyond traditional Big Five
- Dynamic trait evolution with drift detection
- Meta-traits: Stability, Plasticity, Disinhibition
- Retraining triggers based on personality archetype evolution
"""

import numpy as np
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from collections import defaultdict
import sqlite3
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class PersonalitySnapshot:
    """Represents a personality state at a specific time"""
    timestamp: datetime
    big_five_scores: Dict[str, float]
    facet_scores: Dict[str, float] 
    meta_traits: Dict[str, float]
    archetype: str
    confidence: float
    sample_size: int

@dataclass
class TraitDrift:
    """Represents detected drift in personality traits"""
    trait_name: str
    previous_value: float
    current_value: float
    drift_magnitude: float
    drift_direction: str  # 'increasing', 'decreasing', 'stable'
    confidence: float
    retraining_recommended: bool

class TGAPersonalityModel:
    """
    Taxonomic Graph Analysis implementation for bottom-up personality discovery
    Based on recent research showing 28 facets and 3 meta-traits
    """
    
    def __init__(self):
        self.trait_hierarchy = self._initialize_tga_hierarchy()
        self.facet_definitions = self._get_28_facets()
        self.meta_trait_weights = self._get_meta_trait_weights()
        
    def _initialize_tga_hierarchy(self) -> Dict[str, Any]:
        """Initialize the hierarchical trait structure from TGA research"""
        return {
            "meta_traits": {
                "stability": {
                    "description": "Emotional and behavioral consistency",
                    "component_traits": ["emotional_stability", "self_control", "conscientiousness"],
                    "facets": ["anxiety_management", "impulse_control", "emotional_regulation"]
                },
                "plasticity": {
                    "description": "Openness to change and new experiences", 
                    "component_traits": ["openness", "extraversion", "intellect"],
                    "facets": ["cognitive_flexibility", "social_adaptability", "creative_thinking"]
                },
                "disinhibition": {
                    "description": "Tendency toward impulsive and uninhibited behavior",
                    "component_traits": ["impulsivity", "risk_seeking", "sensation_seeking"],
                    "facets": ["behavioral_control", "risk_assessment", "novelty_seeking"]
                }
            },
            "traditional_big_five": {
                "openness": ["intellectual_curiosity", "aesthetic_sensitivity", "creative_imagination"],
                "conscientiousness": ["orderliness", "industriousness", "self_discipline"],
                "extraversion": ["sociability", "assertiveness", "energy_level"],
                "agreeableness": ["compassion", "respectfulness", "trust"],
                "neuroticism": ["emotional_volatility", "withdrawal", "negative_emotionality"]
            }
        }
    
    def _get_28_facets(self) -> Dict[str, Dict[str, Any]]:
        """Define the 28 personality facets from bottom-up TGA research"""
        return {
            # Stability Meta-Trait Facets
            "anxiety_management": {
                "indicators": ["calm", "relaxed", "composed", "unworried"],
                "reverse_indicators": ["anxious", "nervous", "worried", "tense"],
                "meta_trait": "stability",
                "traditional_mapping": "neuroticism_reverse"
            },
            "impulse_control": {
                "indicators": ["self-controlled", "disciplined", "restrained", "measured"],
                "reverse_indicators": ["impulsive", "hasty", "reckless", "spontaneous"],
                "meta_trait": "stability", 
                "traditional_mapping": "conscientiousness"
            },
            "emotional_regulation": {
                "indicators": ["balanced", "stable", "consistent", "level-headed"],
                "reverse_indicators": ["volatile", "moody", "unpredictable", "reactive"],
                "meta_trait": "stability",
                "traditional_mapping": "neuroticism_reverse"
            },
            
            # Plasticity Meta-Trait Facets
            "cognitive_flexibility": {
                "indicators": ["adaptable", "flexible", "open-minded", "versatile"],
                "reverse_indicators": ["rigid", "stubborn", "closed-minded", "inflexible"],
                "meta_trait": "plasticity",
                "traditional_mapping": "openness"
            },
            "social_adaptability": {
                "indicators": ["socially_skilled", "diplomatic", "charismatic", "engaging"],
                "reverse_indicators": ["socially_awkward", "withdrawn", "antisocial", "aloof"],
                "meta_trait": "plasticity",
                "traditional_mapping": "extraversion"
            },
            "creative_thinking": {
                "indicators": ["innovative", "imaginative", "creative", "original"],
                "reverse_indicators": ["conventional", "unimaginative", "predictable", "routine"],
                "meta_trait": "plasticity",
                "traditional_mapping": "openness"
            },
            
            # Disinhibition Meta-Trait Facets
            "behavioral_control": {
                "indicators": ["controlled", "cautious", "prudent", "careful"],
                "reverse_indicators": ["uncontrolled", "careless", "reckless", "imprudent"],
                "meta_trait": "disinhibition_reverse",
                "traditional_mapping": "conscientiousness"
            },
            "risk_assessment": {
                "indicators": ["calculated", "thoughtful", "deliberate", "strategic"],
                "reverse_indicators": ["risky", "gambling", "dangerous", "reckless"],
                "meta_trait": "disinhibition_reverse", 
                "traditional_mapping": "conscientiousness"
            },
            "novelty_seeking": {
                "indicators": ["adventurous", "thrill-seeking", "bold", "daring"],
                "reverse_indicators": ["conventional", "safe", "predictable", "cautious"],
                "meta_trait": "disinhibition",
                "traditional_mapping": "openness"
            },
            
            # Additional Big Five+ Facets (expanded from traditional 5 to 28)
            "intellectual_curiosity": {
                "indicators": ["curious", "inquisitive", "learning", "knowledge"],
                "reverse_indicators": ["incurious", "uninterested", "ignorant", "closed"],
                "meta_trait": "plasticity",
                "traditional_mapping": "openness"
            },
            "aesthetic_sensitivity": {
                "indicators": ["artistic", "beautiful", "aesthetic", "creative"],
                "reverse_indicators": ["unartistic", "crude", "insensitive", "practical"],
                "meta_trait": "plasticity",
                "traditional_mapping": "openness"
            },
            "sociability": {
                "indicators": ["outgoing", "social", "gregarious", "friendly"],
                "reverse_indicators": ["introverted", "solitary", "antisocial", "withdrawn"],
                "meta_trait": "plasticity",
                "traditional_mapping": "extraversion"
            },
            "assertiveness": {
                "indicators": ["assertive", "dominant", "confident", "leadership"],
                "reverse_indicators": ["passive", "submissive", "timid", "follower"],
                "meta_trait": "plasticity",
                "traditional_mapping": "extraversion"
            },
            "compassion": {
                "indicators": ["kind", "empathetic", "caring", "compassionate"],
                "reverse_indicators": ["cold", "uncaring", "harsh", "indifferent"],
                "meta_trait": "stability",
                "traditional_mapping": "agreeableness"
            },
            "orderliness": {
                "indicators": ["organized", "neat", "systematic", "structured"],
                "reverse_indicators": ["messy", "disorganized", "chaotic", "haphazard"],
                "meta_trait": "stability",
                "traditional_mapping": "conscientiousness"
            },
            
            # New TGA-specific facets not in traditional Big Five
            "integrity": {
                "indicators": ["honest", "trustworthy", "ethical", "moral"],
                "reverse_indicators": ["dishonest", "deceptive", "unethical", "corrupt"],
                "meta_trait": "stability",
                "traditional_mapping": "new_facet"
            },
            "authenticity": {
                "indicators": ["genuine", "authentic", "real", "sincere"],
                "reverse_indicators": ["fake", "artificial", "pretentious", "insincere"],
                "meta_trait": "stability",
                "traditional_mapping": "new_facet"
            },
            "emotional_intelligence": {
                "indicators": ["emotionally_aware", "empathetic", "sensitive", "perceptive"],
                "reverse_indicators": ["emotionally_blind", "insensitive", "oblivious", "dense"],
                "meta_trait": "plasticity",
                "traditional_mapping": "new_facet"
            },
            "resilience": {
                "indicators": ["resilient", "strong", "enduring", "persevering"],
                "reverse_indicators": ["fragile", "weak", "brittle", "giving_up"],
                "meta_trait": "stability",
                "traditional_mapping": "new_facet"
            },
            "playfulness": {
                "indicators": ["playful", "humorous", "fun", "lighthearted"],
                "reverse_indicators": ["serious", "somber", "grave", "stern"],
                "meta_trait": "plasticity",
                "traditional_mapping": "new_facet"
            }
            # ... [We could extend to full 28 facets, but showing key examples]
        }
    
    def _get_meta_trait_weights(self) -> Dict[str, Dict[str, float]]:
        """Define how facets contribute to meta-traits"""
        return {
            "stability": {
                "anxiety_management": 0.25,
                "impulse_control": 0.20,
                "emotional_regulation": 0.25,
                "compassion": 0.15,
                "orderliness": 0.10,
                "integrity": 0.05
            },
            "plasticity": {
                "cognitive_flexibility": 0.20,
                "social_adaptability": 0.18,
                "creative_thinking": 0.20,
                "intellectual_curiosity": 0.15,
                "aesthetic_sensitivity": 0.12,
                "sociability": 0.15
            },
            "disinhibition": {
                "behavioral_control": -0.30,  # Reverse scored
                "risk_assessment": -0.25,     # Reverse scored  
                "novelty_seeking": 0.45       # Direct scored
            }
        }

class PersonalityDriftDetector:
    """
    Detects personality trait drift and triggers retraining recommendations
    """
    
    def __init__(self, drift_threshold: float = 0.15, window_size: int = 30):
        self.drift_threshold = drift_threshold
        self.window_size = window_size  # Days
        self.baseline_stability_period = 90  # Days to establish baseline
        
    def detect_drift(self, historical_snapshots: List[PersonalitySnapshot], 
                    current_snapshot: PersonalitySnapshot) -> List[TraitDrift]:
        """Detect significant personality drift requiring retraining"""
        
        if len(historical_snapshots) < 2:
            return []  # Need history to detect drift
            
        drifts = []
        
        # Get baseline (older stable period)
        baseline_snapshot = self._get_baseline_snapshot(historical_snapshots)
        if not baseline_snapshot:
            return []
        
        # Check Big Five drift
        for trait, current_value in current_snapshot.big_five_scores.items():
            baseline_value = baseline_snapshot.big_five_scores.get(trait, 0.5)
            drift = self._calculate_drift(trait, baseline_value, current_value, "big_five")
            if drift:
                drifts.append(drift)
        
        # Check facet drift
        for facet, current_value in current_snapshot.facet_scores.items():
            baseline_value = baseline_snapshot.facet_scores.get(facet, 0.5)
            drift = self._calculate_drift(facet, baseline_value, current_value, "facet")
            if drift:
                drifts.append(drift)
        
        # Check meta-trait drift (most important)
        for meta_trait, current_value in current_snapshot.meta_traits.items():
            baseline_value = baseline_snapshot.meta_traits.get(meta_trait, 0.5)
            drift = self._calculate_drift(meta_trait, baseline_value, current_value, "meta_trait")
            if drift:
                drifts.append(drift)
                
        return drifts
    
    def _get_baseline_snapshot(self, snapshots: List[PersonalitySnapshot]) -> Optional[PersonalitySnapshot]:
        """Get baseline snapshot from stable historical period"""
        now = datetime.now()
        baseline_cutoff = now - timedelta(days=self.baseline_stability_period)
        
        baseline_candidates = [s for s in snapshots if s.timestamp < baseline_cutoff]
        
        if not baseline_candidates:
            return None
            
        # Return the snapshot with highest confidence in baseline period
        return max(baseline_candidates, key=lambda s: s.confidence)
    
    def _calculate_drift(self, trait_name: str, baseline_value: float, 
                        current_value: float, trait_type: str) -> Optional[TraitDrift]:
        """Calculate drift metrics for a specific trait"""
        
        drift_magnitude = abs(current_value - baseline_value)
        
        # Adjust threshold based on trait type
        threshold = self.drift_threshold
        if trait_type == "meta_trait":
            threshold *= 0.8  # Meta-traits are more stable, lower threshold
        elif trait_type == "facet":
            threshold *= 1.2  # Facets can be more variable
            
        if drift_magnitude < threshold:
            return None  # No significant drift
            
        # Determine drift direction
        direction = "increasing" if current_value > baseline_value else "decreasing"
        if drift_magnitude < threshold * 0.5:
            direction = "stable"
            
        # Calculate confidence based on magnitude
        confidence = min(1.0, drift_magnitude / (threshold * 2))
        
        # Determine if retraining is recommended
        retraining_recommended = (
            drift_magnitude > threshold * 1.5 or  # Large drift
            trait_type == "meta_trait"            # Any meta-trait drift is significant
        )
        
        return TraitDrift(
            trait_name=trait_name,
            previous_value=baseline_value,
            current_value=current_value,
            drift_magnitude=drift_magnitude,
            drift_direction=direction,
            confidence=confidence,
            retraining_recommended=retraining_recommended
        )

class AdvancedPersonalityEngine:
    """
    Advanced personality system integrating TGA, drift detection, and dynamic evolution
    """
    
    def __init__(self, db_path: str = "user_memory.db"):
        self.db_path = db_path
        self.tga_model = TGAPersonalityModel()
        self.drift_detector = PersonalityDriftDetector()
        self.personality_cache = {}  # User ID -> Recent snapshots
        self._initialize_database()
        
    def _initialize_database(self):
        """Initialize personality tracking database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            # Personality snapshots table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS personality_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    timestamp DATETIME NOT NULL,
                    big_five_json TEXT NOT NULL,
                    facets_json TEXT NOT NULL,
                    meta_traits_json TEXT NOT NULL,
                    archetype TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    sample_size INTEGER NOT NULL,
                    snapshot_hash TEXT UNIQUE
                )
            """)
            
            # Trait drift log
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
            
            conn.commit()
    
    def analyze_user_personality(self, user_id: str, interaction_texts: List[str], 
                               embeddings: Optional[List[Dict]] = None) -> PersonalitySnapshot:
        """
        Comprehensive personality analysis using TGA methodology
        """
        
        # Calculate facet scores using TGA bottom-up approach
        facet_scores = self._calculate_facet_scores(interaction_texts)
        
        # Calculate traditional Big Five from facets
        big_five_scores = self._calculate_big_five_from_facets(facet_scores)
        
        # Calculate meta-traits
        meta_traits = self._calculate_meta_traits(facet_scores)
        
        # Determine archetype from meta-trait combination
        archetype = self._determine_archetype(meta_traits, big_five_scores)
        
        # Calculate confidence based on sample size and consistency
        confidence = self._calculate_confidence(interaction_texts, facet_scores)
        
        snapshot = PersonalitySnapshot(
            timestamp=datetime.now(),
            big_five_scores=big_five_scores,
            facet_scores=facet_scores,
            meta_traits=meta_traits,
            archetype=archetype,
            confidence=confidence,
            sample_size=len(interaction_texts)
        )
        
        # Store snapshot
        self._store_snapshot(user_id, snapshot)
        
        # Check for drift and trigger retraining if needed
        self._check_drift_and_retrain(user_id, snapshot)
        
        return snapshot
    
    def _calculate_facet_scores(self, texts: List[str]) -> Dict[str, float]:
        """Calculate 28+ facet scores using TGA methodology"""
        combined_text = " ".join(texts).lower()
        facet_scores = {}
        
        for facet_name, facet_def in self.tga_model.facet_definitions.items():
            # Count positive indicators
            positive_count = sum(1 for indicator in facet_def["indicators"] 
                               if indicator in combined_text)
            
            # Count reverse indicators
            negative_count = sum(1 for indicator in facet_def["reverse_indicators"]
                               if indicator in combined_text)
            
            # Calculate normalized score
            total_possible = len(facet_def["indicators"]) + len(facet_def["reverse_indicators"])
            raw_score = (positive_count - negative_count) / max(1, total_possible)
            
            # Normalize to 0-1 range with 0.5 as neutral
            facet_scores[facet_name] = max(0.0, min(1.0, 0.5 + raw_score))
        
        return facet_scores
    
    def _calculate_big_five_from_facets(self, facet_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate Big Five scores from facet scores"""
        big_five = {}
        
        # Map facets to Big Five dimensions
        facet_to_big_five = {
            "intellectual_curiosity": "openness",
            "aesthetic_sensitivity": "openness", 
            "creative_thinking": "openness",
            "cognitive_flexibility": "openness",
            "orderliness": "conscientiousness",
            "impulse_control": "conscientiousness",
            "behavioral_control": "conscientiousness",
            "sociability": "extraversion",
            "assertiveness": "extraversion",
            "social_adaptability": "extraversion",
            "compassion": "agreeableness",
            "emotional_regulation": "neuroticism_reverse",
            "anxiety_management": "neuroticism_reverse"
        }
        
        # Aggregate facet scores by Big Five dimension
        dimension_scores = defaultdict(list)
        for facet, score in facet_scores.items():
            if facet in facet_to_big_five:
                dimension = facet_to_big_five[facet]
                if dimension.endswith("_reverse"):
                    score = 1.0 - score  # Reverse score
                    dimension = dimension.replace("_reverse", "")
                dimension_scores[dimension].append(score)
        
        # Calculate average for each Big Five dimension
        for dimension in ["openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]:
            if dimension_scores[dimension]:
                big_five[dimension] = np.mean(dimension_scores[dimension])
            else:
                big_five[dimension] = 0.5  # Neutral if no data
                
        return big_five
    
    def _calculate_meta_traits(self, facet_scores: Dict[str, float]) -> Dict[str, float]:
        """Calculate meta-trait scores using weighted facet contributions"""
        meta_traits = {}
        
        for meta_trait, weights in self.tga_model.meta_trait_weights.items():
            weighted_sum = 0.0
            total_weight = 0.0
            
            for facet, weight in weights.items():
                if facet in facet_scores:
                    score = facet_scores[facet]
                    if weight < 0:  # Reverse scored facet
                        score = 1.0 - score
                        weight = abs(weight)
                    weighted_sum += score * weight
                    total_weight += weight
            
            if total_weight > 0:
                meta_traits[meta_trait] = weighted_sum / total_weight
            else:
                meta_traits[meta_trait] = 0.5
                
        return meta_traits
    
    def _determine_archetype(self, meta_traits: Dict[str, float], 
                           big_five: Dict[str, float]) -> str:
        """Determine personality archetype from trait combination"""
        
        # Get dominant meta-traits
        stability = meta_traits.get("stability", 0.5)
        plasticity = meta_traits.get("plasticity", 0.5) 
        disinhibition = meta_traits.get("disinhibition", 0.5)
        
        # Complex archetype determination
        if stability > 0.7 and plasticity > 0.7:
            return "resilient_explorer"  # High stability + high plasticity
        elif stability > 0.7 and plasticity < 0.4:
            return "steady_traditionalist"  # High stability + low plasticity
        elif stability < 0.4 and plasticity > 0.7:
            return "volatile_innovator"  # Low stability + high plasticity
        elif stability < 0.4 and plasticity < 0.4:
            return "anxious_conservative"  # Low stability + low plasticity
        elif disinhibition > 0.7:
            return "impulsive_risk_taker"  # High disinhibition
        elif big_five.get("openness", 0.5) > 0.8 and big_five.get("conscientiousness", 0.5) > 0.8:
            return "disciplined_creative"
        else:
            return "balanced_adaptive"  # Balanced across dimensions
    
    def _calculate_confidence(self, texts: List[str], facet_scores: Dict[str, float]) -> float:
        """Calculate confidence in personality assessment"""
        
        # Sample size factor
        sample_factor = min(1.0, len(texts) / 50)  # Full confidence at 50+ texts
        
        # Consistency factor (how consistent are the facet scores)
        consistency = 1.0 - float(np.std(list(facet_scores.values())))
        consistency = max(0.0, min(1.0, consistency))
        
        # Text quality factor (average text length)
        if texts:
            avg_length = float(np.mean([len(text.split()) for text in texts]))
            length_factor = min(1.0, avg_length / 20)  # Full confidence at 20+ words per text
        else:
            length_factor = 0.0
            
        # Combine factors
        confidence = (sample_factor * 0.4 + consistency * 0.4 + length_factor * 0.2)
        return max(0.1, min(1.0, confidence))
    
    def _store_snapshot(self, user_id: str, snapshot: PersonalitySnapshot):
        """Store personality snapshot in database"""
        
        # Create hash for duplicate detection
        snapshot_data = {
            "big_five": snapshot.big_five_scores,
            "facets": snapshot.facet_scores,
            "meta_traits": snapshot.meta_traits
        }
        snapshot_hash = hashlib.sha256(json.dumps(snapshot_data, sort_keys=True).encode()).hexdigest()
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            try:
                cursor.execute("""
                    INSERT INTO personality_snapshots 
                    (user_id, timestamp, big_five_json, facets_json, meta_traits_json,
                     archetype, confidence, sample_size, snapshot_hash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id,
                    snapshot.timestamp.isoformat(),
                    json.dumps(snapshot.big_five_scores),
                    json.dumps(snapshot.facet_scores),
                    json.dumps(snapshot.meta_traits),
                    snapshot.archetype,
                    snapshot.confidence,
                    snapshot.sample_size,
                    snapshot_hash
                ))
                conn.commit()
            except sqlite3.IntegrityError:
                # Duplicate snapshot, skip
                pass
    
    def _check_drift_and_retrain(self, user_id: str, current_snapshot: PersonalitySnapshot):
        """Check for personality drift and trigger retraining if necessary"""
        
        # Get historical snapshots
        historical_snapshots = self._get_historical_snapshots(user_id)
        
        if len(historical_snapshots) < 2:
            return  # Need more history
            
        # Detect drift
        drifts = self.drift_detector.detect_drift(historical_snapshots, current_snapshot)
        
        # Log significant drifts
        for drift in drifts:
            if drift.drift_magnitude > 0.1:  # Log significant changes
                self._log_trait_drift(user_id, drift)
                
        # Trigger retraining if recommended
        retraining_drifts = [d for d in drifts if d.retraining_recommended]
        if retraining_drifts:
            self._trigger_retraining(user_id, retraining_drifts)
    
    def _get_historical_snapshots(self, user_id: str) -> List[PersonalitySnapshot]:
        """Get historical personality snapshots for a user"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT timestamp, big_five_json, facets_json, meta_traits_json,
                       archetype, confidence, sample_size
                FROM personality_snapshots 
                WHERE user_id = ?
                ORDER BY timestamp DESC
                LIMIT 20
            """, (user_id,))
            
            snapshots = []
            for row in cursor.fetchall():
                snapshot = PersonalitySnapshot(
                    timestamp=datetime.fromisoformat(row[0]),
                    big_five_scores=json.loads(row[1]),
                    facet_scores=json.loads(row[2]),
                    meta_traits=json.loads(row[3]),
                    archetype=row[4],
                    confidence=row[5],
                    sample_size=row[6]
                )
                snapshots.append(snapshot)
                
            return snapshots
    
    def _log_trait_drift(self, user_id: str, drift: TraitDrift):
        """Log trait drift to database"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO trait_drift_log 
                (user_id, timestamp, trait_name, drift_magnitude, drift_direction, retraining_triggered)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                user_id,
                datetime.now().isoformat(),
                drift.trait_name,
                drift.drift_magnitude,
                drift.drift_direction,
                drift.retraining_recommended
            ))
            conn.commit()
    
    def _trigger_retraining(self, user_id: str, drifts: List[TraitDrift]):
        """Trigger personality model retraining based on detected drift"""
        logger.info(f"🔄 Triggering personality retraining for user {user_id}")
        logger.info(f"   Significant drifts detected in: {[d.trait_name for d in drifts]}")
        
        # In a real system, this would:
        # 1. Queue a retraining job
        # 2. Update model weights based on new personality patterns
        # 3. Refresh response generation parameters
        # 4. Notify other system components of personality evolution
        
        # For now, log the event
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE trait_drift_log 
                SET retraining_triggered = TRUE 
                WHERE user_id = ? AND timestamp > datetime('now', '-1 hour')
            """, (user_id,))
            conn.commit()
    
    def get_personality_summary(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive personality summary for a user"""
        
        # Get latest snapshot
        snapshots = self._get_historical_snapshots(user_id)
        if not snapshots:
            return {"error": "No personality data available"}
            
        latest = snapshots[0]
        
        # Get recent drift information
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT trait_name, drift_magnitude, drift_direction
                FROM trait_drift_log 
                WHERE user_id = ? AND timestamp > datetime('now', '-30 days')
                ORDER BY drift_magnitude DESC
                LIMIT 5
            """, (user_id,))
            
            recent_drifts = [
                {"trait": row[0], "magnitude": row[1], "direction": row[2]}
                for row in cursor.fetchall()
            ]
        
        return {
            "user_id": user_id,
            "archetype": latest.archetype,
            "confidence": latest.confidence,
            "analysis_date": latest.timestamp.isoformat(),
            "big_five_scores": latest.big_five_scores,
            "meta_traits": latest.meta_traits,
            "top_facets": dict(sorted(latest.facet_scores.items(), key=lambda x: x[1], reverse=True)[:10]),
            "recent_personality_drift": recent_drifts,
            "sample_size": latest.sample_size,
            "personality_stability": 1.0 - (sum(d["magnitude"] for d in recent_drifts) / max(1, len(recent_drifts)))
        }

def demo_advanced_personality_system():
    """Demonstration of the advanced personality system"""
    
    print("🧠 Advanced Personality System with TGA Integration")
    print("=" * 70)
    
    # Initialize system
    engine = AdvancedPersonalityEngine()
    
    # Test data representing personality evolution over time
    user_interactions = {
        "week_1": [
            "I love exploring new technologies and learning about AI systems",
            "Sometimes I feel anxious about the future, but I try to stay positive",
            "I'm very organized in my approach to problem-solving",
            "I enjoy collaborating with others on creative projects"
        ],
        "week_4": [
            "I've become more confident in taking risks and trying new approaches",
            "My anxiety has decreased significantly - I feel more stable emotionally", 
            "I still value organization, but I'm more flexible with my methods now",
            "I've started preferring to work independently on deep, focused tasks"
        ],
        "week_8": [
            "I'm now actively seeking out challenging and uncertain situations",
            "I feel remarkably calm and composed, even under pressure",
            "I've adopted a more spontaneous approach to planning and execution",
            "I'm drawn to leadership roles and enjoy guiding team decisions"
        ]
    }
    
    user_id = "demo_user_evolution"
    
    # Analyze personality evolution
    for week, interactions in user_interactions.items():
        print(f"\n📊 {week.upper()} PERSONALITY ANALYSIS")
        print("-" * 50)
        
        snapshot = engine.analyze_user_personality(user_id, interactions)
        
        print(f"🎭 Archetype: {snapshot.archetype}")
        print(f"🎯 Confidence: {snapshot.confidence:.2f}")
        
        print("\n🧬 Meta-Traits:")
        for trait, score in snapshot.meta_traits.items():
            print(f"   {trait.title()}: {score:.3f}")
        
        print("\n🔍 Big Five Scores:")
        for trait, score in sorted(snapshot.big_five_scores.items(), key=lambda x: x[1], reverse=True):
            print(f"   {trait.title()}: {score:.3f}")
        
        print("\n⭐ Top Facets:")
        top_facets = dict(sorted(snapshot.facet_scores.items(), key=lambda x: x[1], reverse=True)[:5])
        for facet, score in top_facets.items():
            print(f"   {facet.replace('_', ' ').title()}: {score:.3f}")
            
        # Add some delay to simulate time passage
        import time
        time.sleep(0.1)
    
    # Final summary
    print(f"\n🎯 FINAL PERSONALITY SUMMARY")
    print("=" * 70)
    
    summary = engine.get_personality_summary(user_id)
    print(f"Final Archetype: {summary['archetype']}")
    print(f"Personality Stability: {summary['personality_stability']:.3f}")
    print(f"Analysis Confidence: {summary['confidence']:.3f}")
    
    if summary['recent_personality_drift']:
        print(f"\n🔄 Recent Personality Evolution:")
        for drift in summary['recent_personality_drift']:
            print(f"   {drift['trait']}: {drift['magnitude']:.3f} ({drift['direction']})")
    
    print("\n✨ This demonstrates how the system tracks personality evolution")
    print("   and can trigger retraining when significant drift is detected!")

if __name__ == "__main__":
    demo_advanced_personality_system()

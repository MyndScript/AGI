#!/usr/bin/env python3
"""
🧬 Longitudinal Trait Tracking & Archetype Evolution System
Tracks personality changes over time and triggers archetype transitions
"""

import numpy as np
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

class ArchetypeEvolutionEngine:
    """Manages personality archetype transitions based on trait evolution"""
    
    def __init__(self, memory_db_path: str = "memory/user_memory.db"):
        self.memory_db_path = memory_db_path
        self.init_tracking_database()
        
        # Define archetype boundaries based on trait combinations
        self.archetypes = {
            'strategist': {
                'required_traits': {
                    'analytical_thinking': 0.7,
                    'detail_orientation': 0.6,
                    'emotional_expressiveness': 0.4  # Low emotional, high analytical
                },
                'description': 'Logical, systematic, plans ahead',
                'response_style': 'structured and methodical'
            },
            'explorer': {
                'required_traits': {
                    'curiosity_score': 0.7,
                    'innovation_drive': 0.6,
                    'social_engagement': 0.5
                },
                'description': 'Curious, adventurous, seeks new experiences',
                'response_style': 'enthusiastic and exploratory'
            },
            'nurturer': {
                'required_traits': {
                    'emotional_expressiveness': 0.7,
                    'social_engagement': 0.7,
                    'analytical_thinking': 0.4  # High emotional, moderate analytical
                },
                'description': 'Caring, supportive, emotionally attuned',
                'response_style': 'warm and empathetic'
            },
            'innovator': {
                'required_traits': {
                    'innovation_drive': 0.8,
                    'curiosity_score': 0.6,
                    'analytical_thinking': 0.6
                },
                'description': 'Creative, forward-thinking, solution-oriented',
                'response_style': 'inspiring and visionary'
            },
            'sage': {
                'required_traits': {
                    'detail_orientation': 0.6,
                    'analytical_thinking': 0.5,
                    'emotional_expressiveness': 0.5,
                    'social_engagement': 0.6
                },
                'description': 'Balanced, wise, thoughtful',
                'response_style': 'measured and insightful'
            },
            'catalyst': {
                'required_traits': {
                    'social_engagement': 0.8,
                    'innovation_drive': 0.7,
                    'emotional_expressiveness': 0.6
                },
                'description': 'Energetic, inspiring, brings change',
                'response_style': 'dynamic and motivating'
            }
        }
        
        # Transition thresholds
        self.transition_thresholds = {
            'minor_drift': 0.15,    # 15% trait change
            'moderate_shift': 0.3,  # 30% trait change
            'major_evolution': 0.5  # 50% trait change - triggers archetype transition
        }

    def init_tracking_database(self):
        """Initialize tables for longitudinal trait tracking"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Create trait tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trait_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trait_name TEXT NOT NULL,
                    trait_value REAL NOT NULL,
                    interaction_context TEXT,
                    timestamp BIGINT NOT NULL,
                    session_id TEXT,
                    FOREIGN KEY(user_id) REFERENCES user_memory(user_id)
                )
            """)
            
            # Create archetype evolution table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS archetype_evolution (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    previous_archetype TEXT,
                    new_archetype TEXT NOT NULL,
                    transition_reason TEXT,
                    confidence_score REAL,
                    trait_snapshot TEXT,  -- JSON of traits at transition
                    timestamp BIGINT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES user_memory(user_id)
                )
            """)
            
            # Create trait evolution events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS trait_evolution_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    trait_name TEXT NOT NULL,
                    old_value REAL,
                    new_value REAL,
                    change_magnitude REAL,
                    change_direction TEXT, -- 'increase', 'decrease', 'stable'
                    event_type TEXT, -- 'minor_drift', 'moderate_shift', 'major_evolution'
                    context TEXT,
                    timestamp BIGINT NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES user_memory(user_id)
                )
            """)
            
            # Create indices for performance
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trait_timeline_user_time ON trait_timeline(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_archetype_evolution_user_time ON archetype_evolution(user_id, timestamp)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trait_events_user_time ON trait_evolution_events(user_id, timestamp)")
            
            conn.commit()
            conn.close()
            logger.info("Trait tracking database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking database: {e}")

    def record_trait_scores(self, user_id: str, trait_scores: Dict[str, float], 
                           interaction_context: str = "", session_id: str = ""):
        """Record trait scores for longitudinal tracking"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            timestamp = int(datetime.now().timestamp())
            
            for trait_name, trait_value in trait_scores.items():
                cursor.execute("""
                    INSERT INTO trait_timeline (user_id, trait_name, trait_value, interaction_context, timestamp, session_id)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (user_id, trait_name, trait_value, interaction_context, timestamp, session_id))
            
            conn.commit()
            conn.close()
            
            # Check for trait evolution
            self._detect_trait_evolution(user_id, trait_scores)
            
        except Exception as e:
            logger.error(f"Failed to record trait scores: {e}")

    def _detect_trait_evolution(self, user_id: str, current_traits: Dict[str, float]):
        """Detect significant changes in trait patterns"""
        try:
            # Get historical trait averages for comparison
            historical_traits = self.get_trait_averages(user_id, days_back=30)
            
            if not historical_traits:
                return  # No historical data to compare
            
            evolution_events = []
            
            for trait_name, current_value in current_traits.items():
                if trait_name in historical_traits:
                    historical_value = historical_traits[trait_name]
                    change_magnitude = abs(current_value - historical_value)
                    
                    # Determine change significance
                    event_type = 'stable'
                    if change_magnitude > self.transition_thresholds['major_evolution']:
                        event_type = 'major_evolution'
                    elif change_magnitude > self.transition_thresholds['moderate_shift']:
                        event_type = 'moderate_shift'
                    elif change_magnitude > self.transition_thresholds['minor_drift']:
                        event_type = 'minor_drift'
                    
                    if event_type != 'stable':
                        change_direction = 'increase' if current_value > historical_value else 'decrease'
                        
                        evolution_events.append({
                            'trait_name': trait_name,
                            'old_value': historical_value,
                            'new_value': current_value,
                            'change_magnitude': change_magnitude,
                            'change_direction': change_direction,
                            'event_type': event_type
                        })
            
            # Record evolution events
            if evolution_events:
                self._record_evolution_events(user_id, evolution_events)
                
                # Check if archetype transition is needed
                major_changes = [e for e in evolution_events if e['event_type'] == 'major_evolution']
                if major_changes:
                    self._evaluate_archetype_transition(user_id, current_traits, major_changes)
            
        except Exception as e:
            logger.error(f"Error detecting trait evolution: {e}")

    def _record_evolution_events(self, user_id: str, events: List[Dict]):
        """Record trait evolution events in database"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            timestamp = int(datetime.now().timestamp())
            
            for event in events:
                cursor.execute("""
                    INSERT INTO trait_evolution_events 
                    (user_id, trait_name, old_value, new_value, change_magnitude, change_direction, event_type, context, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user_id, event['trait_name'], event['old_value'], event['new_value'],
                    event['change_magnitude'], event['change_direction'], event['event_type'],
                    f"Trait evolution detected: {event['change_direction']} of {event['change_magnitude']:.3f}",
                    timestamp
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"Recorded {len(events)} trait evolution events for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to record evolution events: {e}")

    def _evaluate_archetype_transition(self, user_id: str, current_traits: Dict[str, float], major_changes: List[Dict]):
        """Evaluate if user should transition to a different archetype"""
        try:
            # Get current archetype
            current_archetype = self.get_current_archetype(user_id)
            
            # Calculate archetype scores based on current traits
            archetype_scores = {}
            for archetype_name, archetype_def in self.archetypes.items():
                score = self._calculate_archetype_match(current_traits, archetype_def['required_traits'])
                archetype_scores[archetype_name] = score
            
            # Find best matching archetype
            best_archetype = max(archetype_scores.items(), key=lambda x: x[1])
            best_archetype_name, best_score = best_archetype
            
            # Check if transition is warranted (confidence > 0.7 and different from current)
            if best_score > 0.7 and best_archetype_name != current_archetype:
                self._execute_archetype_transition(
                    user_id, current_archetype, best_archetype_name, 
                    best_score, current_traits, major_changes
                )
            
        except Exception as e:
            logger.error(f"Error evaluating archetype transition: {e}")

    def _calculate_archetype_match(self, user_traits: Dict[str, float], required_traits: Dict[str, float]) -> float:
        """Calculate how well user traits match archetype requirements"""
        matches = []
        
        for trait_name, required_value in required_traits.items():
            user_value = user_traits.get(trait_name, 0.0)
            
            # For traits that should be low (like emotional_expressiveness for strategist)
            if required_value < 0.5:
                # Score higher when user value is closer to required low value
                match = 1.0 - abs(user_value - required_value)
            else:
                # For traits that should be high, score based on meeting threshold
                match = min(user_value / required_value, 1.0) if required_value > 0 else 0.0
            
            matches.append(max(0.0, match))
        
        return float(np.mean(matches)) if matches else 0.0

    def _execute_archetype_transition(self, user_id: str, old_archetype: str, new_archetype: str,
                                    confidence: float, trait_snapshot: Dict[str, float], 
                                    trigger_changes: List[Dict]):
        """Execute archetype transition and record it"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            timestamp = int(datetime.now().timestamp())
            
            # Create transition reason
            major_traits = [change['trait_name'] for change in trigger_changes]
            transition_reason = f"Major changes in traits: {', '.join(major_traits[:3])}"
            
            # Record the transition
            cursor.execute("""
                INSERT INTO archetype_evolution 
                (user_id, previous_archetype, new_archetype, transition_reason, confidence_score, trait_snapshot, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                user_id, old_archetype, new_archetype, transition_reason,
                confidence, json.dumps(trait_snapshot), timestamp
            ))
            
            conn.commit()
            conn.close()
            
            # Update personality system with new archetype
            self._update_personality_archetype(user_id, new_archetype)
            
            logger.info(f"Archetype transition: {user_id} {old_archetype} -> {new_archetype} (confidence: {confidence:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to execute archetype transition: {e}")

    def _update_personality_archetype(self, user_id: str, new_archetype: str):
        """Update the personality system with new archetype"""
        try:
            # Update via memory server API (if available)
            response = requests.post("http://localhost:8001/personality/update-archetype", 
                                   json={
                                       'user_id': user_id,
                                       'archetype': new_archetype
                                   }, timeout=5)
            
            if response.status_code == 200:
                logger.info(f"Updated archetype for {user_id} to {new_archetype}")
            else:
                logger.warning(f"Failed to update archetype via API: {response.text}")
                
        except Exception as e:
            logger.warning(f"Could not update archetype via API: {e}")

    def get_current_archetype(self, user_id: str) -> str:
        """Get user's current archetype"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT new_archetype FROM archetype_evolution 
                WHERE user_id = ? 
                ORDER BY timestamp DESC 
                LIMIT 1
            """, (user_id,))
            
            result = cursor.fetchone()
            conn.close()
            
            return result[0] if result else 'sage'  # Default archetype
            
        except Exception as e:
            logger.error(f"Error getting current archetype: {e}")
            return 'sage'

    def get_trait_averages(self, user_id: str, days_back: int = 30) -> Dict[str, float]:
        """Get average trait scores over specified time period"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            cursor.execute("""
                SELECT trait_name, AVG(trait_value) as avg_value 
                FROM trait_timeline 
                WHERE user_id = ? AND timestamp > ?
                GROUP BY trait_name
            """, (user_id, cutoff_time))
            
            results = cursor.fetchall()
            conn.close()
            
            return {trait_name: avg_value for trait_name, avg_value in results}
            
        except Exception as e:
            logger.error(f"Error getting trait averages: {e}")
            return {}

    def generate_evolution_report(self, user_id: str, days_back: int = 30) -> Dict[str, Any]:
        """Generate comprehensive personality evolution report"""
        try:
            # Get trait evolution events
            evolution_events = self.get_evolution_events(user_id, days_back)
            
            # Get archetype history
            archetype_history = self.get_archetype_history(user_id, days_back)
            
            # Get current trait averages
            current_traits = self.get_trait_averages(user_id, days_back=7)  # Recent average
            historical_traits = self.get_trait_averages(user_id, days_back=days_back)
            
            # Calculate overall evolution metrics
            evolution_metrics = self._calculate_evolution_metrics(evolution_events)
            
            # Current archetype info
            current_archetype = self.get_current_archetype(user_id)
            archetype_info = self.archetypes.get(current_archetype, {})
            
            return {
                'user_id': user_id,
                'report_period_days': days_back,
                'current_archetype': {
                    'name': current_archetype,
                    'description': archetype_info.get('description', 'Unknown'),
                    'response_style': archetype_info.get('response_style', 'Adaptive')
                },
                'trait_evolution': {
                    'current_traits': current_traits,
                    'historical_traits': historical_traits,
                    'evolution_events': evolution_events[:10],  # Latest 10 events
                    'total_events': len(evolution_events)
                },
                'archetype_history': archetype_history,
                'evolution_metrics': evolution_metrics,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating evolution report: {e}")
            return {'error': str(e)}

    def get_evolution_events(self, user_id: str, days_back: int = 30) -> List[Dict]:
        """Get trait evolution events for user"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            cursor.execute("""
                SELECT trait_name, old_value, new_value, change_magnitude, 
                       change_direction, event_type, context, timestamp
                FROM trait_evolution_events 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, cutoff_time))
            
            results = cursor.fetchall()
            conn.close()
            
            events = []
            for row in results:
                events.append({
                    'trait_name': row[0],
                    'old_value': row[1],
                    'new_value': row[2],
                    'change_magnitude': row[3],
                    'change_direction': row[4],
                    'event_type': row[5],
                    'context': row[6],
                    'timestamp': row[7]
                })
            
            return events
            
        except Exception as e:
            logger.error(f"Error getting evolution events: {e}")
            return []

    def get_archetype_history(self, user_id: str, days_back: int = 30) -> List[Dict]:
        """Get archetype transition history"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            cutoff_time = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            cursor.execute("""
                SELECT previous_archetype, new_archetype, transition_reason, 
                       confidence_score, timestamp
                FROM archetype_evolution 
                WHERE user_id = ? AND timestamp > ?
                ORDER BY timestamp DESC
            """, (user_id, cutoff_time))
            
            results = cursor.fetchall()
            conn.close()
            
            history = []
            for row in results:
                history.append({
                    'from': row[0],
                    'to': row[1],
                    'reason': row[2],
                    'confidence': row[3],
                    'timestamp': row[4]
                })
            
            return history
            
        except Exception as e:
            logger.error(f"Error getting archetype history: {e}")
            return []

    def _calculate_evolution_metrics(self, events: List[Dict]) -> Dict[str, Any]:
        """Calculate overall evolution metrics"""
        if not events:
            return {'stability_score': 1.0, 'change_frequency': 0.0, 'dominant_traits': []}
        
        # Stability score (inverse of change frequency)
        total_changes = len(events)
        major_changes = len([e for e in events if e['event_type'] == 'major_evolution'])
        stability_score = max(0.0, 1.0 - (major_changes / max(total_changes, 1)))
        
        # Change frequency (changes per day)
        if events:
            time_span_days = (events[0]['timestamp'] - events[-1]['timestamp']) / (24 * 3600)
            change_frequency = total_changes / max(time_span_days, 1)
        else:
            change_frequency = 0.0
        
        # Dominant changing traits
        trait_changes = defaultdict(int)
        for event in events:
            trait_changes[event['trait_name']] += 1
        
        dominant_traits = sorted(trait_changes.items(), key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'stability_score': stability_score,
            'change_frequency': change_frequency,
            'total_events': total_changes,
            'major_evolution_events': major_changes,
            'dominant_changing_traits': [{'trait': trait, 'change_count': count} for trait, count in dominant_traits]
        }


def demo_longitudinal_tracking():
    """Demonstration of longitudinal trait tracking and archetype evolution"""
    print("🧬 Longitudinal Trait Tracking & Archetype Evolution Demo")
    print("=" * 70)
    
    evolution_engine = ArchetypeEvolutionEngine()
    user_id = "test_user_123"
    
    # Simulate trait evolution over time
    trait_scenarios = [
        {
            'day': 1,
            'traits': {'curiosity_score': 0.8, 'analytical_thinking': 0.3, 'emotional_expressiveness': 0.6},
            'context': 'High curiosity phase - asking lots of questions'
        },
        {
            'day': 7,
            'traits': {'curiosity_score': 0.7, 'analytical_thinking': 0.6, 'emotional_expressiveness': 0.4},
            'context': 'Becoming more analytical, less emotional'
        },
        {
            'day': 14,
            'traits': {'curiosity_score': 0.6, 'analytical_thinking': 0.8, 'emotional_expressiveness': 0.2},
            'context': 'Strong shift toward analytical thinking'
        },
        {
            'day': 21,
            'traits': {'curiosity_score': 0.5, 'analytical_thinking': 0.9, 'emotional_expressiveness': 0.1, 'detail_orientation': 0.8},
            'context': 'Becoming highly systematic and detail-oriented'
        }
    ]
    
    print("📈 SIMULATING TRAIT EVOLUTION...")
    for scenario in trait_scenarios:
        print(f"Day {scenario['day']}: {scenario['context']}")
        evolution_engine.record_trait_scores(
            user_id, 
            scenario['traits'], 
            scenario['context'],
            f"session_day_{scenario['day']}"
        )
        
        # Check current archetype
        current_archetype = evolution_engine.get_current_archetype(user_id)
        print(f"  Current Archetype: {current_archetype}")
        print()
    
    print("📊 GENERATING EVOLUTION REPORT...")
    print("-" * 50)
    
    report = evolution_engine.generate_evolution_report(user_id, days_back=30)
    
    # Display current archetype
    archetype_info = report['current_archetype']
    print(f"🎭 CURRENT ARCHETYPE: {archetype_info['name'].upper()}")
    print(f"   Description: {archetype_info['description']}")
    print(f"   Response Style: {archetype_info['response_style']}")
    print()
    
    # Display trait evolution
    trait_evolution = report['trait_evolution']
    print("🔬 TRAIT EVOLUTION:")
    print(f"   Total Evolution Events: {trait_evolution['total_events']}")
    
    current_traits = trait_evolution['current_traits']
    for trait, value in sorted(current_traits.items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"   {trait}: {value:.3f}")
    print()
    
    # Display recent evolution events
    events = trait_evolution['evolution_events'][:3]
    print("📈 RECENT EVOLUTION EVENTS:")
    for event in events:
        direction_icon = "📈" if event['change_direction'] == 'increase' else "📉"
        print(f"   {direction_icon} {event['trait_name']}: {event['old_value']:.3f} → {event['new_value']:.3f} ({event['event_type']})")
    print()
    
    # Display archetype history
    archetype_history = report['archetype_history']
    if archetype_history:
        print("🔄 ARCHETYPE TRANSITIONS:")
        for transition in archetype_history:
            print(f"   {transition['from']} → {transition['to']} (confidence: {transition['confidence']:.3f})")
            print(f"      Reason: {transition['reason']}")
    else:
        print("🔄 ARCHETYPE TRANSITIONS: None detected")
    print()
    
    # Display evolution metrics
    metrics = report['evolution_metrics']
    print("📊 EVOLUTION METRICS:")
    print(f"   Stability Score: {metrics['stability_score']:.3f}")
    print(f"   Change Frequency: {metrics['change_frequency']:.3f} changes/day")
    print(f"   Major Evolution Events: {metrics['major_evolution_events']}")
    
    dominant_traits = metrics['dominant_changing_traits']
    if dominant_traits:
        print("   Most Changing Traits:")
        for trait_info in dominant_traits:
            print(f"     {trait_info['trait']}: {trait_info['change_count']} changes")
    
    print("\n" + "=" * 70)
    print("🎯 This system now provides:")
    print("• Longitudinal tracking of personality traits")
    print("• Automatic archetype transition detection")
    print("• Evolution metrics and stability scoring")
    print("• Complete audit trail of personality changes")
    print("• Foundation for adaptive AI that grows with users")

if __name__ == "__main__":
    demo_longitudinal_tracking()

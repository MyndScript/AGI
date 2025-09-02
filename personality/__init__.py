"""
🧠 AGI Personality System
Unified personality analysis with trait tracking and archetype evolution
"""

# Legacy compatibility - now points to legacy file
from .legacy_personality import Personality

# New unified system (recommended)
from .unified_personality_engine import UnifiedPersonalityEngine

__all__ = [
    'Personality',  # Legacy system
    'UnifiedPersonalityEngine'  # New unified system
]

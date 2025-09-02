#!/usr/bin/env python3
"""
🧠 Complete AGI Personality Analysis Integration
Brings together embedding analysis, trait scoring, and longitudinal tracking
"""

from embedding_personality_analyzer import EmbeddingAnalyzer
from trait_scoring_engine import AdvancedPersonalityEngine
from longitudinal_trait_tracker import ArchetypeEvolutionEngine
from datetime import datetime
from typing import Dict, List, Any

class IntegratedPersonalitySystem:
    """Complete integrated personality analysis system"""
    
    def __init__(self):
        self.embedding_analyzer = EmbeddingAnalyzer()
        self.personality_engine = AdvancedPersonalityEngine(self.embedding_analyzer)
        self.evolution_engine = ArchetypeEvolutionEngine()
        
    def process_user_interaction(self, user_id: str, user_message: str, session_id: str = "") -> Dict[str, Any]:
        """Complete processing pipeline for user interaction"""
        
        print(f"🔄 Processing interaction for {user_id}")
        print(f"📝 Message: {user_message}")
        print("-" * 50)
        
        # Step 1: Comprehensive personality analysis
        analysis = self.personality_engine.analyze_user_interaction(user_id, user_message)
        
        # Step 2: Record traits for longitudinal tracking  
        trait_scores = analysis['message_traits']
        self.evolution_engine.record_trait_scores(
            user_id, trait_scores, user_message[:100], session_id
        )
        
        # Step 3: Get current archetype and evolution status
        current_archetype = self.evolution_engine.get_current_archetype(user_id)
        evolution_report = self.evolution_engine.generate_evolution_report(user_id, days_back=30)
        
        # Step 4: Generate response recommendations
        response_modulation = analysis['response_modulation']
        
        return {
            'user_id': user_id,
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'analysis': {
                'message_traits': trait_scores,
                'glyph_matches': analysis['glyph_matches'],
                'embedding_analysis': analysis['embedding_analysis'],
                'current_archetype': current_archetype,
                'evolution_metrics': evolution_report.get('evolution_metrics', {})
            },
            'response_guidance': {
                'glyph_type': response_modulation['glyph_type'],
                'emotional_intensity': response_modulation['tone_guidance']['emotional_intensity'],
                'pacing': response_modulation['tone_guidance']['pacing'],
                'suggested_template': response_modulation['suggested_template'],
                'archetype_style': evolution_report['current_archetype']['response_style']
            },
            'insights': self._generate_insights(analysis, evolution_report)
        }
    
    def _generate_insights(self, analysis: Dict, evolution_report: Dict) -> List[str]:
        """Generate actionable insights from the analysis"""
        insights = []
        
        # Trait insights
        traits = analysis['message_traits']
        high_traits = [trait for trait, score in traits.items() if score > 0.5]
        if high_traits:
            insights.append(f"🎯 Strong traits detected: {', '.join(high_traits[:3])}")
        
        # Glyph insights
        glyph_matches = analysis['glyph_matches']
        if glyph_matches and glyph_matches[0]['similarity'] > 0.6:
            insights.append(f"🔮 Strong semantic resonance with past content (similarity: {glyph_matches[0]['similarity']:.3f})")
        
        # Evolution insights
        evolution_metrics = evolution_report.get('evolution_metrics', {})
        stability = evolution_metrics.get('stability_score', 1.0)
        if stability < 0.5:
            insights.append("🌊 Significant personality evolution detected - consider archetype adjustment")
        elif stability > 0.8:
            insights.append("⚖️ Personality profile is stable and consistent")
        
        # Archetype insights
        current_archetype = evolution_report.get('current_archetype', {})
        if current_archetype:
            insights.append(f"🎭 Current archetype: {current_archetype['name']} - {current_archetype['description']}")
        
        return insights

def demonstrate_complete_system():
    """Comprehensive demonstration of the integrated personality system"""
    
    print("🧠 Complete AGI Personality Analysis System")
    print("=" * 60)
    print("🎯 This system provides:")
    print("• Real-time personality trait analysis")
    print("• Semantic glyph matching and response modulation")
    print("• Longitudinal trait tracking and archetype evolution")
    print("• Comprehensive insights and response guidance")
    print("=" * 60)
    print()
    
    system = IntegratedPersonalitySystem()
    
    # Test scenarios showing different personality expressions
    test_scenarios = [
        {
            'user_id': 'demo_user_001',
            'session_id': 'session_1',
            'message': 'I love analyzing complex problems and finding systematic solutions. Can you help me understand how machine learning algorithms optimize their parameters?',
            'expected_archetype': 'strategist/analytical'
        },
        {
            'user_id': 'demo_user_001', 
            'session_id': 'session_1',
            'message': 'Actually, I feel overwhelmed by all this technical stuff. I just want to connect with others and understand their experiences.',
            'expected_archetype': 'potential shift toward nurturer'
        },
        {
            'user_id': 'demo_user_001',
            'session_id': 'session_2', 
            'message': 'You know what? I have this amazing vision for how AI could help people be more creative and expressive. What do you think about building tools that amplify human imagination?',
            'expected_archetype': 'potential shift toward innovator'
        }
    ]
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"🔍 SCENARIO {i}")
        print(f"Expected: {scenario['expected_archetype']}")
        print()
        
        result = system.process_user_interaction(
            scenario['user_id'],
            scenario['message'], 
            scenario['session_id']
        )
        
        # Display key results
        analysis = result['analysis']
        response_guidance = result['response_guidance']
        insights = result['insights']
        
        print(f"🎭 Current Archetype: {analysis['current_archetype']}")
        print(f"🔮 Glyph Type: {response_guidance['glyph_type']}")
        print(f"💡 Response Style: {response_guidance['archetype_style']}")
        print(f"🎵 Emotional Intensity: {response_guidance['emotional_intensity']:.2f}")
        print(f"⏱️  Pacing: {response_guidance['pacing']}")
        
        print("\n📊 Key Insights:")
        for insight in insights:
            print(f"   {insight}")
        
        print(f"\n💬 Suggested Response Template:")
        print(f"   {response_guidance['suggested_template']}")
        
        print("\n" + "=" * 60 + "\n")
    
    print("🎯 SYSTEM CAPABILITIES ACHIEVED:")
    print("✅ Real-time embedding-based personality analysis") 
    print("✅ Advanced trait scoring with 8+ personality dimensions")
    print("✅ Semantic glyph matching for contextual responses")
    print("✅ Longitudinal trait tracking and evolution detection")
    print("✅ Automatic archetype transition management")
    print("✅ Response modulation based on personality insights")
    print("✅ Complete audit trail of personality development")
    print()
    print("🚀 WHAT THIS ENABLES:")
    print("• AI that adapts its communication style to user personality")
    print("• Detection of emotional drift and cognitive changes") 
    print("• Personalized interaction patterns that evolve over time")
    print("• Deep semantic understanding beyond keyword matching")
    print("• Foundation for truly relational artificial intelligence")
    print()
    print("🔧 PRODUCTION DEPLOYMENT:")
    print("• Add this to your interaction pipeline after message processing")
    print("• Store results in user context for agent decision-making")
    print("• Set up scheduled jobs to analyze trait evolution trends")
    print("• Integrate with response generation for personality-aware outputs")

if __name__ == "__main__":
    demonstrate_complete_system()

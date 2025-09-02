#!/usr/bin/env python3
"""
🔮 AGI Emotional Drift Detection & Personality Matrix Scorer
Advanced embedding-based personality analysis with drift detection
"""

import numpy as np
import json
import sqlite3
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EmbeddingAnalyzer:
    """Advanced embedding analysis for personality insights and drift detection"""
    
    def __init__(self, memory_db_path: str = "memory/user_memory.db", 
                 memory_server_url: str = "http://localhost:8001"):
        self.memory_db_path = memory_db_path
        self.memory_server_url = memory_server_url
        
        # Personality archetype definitions based on embedding clusters
        self.archetypes = {
            'creative_visionary': {
                'traits': {'openness': 0.9, 'creativity': 0.8, 'intellect': 0.7},
                'keywords': ['imagine', 'create', 'vision', 'innovative', 'artistic']
            },
            'analytical_thinker': {
                'traits': {'conscientiousness': 0.8, 'intellect': 0.9, 'openness': 0.7},
                'keywords': ['analyze', 'logic', 'systematic', 'rational', 'data']
            },
            'empathetic_supporter': {
                'traits': {'agreeableness': 0.9, 'empathy': 0.8, 'extraversion': 0.6},
                'keywords': ['understand', 'support', 'feel', 'help', 'care']
            },
            'adventurous_explorer': {
                'traits': {'extraversion': 0.8, 'openness': 0.8, 'neuroticism': 0.2},
                'keywords': ['adventure', 'explore', 'experience', 'travel', 'discover']
            },
            'peaceful_guardian': {
                'traits': {'agreeableness': 0.8, 'conscientiousness': 0.7, 'neuroticism': 0.3},
                'keywords': ['protect', 'peace', 'stable', 'secure', 'safe']
            }
        }

        # Emotional drift thresholds
        self.drift_thresholds = {
            'minor': 0.15,    # 15% change
            'moderate': 0.3,  # 30% change  
            'major': 0.5      # 50% change
        }

    def get_user_embeddings(self, user_id: str, days_back: int = 30) -> List[Dict]:
        """Retrieve user embeddings from the last N days"""
        try:
            conn = sqlite3.connect(self.memory_db_path)
            cursor = conn.cursor()
            
            # Get embeddings from the last N days
            cutoff_timestamp = (datetime.now() - timedelta(days=days_back)).timestamp()
            
            cursor.execute("""
                SELECT id, post_id, text, embedding_json, model, dimensions, created_at
                FROM embeddings 
                WHERE user_id = ? AND created_at > ?
                ORDER BY created_at DESC
            """, (user_id, cutoff_timestamp))
            
            results = cursor.fetchall()
            conn.close()
            
            embeddings = []
            for row in results:
                embedding_data = json.loads(row[3])  # embedding_json
                embeddings.append({
                    'id': row[0],
                    'post_id': row[1],
                    'text': row[2],
                    'embedding': np.array(embedding_data),
                    'model': row[4],
                    'dimensions': row[5],
                    'timestamp': row[6]
                })
            
            logger.info(f"Retrieved {len(embeddings)} embeddings for user {user_id}")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error retrieving embeddings: {e}")
            return []

    def detect_emotional_drift(self, user_id: str, time_window_days: int = 7) -> Dict[str, Any]:
        """Detect emotional drift by comparing embedding clusters across time periods"""
        embeddings = self.get_user_embeddings(user_id, days_back=30)
        
        if len(embeddings) < 10:
            return {'drift_detected': False, 'reason': 'Insufficient data'}
        
        # Split embeddings into time periods
        now = datetime.now().timestamp()
        recent_cutoff = now - (time_window_days * 24 * 3600)
        older_cutoff = now - (2 * time_window_days * 24 * 3600)
        
        recent_embeddings = [e for e in embeddings if e['timestamp'] > recent_cutoff]
        older_embeddings = [e for e in embeddings if older_cutoff <= e['timestamp'] <= recent_cutoff]
        
        if len(recent_embeddings) < 3 or len(older_embeddings) < 3:
            return {'drift_detected': False, 'reason': 'Insufficient data in time windows'}
        
        # Calculate centroids for each period
        recent_centroid = np.mean([e['embedding'] for e in recent_embeddings], axis=0)
        older_centroid = np.mean([e['embedding'] for e in older_embeddings], axis=0)
        
        # Calculate drift magnitude (cosine distance)
        drift_magnitude = 1 - cosine_similarity(
            np.array([recent_centroid]), 
            np.array([older_centroid])
        )[0][0]
        
        # Determine drift level
        drift_level = 'none'
        if drift_magnitude > self.drift_thresholds['major']:
            drift_level = 'major'
        elif drift_magnitude > self.drift_thresholds['moderate']:
            drift_level = 'moderate'
        elif drift_magnitude > self.drift_thresholds['minor']:
            drift_level = 'minor'
        
        # Analyze semantic drift direction
        drift_analysis = self._analyze_drift_direction(recent_embeddings, older_embeddings)
        
        result = {
            'drift_detected': drift_level != 'none',
            'drift_level': drift_level,
            'drift_magnitude': float(drift_magnitude),
            'recent_period_count': len(recent_embeddings),
            'older_period_count': len(older_embeddings),
            'time_window_days': time_window_days,
            'analysis': drift_analysis,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Drift detection for {user_id}: {drift_level} drift ({drift_magnitude:.3f})")
        return result

    def _analyze_drift_direction(self, recent_embeddings: List[Dict], older_embeddings: List[Dict]) -> Dict[str, Any]:
        """Analyze the semantic direction of personality drift"""
        
        # Cluster embeddings to identify themes
        all_embeddings = recent_embeddings + older_embeddings
        embedding_matrix = np.array([e['embedding'] for e in all_embeddings])
        
        # Use PCA to reduce dimensionality for interpretability (adjust for data size)
        all_embeddings_count = len(all_embeddings)
        embedding_dims = embedding_matrix.shape[1]
        n_components = min(10, all_embeddings_count - 1, embedding_dims)
        
        if n_components < 2:
            # Too few samples for PCA, use original embeddings
            reduced_embeddings = embedding_matrix
        else:
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embedding_matrix)
        
        # Cluster into semantic groups (adjust for small datasets)
        min_clusters = 2
        max_clusters = min(5, len(all_embeddings) // 2)
        n_clusters = max(min_clusters, max_clusters)
        
        if len(all_embeddings) < min_clusters:
            return {'themes': [], 'shift_direction': 'insufficient_data'}
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # Analyze cluster distribution changes
        recent_start = 0
        older_start = len(recent_embeddings)
        
        recent_clusters = cluster_labels[recent_start:recent_start + len(recent_embeddings)]
        older_clusters = cluster_labels[older_start:older_start + len(older_embeddings)]
        
        # Calculate cluster distribution changes
        recent_dist = np.bincount(recent_clusters, minlength=n_clusters) / len(recent_clusters)
        older_dist = np.bincount(older_clusters, minlength=n_clusters) / len(older_clusters)
        
        distribution_shift = recent_dist - older_dist
        
        # Find dominant themes in each cluster
        themes = []
        for cluster_id in range(n_clusters):
            cluster_texts = []
            for i, label in enumerate(cluster_labels):
                if label == cluster_id:
                    cluster_texts.append(all_embeddings[i]['text'])
            
            # Simple keyword extraction (you could use more sophisticated methods)
            theme_words = self._extract_keywords(cluster_texts)
            themes.append({
                'cluster_id': cluster_id,
                'theme_words': theme_words,
                'recent_weight': float(recent_dist[cluster_id]),
                'older_weight': float(older_dist[cluster_id]),
                'shift': float(distribution_shift[cluster_id])
            })
        
        # Identify the direction of shift
        max_shift_cluster = np.argmax(np.abs(distribution_shift))
        shift_direction = 'positive' if distribution_shift[max_shift_cluster] > 0 else 'negative'
        
        return {
            'themes': themes,
            'shift_direction': shift_direction,
            'primary_shift_cluster': int(max_shift_cluster),
            'shift_magnitude': float(np.abs(distribution_shift[max_shift_cluster]))
        }

    def _extract_keywords(self, texts: List[str], top_k: int = 5) -> List[str]:
        """Simple keyword extraction from cluster texts"""
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts).lower()
        words = combined_text.split()
        
        # Filter out common words (simple stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'}
        
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
        
        # Count word frequencies
        word_counts = defaultdict(int)
        for word in filtered_words:
            word_counts[word] += 1
        
        # Return top keywords
        top_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [word for word, count in top_words]

    def update_personality_matrix(self, user_id: str, embedding_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Update personality matrix based on embedding cluster analysis"""
        
        embeddings = self.get_user_embeddings(user_id, days_back=7)  # Focus on recent week
        if len(embeddings) < 5:
            return {'updated': False, 'reason': 'Insufficient recent data'}
        
        # Cluster embeddings to identify personality patterns
        embedding_matrix = np.array([e['embedding'] for e in embeddings])
        
        # Reduce dimensionality for clustering (adjust components based on available data)
        n_components = min(50, embedding_matrix.shape[0] - 1, embedding_matrix.shape[1])
        if n_components < 2:
            # If we can't do PCA, use original embeddings
            reduced_embeddings = embedding_matrix
        else:
            pca = PCA(n_components=n_components)
            reduced_embeddings = pca.fit_transform(embedding_matrix)
        
        # Cluster to find personality themes (adjust for small datasets)
        min_clusters = 2
        max_clusters = min(5, len(embeddings) // 2)
        n_clusters = max(min_clusters, max_clusters)
        
        if len(embeddings) < min_clusters:
            # Too few embeddings for meaningful clustering
            return {'updated': False, 'reason': f'Need at least {min_clusters} embeddings for analysis'}
            
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(reduced_embeddings)
        
        # Analyze each cluster for personality traits
        personality_scores = defaultdict(float)
        trait_evidence = defaultdict(list)
        
        for cluster_id in range(n_clusters):
            cluster_indices = np.where(cluster_labels == cluster_id)[0]
            cluster_embeddings = [embeddings[i] for i in cluster_indices]
            cluster_texts = [embeddings[i]['text'] for i in cluster_indices]
            
            # Calculate cluster centroid
            cluster_centroid = np.mean([embeddings[i]['embedding'] for i in cluster_indices], axis=0)
            
            # Compare with archetype embeddings (would need pre-computed archetype embeddings)
            archetype_scores = self._score_archetype_similarity(cluster_centroid, cluster_texts)
            
            # Weight by cluster size
            cluster_weight = len(cluster_indices) / len(embeddings)
            
            # Update personality scores
            for trait, score in archetype_scores.items():
                personality_scores[trait] += score * cluster_weight
                trait_evidence[trait].extend(cluster_texts[:3])  # Top 3 examples
        
        # Normalize scores
        for trait in personality_scores:
            personality_scores[trait] = min(1.0, max(0.0, personality_scores[trait]))
        
        # Update via Memory Server API
        update_results = []
        for trait, score in personality_scores.items():
            try:
                response = requests.post(f"{self.memory_server_url}/personality/update-trait", 
                                       json={
                                           'user_id': user_id,
                                           'trait': trait,
                                           'score': score
                                       })
                update_results.append({
                    'trait': trait,
                    'score': score,
                    'success': response.status_code == 200,
                    'evidence': trait_evidence[trait][:2]  # Top 2 examples
                })
            except Exception as e:
                logger.error(f"Failed to update trait {trait}: {e}")
                update_results.append({
                    'trait': trait,
                    'score': score,
                    'success': False,
                    'error': str(e)
                })
        
        return {
            'updated': True,
            'user_id': user_id,
            'personality_scores': dict(personality_scores),
            'trait_updates': update_results,
            'embedding_count': len(embeddings),
            'clusters_analyzed': n_clusters,
            'timestamp': datetime.now().isoformat()
        }

    def _score_archetype_similarity(self, cluster_centroid: np.ndarray, cluster_texts: List[str]) -> Dict[str, float]:
        """Score similarity to personality archetypes"""
        scores = {}
        
        # Text-based archetype scoring (simplified)
        combined_text = ' '.join(cluster_texts).lower()
        
        for archetype, data in self.archetypes.items():
            # Count archetype keywords
            keyword_matches = sum(1 for keyword in data['keywords'] if keyword in combined_text)
            keyword_score = keyword_matches / len(data['keywords'])
            
            # Weight by text length (normalize)
            text_length_factor = min(1.0, len(combined_text.split()) / 100)
            
            archetype_score = keyword_score * text_length_factor
            
            # Update trait scores based on archetype
            for trait, trait_score in data['traits'].items():
                if trait not in scores:
                    scores[trait] = 0.0
                scores[trait] += archetype_score * trait_score
        
        return scores

    def find_semantic_glyphs(self, user_id: str, query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Find glyphs using embedding similarity instead of keyword matching"""
        
        # Get query embedding
        try:
            response = requests.post("http://localhost:8003/embed", 
                                   json={"text": query_text, "model": "personality"})
            if response.status_code != 200:
                return []
            
            query_embedding = np.array(response.json()['embedding'])
        except Exception as e:
            logger.error(f"Failed to get query embedding: {e}")
            return []
        
        # Get user's embeddings with text
        embeddings = self.get_user_embeddings(user_id, days_back=60)
        if not embeddings:
            return []
        
        # Calculate similarities
        similarities = []
        for embedding_data in embeddings:
            similarity = cosine_similarity(
                np.array([query_embedding]), 
                np.array([embedding_data['embedding']])
            )[0][0]
            similarities.append({
                'post_id': embedding_data['post_id'],
                'text': embedding_data['text'],
                'similarity': float(similarity),
                'timestamp': embedding_data['timestamp']
            })
        
        # Sort by similarity and return top matches
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        return similarities[:top_k]

    def generate_drift_report(self, user_id: str) -> Dict[str, Any]:
        """Generate comprehensive drift and personality analysis report"""
        
        # Get drift analysis
        drift_analysis = self.detect_emotional_drift(user_id)
        
        # Get personality matrix update
        personality_analysis = self.update_personality_matrix(user_id, drift_analysis)
        
        # Get embedding statistics
        embeddings = self.get_user_embeddings(user_id, days_back=30)
        
        embedding_stats = {
            'total_embeddings': len(embeddings),
            'avg_dimensions': np.mean([e['dimensions'] for e in embeddings]) if embeddings else 0,
            'date_range': {
                'earliest': min([e['timestamp'] for e in embeddings]) if embeddings else 0,
                'latest': max([e['timestamp'] for e in embeddings]) if embeddings else 0
            }
        }
        
        return {
            'user_id': user_id,
            'report_timestamp': datetime.now().isoformat(),
            'drift_analysis': drift_analysis,
            'personality_analysis': personality_analysis,
            'embedding_statistics': embedding_stats,
            'recommendations': self._generate_recommendations(drift_analysis, personality_analysis)
        }

    def _generate_recommendations(self, drift_analysis: Dict, personality_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis"""
        recommendations = []
        
        if drift_analysis.get('drift_detected', False):
            drift_level = drift_analysis.get('drift_level', 'none')
            if drift_level == 'major':
                recommendations.append("🚨 Major personality drift detected. Consider archetype transition or mood recalibration.")
            elif drift_level == 'moderate':
                recommendations.append("⚠️ Moderate drift detected. Monitor for continued changes.")
            else:
                recommendations.append("📊 Minor drift detected. Normal personality evolution.")
        
        if personality_analysis.get('updated', False):
            high_traits = [t['trait'] for t in personality_analysis.get('trait_updates', []) 
                          if t.get('score', 0) > 0.7]
            if high_traits:
                recommendations.append(f"🎯 Strong traits detected: {', '.join(high_traits[:3])}")
        
        if not recommendations:
            recommendations.append("✅ Personality profile is stable and consistent.")
        
        return recommendations


def main():
    """Example usage and testing"""
    analyzer = EmbeddingAnalyzer()
    
    # Test with user from our earlier tests
    user_id = "test_user_123"
    
    print("🔮 Starting Advanced Personality Analysis")
    print("=" * 50)
    
    # Generate comprehensive report
    report = analyzer.generate_drift_report(user_id)
    
    print(f"📊 Analysis Report for {user_id}")
    print(f"📅 Generated: {report['report_timestamp']}")
    print()
    
    # Drift Analysis
    drift = report['drift_analysis']
    print("🌊 DRIFT ANALYSIS")
    print(f"   Drift Detected: {drift.get('drift_detected', False)}")
    if drift.get('drift_detected', False):
        print(f"   Level: {drift.get('drift_level', 'none').upper()}")
        print(f"   Magnitude: {drift.get('drift_magnitude', 0):.3f}")
    print()
    
    # Personality Analysis  
    personality = report['personality_analysis']
    print("🧠 PERSONALITY MATRIX")
    if personality.get('updated', False):
        print(f"   Traits Updated: {len(personality.get('trait_updates', []))}")
        for trait_update in personality.get('trait_updates', [])[:5]:
            status = "✅" if trait_update.get('success', False) else "❌"
            print(f"   {status} {trait_update.get('trait', '')}: {trait_update.get('score', 0):.3f}")
    print()
    
    # Recommendations
    recommendations = report.get('recommendations', [])
    print("💡 RECOMMENDATIONS")
    for rec in recommendations:
        print(f"   {rec}")
    print()
    
    # Test semantic glyph matching
    print("🔮 SEMANTIC GLYPH MATCHING")
    test_queries = [
        "I feel peaceful and content",
        "I'm excited about new adventures", 
        "I need to think this through logically"
    ]
    
    for query in test_queries:
        matches = analyzer.find_semantic_glyphs(user_id, query, top_k=2)
        print(f"   Query: '{query}'")
        for match in matches:
            print(f"     → {match['similarity']:.3f}: {match['text'][:60]}...")
        print()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script for the complete AGI embedding pipeline.
Tests: Embedding Service → Go Memory Server → SQLite Storage
"""

import requests
import json
import time
import sqlite3
import os
from typing import List, Dict

class EmbeddingPipelineTest:
    def __init__(self):
        self.embedding_service_url = "http://localhost:8003"
        self.memory_server_url = "http://localhost:8001"
        self.test_data = [
            {
                "content": "I love spending time in nature, especially hiking in the mountains. There's something peaceful about the fresh air and beautiful views."
            },
            {
                "content": "Technology fascinates me. I'm always excited to learn about new programming languages and frameworks that can solve complex problems."
            },
            {
                "content": "Cooking is one of my favorite hobbies. I enjoy experimenting with different cuisines and creating new recipes for my family."
            }
        ]
        
    def test_embedding_service(self) -> bool:
        """Test the Python embedding service directly."""
        print("🧪 Testing Embedding Service...")
        
        try:
            # Test health check
            response = requests.get(f"{self.embedding_service_url}/health")
            if response.status_code != 200:
                print(f"❌ Health check failed: {response.status_code}")
                return False
            print("✅ Embedding service health check passed")
            
            # Test single embedding
            test_request = {
                "text": "This is a test sentence for embedding generation.",
                "model": "personality"
            }
            response = requests.post(f"{self.embedding_service_url}/embed", json=test_request)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Single embedding: {result['dimensions']} dimensions, model: {result['model']}")
                return True
            else:
                print(f"❌ Single embedding failed: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Embedding service test failed: {e}")
            return False
    
    def test_memory_server_ingestion(self) -> bool:
        """Test the Go memory server ingestion with embedding generation."""
        print("🧪 Testing Memory Server Ingestion...")
        
        try:
            ingest_request = {
                "user_id": "test_user_123",
                "posts": self.test_data
            }
            
            response = requests.post(f"{self.memory_server_url}/ingest-posts", json=ingest_request)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Ingestion successful: {result['embeddings_count']}/{result['total_posts']} embeddings stored")
                return True
            else:
                print(f"❌ Ingestion failed: {response.status_code}, {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ Memory server test failed: {e}")
            return False
    
    def test_database_storage(self) -> bool:
        """Verify embeddings were stored in SQLite database."""
        print("🧪 Testing Database Storage...")
        
        try:
            # Connect to SQLite database - using the correct filename
            db_path = os.path.join(os.path.dirname(__file__), "memory", "user_memory.db")
            if not os.path.exists(db_path):
                print(f"❌ Database file not found at: {db_path}")
                return False
                
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Query embeddings table
            cursor.execute("""
                SELECT COUNT(*) as total,
                       AVG(LENGTH(embedding_json)) as avg_size,
                       MAX(dimensions) as max_dims
                FROM embeddings 
                WHERE user_id = 'test_user_123'
            """)
            
            result = cursor.fetchone()
            if result and result[0] > 0:
                total, avg_size, max_dims = result
                print(f"✅ Database storage verified: {total} embeddings, avg size: {avg_size:.0f} bytes, dims: {max_dims}")
                
                # Show sample data
                cursor.execute("""
                    SELECT id, user_id, post_id, model, dimensions, 
                           LENGTH(text) as text_len,
                           LENGTH(embedding_json) as embed_size,
                           created_at
                    FROM embeddings 
                    WHERE user_id = 'test_user_123'
                    LIMIT 3
                """)
                
                samples = cursor.fetchall()
                print("📊 Sample stored embeddings:")
                for row in samples:
                    print(f"   ID: {row[0]}, Post: {row[2]}, Model: {row[3]}, Dims: {row[4]}, Text: {row[5]} chars, Embed: {row[6]} bytes")
                
                conn.close()
                return True
            else:
                print("❌ No embeddings found in database")
                conn.close()
                return False
                
        except Exception as e:
            print(f"❌ Database test failed: {e}")
            return False
    
    def run_full_test(self):
        """Run complete end-to-end test of the embedding pipeline."""
        print("🚀 Starting AGI Embedding Pipeline Test\n")
        
        # Wait for services to be ready
        print("⏳ Waiting for services to start...")
        time.sleep(2)
        
        tests = [
            ("Embedding Service", self.test_embedding_service),
            ("Memory Server Ingestion", self.test_memory_server_ingestion),
            ("Database Storage", self.test_database_storage)
        ]
        
        results = []
        for test_name, test_func in tests:
            result = test_func()
            results.append((test_name, result))
            print()  # Add spacing between tests
            
        # Summary
        print("=" * 50)
        print("🏁 TEST SUMMARY:")
        print("=" * 50)
        
        all_passed = True
        for test_name, passed in results:
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"{status} - {test_name}")
            if not passed:
                all_passed = False
        
        print("=" * 50)
        if all_passed:
            print("🎉 ALL TESTS PASSED! Embedding pipeline is working correctly.")
        else:
            print("⚠️  Some tests failed. Check the output above for details.")
        print("=" * 50)
        
        return all_passed

if __name__ == "__main__":
    tester = EmbeddingPipelineTest()
    success = tester.run_full_test()
    exit(0 if success else 1)

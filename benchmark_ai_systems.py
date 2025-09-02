#!/usr/bin/env python3
"""
AI System Benchmarking Script
Compares performance between our AGI system and reference models
"""

import requests
import time
import json
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.base_agent import BaseAgent

class AISystemBenchmark:
    def __init__(self):
        self.agi_agent = BaseAgent()
        self.test_prompts = [
            "Hello, how are you today?",
            "What is the meaning of life?",
            "Tell me about artificial intelligence",
            "How does machine learning work?",
            "What are your thoughts on creativity?",
            "Can you help me solve a problem?",
            "What's the weather like today?",
            "Tell me a joke",
            "How do you learn and improve?",
            "What makes you unique?"
        ]

    def benchmark_agi_system(self, iterations=3):
        """Benchmark our AGI system"""
        print("🔬 Benchmarking AGI System...")
        return self.agi_agent.benchmark_backends(self.test_prompts, iterations=iterations)

    def benchmark_reference_api(self, api_url="http://localhost:8003", iterations=3):
        """Benchmark reference OpenAI-style API"""
        print("🔬 Benchmarking Reference API...")
        results = {
            "timestamp": time.time(),
            "backend": "reference_api",
            "metrics": {},
            "results": []
        }

        for prompt in self.test_prompts:
            prompt_latencies = []

            for i in range(iterations):
                start_time = time.time()

                try:
                    payload = {
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": prompt}],
                        "max_tokens": 150,
                        "temperature": 0.7
                    }

                    resp = requests.post(f"{api_url}/chat/completions",
                                       json=payload, timeout=30)

                    if resp.ok:
                        data = resp.json()
                        response = data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        end_time = time.time()
                        latency = (end_time - start_time) * 1000

                        results["results"].append({
                            "prompt": prompt,
                            "iteration": i + 1,
                            "latency_ms": latency,
                            "response_length": len(response),
                            "success": True
                        })
                        prompt_latencies.append(latency)
                    else:
                        results["results"].append({
                            "prompt": prompt,
                            "iteration": i + 1,
                            "latency_ms": -1,
                            "response_length": 0,
                            "success": False,
                            "error": f"HTTP {resp.status_code}"
                        })

                except Exception as e:
                    results["results"].append({
                        "prompt": prompt,
                        "iteration": i + 1,
                        "latency_ms": -1,
                        "response_length": 0,
                        "success": False,
                        "error": str(e)[:100]
                    })

        # Calculate metrics
        successful_results = [r for r in results["results"] if r["success"]]
        latencies = [r["latency_ms"] for r in successful_results if r["latency_ms"] > 0]

        results["metrics"] = {
            "total_requests": len(results["results"]),
            "successful_requests": len(successful_results),
            "success_rate": len(successful_results) / len(results["results"]) if results["results"] else 0,
            "avg_latency_ms": sum(latencies) / len(latencies) if latencies else -1,
            "min_latency_ms": min(latencies) if latencies else -1,
            "max_latency_ms": max(latencies) if latencies else -1,
            "avg_response_length": sum(r["response_length"] for r in successful_results) / len(successful_results) if successful_results else 0
        }

        return results

    def compare_systems(self, agi_results, reference_results):
        """Compare performance between AGI and reference systems"""
        print("\n📊 COMPARISON RESULTS")
        print("=" * 50)

        agi_metrics = agi_results.get("metrics", {})
        ref_metrics = reference_results.get("metrics", {})

        # Compare each backend in AGI against reference
        for backend_name, backend_metrics in agi_metrics.items():
            print(f"\n🔍 {backend_name.upper()} vs REFERENCE API")
            print("-" * 30)

            comparisons = [
                ("Success Rate", backend_metrics.get("success_rate", 0), ref_metrics.get("success_rate", 0)),
                ("Avg Latency (ms)", backend_metrics.get("avg_latency_ms", -1), ref_metrics.get("avg_latency_ms", -1)),
                ("Avg Response Length", backend_metrics.get("avg_response_length", 0), ref_metrics.get("avg_response_length", 0))
            ]

            for metric_name, agi_value, ref_value in comparisons:
                if agi_value == -1 or ref_value == -1:
                    winner = "N/A"
                elif metric_name == "Avg Latency (ms)":
                    winner = f"{'AGI' if agi_value < ref_value else 'Reference'} (faster)"
                elif metric_name == "Success Rate":
                    winner = f"{'AGI' if agi_value > ref_value else 'Reference'} (more reliable)"
                else:
                    winner = f"{'AGI' if agi_value > ref_value else 'Reference'} (longer responses)"

                print(f"{metric_name}: AGI={agi_value:.2f}, Ref={ref_value:.2f} → {winner}")

    def run_full_benchmark(self, reference_api_url=None, iterations=3):
        """Run complete benchmark suite"""
        print("🚀 Starting AI System Benchmark Suite")
        print(f"Test prompts: {len(self.test_prompts)}")
        print(f"Iterations per prompt: {iterations}")
        print()

        # Benchmark AGI system
        agi_results = self.benchmark_agi_system(iterations)

        # Benchmark reference API if URL provided
        reference_results = None
        if reference_api_url:
            reference_results = self.benchmark_reference_api(reference_api_url, iterations)
        else:
            print("⚠️ No reference API URL provided - skipping reference benchmark")

        # Save results
        timestamp = int(time.time())
        results_file = f"benchmark_results_{timestamp}.json"

        results = {
            "agi_results": agi_results,
            "reference_results": reference_results,
            "test_config": {
                "prompts_count": len(self.test_prompts),
                "iterations": iterations,
                "reference_api_url": reference_api_url
            }
        }

        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\n💾 Results saved to: {results_file}")

        # Compare systems if both results available
        if reference_results:
            self.compare_systems(agi_results, reference_results)

        return results

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI System Benchmarking")
    parser.add_argument("--reference-api", help="Reference API URL to benchmark against")
    parser.add_argument("--iterations", type=int, default=3, help="Iterations per prompt")
    parser.add_argument("--prompts", nargs="+", help="Custom test prompts")

    args = parser.parse_args()

    benchmark = AISystemBenchmark()

    if args.prompts:
        benchmark.test_prompts = args.prompts

    benchmark.run_full_benchmark(
        reference_api_url=args.reference_api,
        iterations=args.iterations
    )

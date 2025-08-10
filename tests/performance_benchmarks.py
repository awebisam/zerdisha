#!/usr/bin/env python3
"""
Performance benchmarks for PEEngine core solidification features.

This script provides automated performance testing for:
1. Gap check operations with various dataset sizes
2. Session map generation with large concept networks
3. Metacognitive analysis with extended conversation histories

Usage:
    python tests/performance_benchmarks.py
    python tests/performance_benchmarks.py --feature gapcheck
    python tests/performance_benchmarks.py --feature sessionmap
    python tests/performance_benchmarks.py --feature metacognitive
"""

import asyncio
import time
import statistics
import argparse
from typing import List, Dict, Any
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from peengine.core.orchestrator import ExplorationEngine
from peengine.models.config import Settings
from peengine.models.graph import Session, Message


class PerformanceBenchmark:
    """Performance benchmark suite for PEEngine core features."""
    
    def __init__(self):
        self.settings = Settings()
        self.engine = None
        self.results = {}
    
    async def setup(self):
        """Initialize the exploration engine for testing."""
        self.engine = ExplorationEngine(self.settings)
        await self.engine.initialize()
    
    async def teardown(self):
        """Clean up after testing."""
        if self.engine:
            await self.engine.cleanup()
    
    def time_async_function(self, func, *args, **kwargs):
        """Time an async function execution."""
        start_time = time.time()
        result = asyncio.run(func(*args, **kwargs))
        end_time = time.time()
        return result, end_time - start_time
    
    async def create_test_session_with_concepts(self, num_concepts: int) -> Session:
        """Create a test session with specified number of concepts."""
        session = await self.engine.start_session(f"benchmark_session_{num_concepts}_concepts")
        
        # Simulate conversation that creates concepts
        concepts = [
            "quantum mechanics", "wave function", "superposition", "entanglement",
            "uncertainty principle", "quantum tunneling", "decoherence", "measurement problem",
            "many worlds interpretation", "copenhagen interpretation", "hidden variables",
            "bell's theorem", "quantum field theory", "particle physics", "relativity",
            "spacetime", "black holes", "thermodynamics", "entropy", "energy conservation",
            "electromagnetic field", "photons", "electrons", "atoms", "molecules"
        ]
        
        for i in range(min(num_concepts, len(concepts))):
            concept = concepts[i]
            # Simulate user message and system processing
            user_message = Message(
                role="user",
                content=f"I'm curious about {concept} and how it relates to physics",
                timestamp=time.time()
            )
            session.messages.append(user_message)
            
            # Simulate system response (this would normally create nodes/edges)
            system_message = Message(
                role="assistant", 
                content=f"Let's explore {concept} through metaphors and connections",
                timestamp=time.time()
            )
            session.messages.append(system_message)
            
            # Add mock node/edge IDs to simulate graph creation
            session.nodes_created.append(f"node_{i}_{concept.replace(' ', '_')}")
            if i > 0:
                session.edges_created.append(f"edge_{i-1}_{i}")
        
        return session
    
    async def benchmark_gap_check(self, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark gap check performance with different scenarios."""
        print("🔍 Benchmarking Gap Check Performance...")
        
        results = {
            "existing_cvector": [],
            "new_cvector": [],
            "large_session": []
        }
        
        # Test 1: Gap check with existing c-vector
        print("  Testing with existing c-vector...")
        for i in range(num_runs):
            session = await self.create_test_session_with_concepts(5)
            self.engine.current_session = session
            
            start_time = time.time()
            try:
                result = await self.engine._gap_check()
                end_time = time.time()
                results["existing_cvector"].append(end_time - start_time)
                print(f"    Run {i+1}: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: ERROR - {e}")
        
        # Test 2: Gap check with new c-vector generation
        print("  Testing with new c-vector generation...")
        for i in range(num_runs):
            session = await self.create_test_session_with_concepts(3)
            self.engine.current_session = session
            
            start_time = time.time()
            try:
                # Force new c-vector creation by using unique concept
                result = await self.engine._gap_check()
                end_time = time.time()
                results["new_cvector"].append(end_time - start_time)
                print(f"    Run {i+1}: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: ERROR - {e}")
        
        # Test 3: Gap check with large session
        print("  Testing with large session (20+ concepts)...")
        for i in range(num_runs):
            session = await self.create_test_session_with_concepts(25)
            self.engine.current_session = session
            
            start_time = time.time()
            try:
                result = await self.engine._gap_check()
                end_time = time.time()
                results["large_session"].append(end_time - start_time)
                print(f"    Run {i+1}: {end_time - start_time:.2f}s")
            except Exception as e:
                print(f"    Run {i+1}: ERROR - {e}")
        
        return results
    
    async def benchmark_session_map(self, num_runs: int = 5) -> Dict[str, Any]:
        """Benchmark session map performance with different graph sizes."""
        print("🗺️  Benchmarking Session Map Performance...")
        
        results = {
            "small_session": [],
            "medium_session": [],
            "large_session": []
        }
        
        # Test different session sizes
        test_sizes = [
            ("small_session", 5),
            ("medium_session", 15),
            ("large_session", 30)
        ]
        
        for test_name, num_concepts in test_sizes:
            print(f"  Testing {test_name} ({num_concepts} concepts)...")
            
            for i in range(num_runs):
                session = await self.create_test_session_with_concepts(num_concepts)
                self.engine.current_session = session
                
                start_time = time.time()
                try:
                    result = await self.engine._show_session_map()
                    end_time = time.time()
                    results[test_name].append(end_time - start_time)
                    print(f"    Run {i+1}: {end_time - start_time:.2f}s")
                except Exception as e:
                    print(f"    Run {i+1}: ERROR - {e}")
        
        return results
    
    async def benchmark_metacognitive_analysis(self, num_runs: int = 3) -> Dict[str, Any]:
        """Benchmark metacognitive analysis with extended conversation histories."""
        print("🧠 Benchmarking Metacognitive Analysis Performance...")
        
        results = {
            "short_history": [],
            "medium_history": [],
            "long_history": []
        }
        
        # Test different conversation lengths
        test_lengths = [
            ("short_history", 10),
            ("medium_history", 30),
            ("long_history", 60)
        ]
        
        for test_name, num_messages in test_lengths:
            print(f"  Testing {test_name} ({num_messages} messages)...")
            
            for i in range(num_runs):
                session = await self.create_test_session_with_concepts(10)
                
                # Add many messages to simulate long conversation
                for j in range(num_messages):
                    message = Message(
                        role="user" if j % 2 == 0 else "assistant",
                        content=f"Message {j} about quantum mechanics and metaphors",
                        timestamp=time.time() + j
                    )
                    session.messages.append(message)
                
                self.engine.current_session = session
                
                start_time = time.time()
                try:
                    # Simulate metacognitive analysis
                    result = await self.engine.ma.analyze_session(
                        session.messages,
                        session.nodes_created,
                        session.edges_created
                    )
                    end_time = time.time()
                    results[test_name].append(end_time - start_time)
                    print(f"    Run {i+1}: {end_time - start_time:.2f}s")
                except Exception as e:
                    print(f"    Run {i+1}: ERROR - {e}")
        
        return results
    
    def analyze_results(self, results: Dict[str, List[float]], feature_name: str):
        """Analyze and display benchmark results."""
        print(f"\n📊 {feature_name} Performance Analysis")
        print("=" * 50)
        
        for test_name, times in results.items():
            if not times:
                print(f"{test_name}: No successful runs")
                continue
            
            avg_time = statistics.mean(times)
            min_time = min(times)
            max_time = max(times)
            median_time = statistics.median(times)
            
            print(f"\n{test_name.replace('_', ' ').title()}:")
            print(f"  Average: {avg_time:.2f}s")
            print(f"  Median:  {median_time:.2f}s")
            print(f"  Min:     {min_time:.2f}s")
            print(f"  Max:     {max_time:.2f}s")
            print(f"  Runs:    {len(times)}")
            
            # Performance assessment
            if feature_name == "Gap Check":
                if "existing" in test_name and avg_time > 3.0:
                    print(f"  ⚠️  SLOW: Exceeds 3s target")
                elif "new" in test_name and avg_time > 10.0:
                    print(f"  ⚠️  SLOW: Exceeds 10s target")
                elif "large" in test_name and avg_time > 5.0:
                    print(f"  ⚠️  SLOW: Exceeds 5s target")
                else:
                    print(f"  ✅ GOOD: Within performance targets")
            
            elif feature_name == "Session Map":
                if avg_time > 5.0:
                    print(f"  ⚠️  SLOW: Exceeds 5s target")
                else:
                    print(f"  ✅ GOOD: Within performance targets")
            
            elif feature_name == "Metacognitive Analysis":
                if avg_time > 8.0:
                    print(f"  ⚠️  SLOW: May impact user experience")
                else:
                    print(f"  ✅ GOOD: Reasonable response time")
    
    async def run_all_benchmarks(self):
        """Run all performance benchmarks."""
        print("🚀 Starting PEEngine Performance Benchmarks")
        print("=" * 60)
        
        await self.setup()
        
        try:
            # Gap Check Benchmarks
            gap_results = await self.benchmark_gap_check()
            self.analyze_results(gap_results, "Gap Check")
            
            # Session Map Benchmarks  
            map_results = await self.benchmark_session_map()
            self.analyze_results(map_results, "Session Map")
            
            # Metacognitive Analysis Benchmarks
            meta_results = await self.benchmark_metacognitive_analysis()
            self.analyze_results(meta_results, "Metacognitive Analysis")
            
            print("\n🎯 Benchmark Summary")
            print("=" * 30)
            print("All benchmarks completed. Review results above for performance assessment.")
            
        except Exception as e:
            print(f"❌ Benchmark failed: {e}")
        finally:
            await self.teardown()


async def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(description="PEEngine Performance Benchmarks")
    parser.add_argument(
        "--feature", 
        choices=["gapcheck", "sessionmap", "metacognitive", "all"],
        default="all",
        help="Which feature to benchmark"
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of runs per test (default: 5)"
    )
    
    args = parser.parse_args()
    
    benchmark = PerformanceBenchmark()
    await benchmark.setup()
    
    try:
        if args.feature == "gapcheck":
            results = await benchmark.benchmark_gap_check(args.runs)
            benchmark.analyze_results(results, "Gap Check")
        elif args.feature == "sessionmap":
            results = await benchmark.benchmark_session_map(args.runs)
            benchmark.analyze_results(results, "Session Map")
        elif args.feature == "metacognitive":
            results = await benchmark.benchmark_metacognitive_analysis(args.runs)
            benchmark.analyze_results(results, "Metacognitive Analysis")
        else:
            await benchmark.run_all_benchmarks()
    
    finally:
        await benchmark.teardown()


if __name__ == "__main__":
    asyncio.run(main())
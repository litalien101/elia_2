from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from elia.cognition.readiness import (
    run_readiness_benchmark,
    run_reasoning_latency_benchmark,
)


class ReadinessBenchmarkTests(unittest.TestCase):
    def test_reasoning_latency_benchmark_checks_expected_closure(self):
        report = run_reasoning_latency_benchmark(graph_nodes=8, repeats=3)
        self.assertEqual(report["benchmark"], "elia_reasoning_latency_synthetic_chain_v1")
        self.assertEqual(report["expected_answers"], 7)
        self.assertEqual(report["source_facts"], 7)
        self.assertGreaterEqual(report["cold_query_ms"], 0)
        self.assertGreaterEqual(report["warm_query_median_ms"], 0)
        self.assertGreater(report["warm_queries_per_second"], 0)

    def test_reasoning_latency_benchmark_rejects_unbounded_inputs(self):
        with self.assertRaises(ValueError):
            run_reasoning_latency_benchmark(graph_nodes=2)
        with self.assertRaises(ValueError):
            run_reasoning_latency_benchmark(repeats=0)

    def test_readiness_report_has_machine_and_local_tool_fields(self):
        report = run_readiness_benchmark(cognition_samples=8, reasoning_repeats=2)
        self.assertEqual(report["benchmark"], "elia_local_hardware_readiness_v1")
        self.assertIn("effective_cpu_cores_estimate", report["host"])
        self.assertIn("memory_available_bytes", report["host"])
        self.assertIn("python_modules", report["speech_tooling"])
        self.assertIn("cognition_throughput", report)
        self.assertIn("reasoning_latency", report)
        self.assertIn("no speech model is downloaded", report["warning"])


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import os
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from .benchmark import run_hardware_profile_benchmark
from .reasoning import BoundedRuleEngine, Fact, parse_literal, parse_rule


def _read_integer(path: Path) -> int | None:
    try:
        value = path.read_text(encoding="ascii").strip()
    except OSError:
        return None
    if value in {"", "max"}:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _cgroup_limits() -> dict[str, int | float | None]:
    cpu_quota: float | None = None
    cpu_max = Path("/sys/fs/cgroup/cpu.max")
    try:
        parts = cpu_max.read_text(encoding="ascii").split()
    except OSError:
        parts = []
    if len(parts) == 2 and parts[0] != "max":
        try:
            quota, period = int(parts[0]), int(parts[1])
        except ValueError:
            quota = period = 0
        if quota > 0 and period > 0:
            cpu_quota = quota / period
    if cpu_quota is None:
        quota = _read_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_quota_us"))
        period = _read_integer(Path("/sys/fs/cgroup/cpu/cpu.cfs_period_us"))
        if quota is not None and period and quota > 0 and period > 0:
            cpu_quota = quota / period

    memory_limit = _read_integer(Path("/sys/fs/cgroup/memory.max"))
    memory_current = _read_integer(Path("/sys/fs/cgroup/memory.current"))
    if memory_limit is None:
        memory_limit = _read_integer(Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"))
        memory_current = _read_integer(Path("/sys/fs/cgroup/memory/memory.usage_in_bytes"))
        if memory_limit is not None and memory_limit >= 1 << 60:
            memory_limit = None
    return {
        "cpu_quota_cores": round(cpu_quota, 3) if cpu_quota is not None else None,
        "memory_limit_bytes": memory_limit,
        "memory_current_bytes": memory_current,
    }


def _host_memory() -> dict[str, int | None]:
    if sys.platform.startswith("linux"):
        values: dict[str, int] = {}
        try:
            for line in Path("/proc/meminfo").read_text(encoding="ascii").splitlines():
                key, _, rest = line.partition(":")
                parts = rest.split()
                if parts and key in {"MemTotal", "MemAvailable"}:
                    values[key] = int(parts[0]) * 1024
        except OSError:
            pass
        return {
            "total_bytes": values.get("MemTotal"),
            "available_bytes": values.get("MemAvailable"),
        }

    if sys.platform == "win32":
        class MemoryStatusEx(ctypes.Structure):
            _fields_ = [
                ("dwLength", ctypes.c_ulong),
                ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
            ]

        status = MemoryStatusEx()
        status.dwLength = ctypes.sizeof(MemoryStatusEx)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {
                "total_bytes": int(status.ullTotalPhys),
                "available_bytes": int(status.ullAvailPhys),
            }
    return {"total_bytes": None, "available_bytes": None}


def _speech_tooling() -> dict[str, Any]:
    executables = (
        "ffmpeg",
        "whisper-cli",
        "piper",
        "espeak-ng",
        "ollama",
        "llama-server",
    )
    modules = (
        "faster_whisper",
        "vosk",
        "onnxruntime",
        "piper",
    )
    return {
        "executables": {name: shutil.which(name) for name in executables},
        "python_modules": {
            name: importlib.util.find_spec(name) is not None
            for name in modules
        },
        "note": (
            "Availability checks only detect installed runtimes; they do not verify model files, "
            "licenses, audio devices, or recognition quality."
        ),
    }


def run_reasoning_latency_benchmark(
    graph_nodes: int = 24,
    repeats: int = 25,
) -> dict[str, Any]:
    if not 3 <= graph_nodes <= 64:
        raise ValueError("graph_nodes must be between 3 and 64")
    if not 1 <= repeats <= 1000:
        raise ValueError("repeats must be between 1 and 1000")

    facts = [
        Fact(
            f"edge-{source}",
            parse_literal(f"edge(n{source}, n{source + 1})"),
            f"observation-{source}",
        )
        for source in range(graph_nodes - 1)
    ]
    rules = [
        parse_rule("edge(?x, ?y) -> reachable(?x, ?y)", "direct"),
        parse_rule(
            "edge(?x, ?y) & reachable(?y, ?z) -> reachable(?x, ?z)",
            "transitive",
        ),
    ]
    engine = BoundedRuleEngine(facts, rules, max_work=20_000)
    query = parse_literal("reachable(n0, ?target)")
    started = time.perf_counter()
    cold = engine.query(query)
    cold_ms = (time.perf_counter() - started) * 1000
    expected_answers = graph_nodes - 1
    if len(cold["answers"]) != expected_answers or cold["state"] != "supported":
        raise RuntimeError(
            "Synthetic reasoning benchmark produced an unexpected answer set"
        )

    latencies_ms: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        result = engine.query(query)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        if len(result["answers"]) != expected_answers or result["state"] != "supported":
            raise RuntimeError(
                "Repeated reasoning query produced an inconsistent answer set"
            )
    ordered = sorted(latencies_ms)
    p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999) - 1))
    return {
        "benchmark": "elia_reasoning_latency_synthetic_chain_v1",
        "graph_nodes": graph_nodes,
        "source_facts": len(facts),
        "rules": len(rules),
        "expected_answers": expected_answers,
        "cold_query_ms": round(cold_ms, 3),
        "warm_query_median_ms": round(statistics.median(ordered), 3),
        "warm_query_p95_ms": round(ordered[p95_index], 3),
        "warm_queries_per_second": round(1000 / max(statistics.mean(ordered), 1e-9), 2),
        "repeats": repeats,
        "warning": (
            "Measures this synthetic transitive-closure workload only; it is not a general reasoning quality or latency guarantee."
        ),
    }


def run_readiness_benchmark(
    cognition_samples: int = 128,
    reasoning_repeats: int = 25,
) -> dict[str, Any]:
    memory = _host_memory()
    limits = _cgroup_limits()
    effective_cores = limits["cpu_quota_cores"] or os.cpu_count() or 1
    total_memory = memory["total_bytes"]
    cgroup_memory_limit = limits["memory_limit_bytes"]
    if cgroup_memory_limit is not None:
        total_memory = (
            min(total_memory, cgroup_memory_limit)
            if total_memory is not None
            else cgroup_memory_limit
        )
    available_memory = memory["available_bytes"]
    if cgroup_memory_limit is not None and limits["memory_current_bytes"] is not None:
        cgroup_available = max(
            0,
            cgroup_memory_limit - limits["memory_current_bytes"],
        )
        available_memory = (
            min(available_memory, cgroup_available)
            if available_memory is not None
            else cgroup_available
        )

    return {
        "benchmark": "elia_local_hardware_readiness_v1",
        "host": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "logical_processors": os.cpu_count(),
            "effective_cpu_cores_estimate": round(float(effective_cores), 3),
            "cgroup": limits,
            "memory_total_bytes": total_memory,
            "memory_available_bytes": available_memory,
            "memory_total_gib": round(total_memory / (1024 ** 3), 3) if total_memory else None,
            "memory_available_gib": (
                round(available_memory / (1024 ** 3), 3)
                if available_memory is not None
                else None
            ),
        },
        "speech_tooling": _speech_tooling(),
        "cognition_throughput": run_hardware_profile_benchmark(cognition_samples),
        "reasoning_latency": run_reasoning_latency_benchmark(repeats=reasoning_repeats),
        "warning": (
            "Run this command on the target Lightsail VPS to measure that machine. "
            "The cognition and reasoning benchmarks are synthetic workloads; no speech model is downloaded or invoked."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure ELIA's local hardware readiness without downloading models."
    )
    parser.add_argument("--cognition-samples", type=int, default=128)
    parser.add_argument("--reasoning-repeats", type=int, default=25)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_readiness_benchmark(args.cognition_samples, args.reasoning_repeats)
    serialized = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
        print(f"Saved readiness report to {args.output.resolve()}")
    print(serialized)


if __name__ == "__main__":
    main()

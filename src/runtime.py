from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RuntimeLimits:
    cpu_threads: int | None = None
    memory_bytes: int | None = None
    memory_limit_applied: bool = False


def format_bytes(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unlimited"
    value = float(num_bytes)
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _parse_positive_int(value: int | str | None, label: str) -> int | None:
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{label} must be a positive integer.")
    return parsed


def _parse_fraction(value: float | str | None, label: str) -> float | None:
    if value is None or value == "":
        return None
    parsed = float(value)
    if not 0 < parsed <= 1:
        raise ValueError(f"{label} must be in the range (0, 1].")
    return parsed


def _total_memory_bytes() -> int | None:
    if hasattr(os, "sysconf"):
        try:
            page_size = int(os.sysconf("SC_PAGE_SIZE"))
            page_count = int(os.sysconf("SC_PHYS_PAGES"))
            if page_size > 0 and page_count > 0:
                return page_size * page_count
        except (OSError, ValueError):
            pass

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemTotal:"):
                parts = line.split()
                if len(parts) >= 2:
                    return int(parts[1]) * 1024
    return None


def resolve_cpu_threads(
    max_cpu_threads: int | None = None,
    cpu_fraction: float | None = None,
) -> int | None:
    max_cpu_threads = _parse_positive_int(
        max_cpu_threads if max_cpu_threads is not None else os.getenv("BUILDING_BDG2_MAX_CPU_THREADS"),
        "max_cpu_threads",
    )
    cpu_fraction = _parse_fraction(
        cpu_fraction if cpu_fraction is not None else os.getenv("BUILDING_BDG2_CPU_FRACTION"),
        "cpu_fraction",
    )
    if max_cpu_threads is None and cpu_fraction is None:
        return None

    total_cpus = os.cpu_count() or 1
    if max_cpu_threads is None:
        max_cpu_threads = max(1, math.floor(total_cpus * cpu_fraction))
    return min(total_cpus, max_cpu_threads)


def resolve_memory_limit_bytes(memory_fraction: float | None = None) -> int | None:
    memory_fraction = _parse_fraction(
        memory_fraction if memory_fraction is not None else os.getenv("BUILDING_BDG2_MEMORY_FRACTION"),
        "memory_fraction",
    )
    if memory_fraction is None:
        return None
    total_memory = _total_memory_bytes()
    if total_memory is None:
        raise RuntimeError("Unable to determine total system memory for memory_fraction.")
    return max(1, math.floor(total_memory * memory_fraction))


def _apply_thread_limits(cpu_threads: int) -> None:
    thread_vars = (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    )
    for var_name in thread_vars:
        os.environ[var_name] = str(cpu_threads)

    try:
        import pyarrow as pa

        pa.set_cpu_count(cpu_threads)
        pa.set_io_thread_count(max(1, min(cpu_threads, 8)))
    except Exception:
        LOGGER.debug("PyArrow thread limit configuration was skipped.", exc_info=True)

    try:
        import torch

        torch.set_num_threads(cpu_threads)
        try:
            torch.set_num_interop_threads(max(1, min(cpu_threads, 4)))
        except RuntimeError:
            LOGGER.debug("PyTorch interop thread count was already initialized.", exc_info=True)
    except Exception:
        LOGGER.debug("PyTorch thread limit configuration was skipped.", exc_info=True)


def _apply_memory_limit(memory_bytes: int) -> bool:
    try:
        import resource
    except ImportError:
        LOGGER.warning("Memory limit requested but `resource` is unavailable on this platform.")
        return False

    current_soft, current_hard = resource.getrlimit(resource.RLIMIT_AS)
    desired_soft = memory_bytes
    if current_hard not in (-1, resource.RLIM_INFINITY):
        desired_soft = min(desired_soft, current_hard)
    if current_soft not in (-1, resource.RLIM_INFINITY):
        desired_soft = min(desired_soft, current_soft)

    resource.setrlimit(resource.RLIMIT_AS, (desired_soft, current_hard))
    return True


def apply_runtime_limits(
    max_cpu_threads: int | None = None,
    cpu_fraction: float | None = None,
    memory_fraction: float | None = None,
) -> RuntimeLimits:
    cpu_threads = resolve_cpu_threads(max_cpu_threads=max_cpu_threads, cpu_fraction=cpu_fraction)
    if cpu_threads is not None:
        _apply_thread_limits(cpu_threads)

    memory_bytes = resolve_memory_limit_bytes(memory_fraction=memory_fraction)
    memory_limit_applied = False
    if memory_bytes is not None:
        memory_limit_applied = _apply_memory_limit(memory_bytes)

    return RuntimeLimits(
        cpu_threads=cpu_threads,
        memory_bytes=memory_bytes,
        memory_limit_applied=memory_limit_applied,
    )

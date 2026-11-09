from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from typing import Any, Dict, Iterator, Optional

import psutil

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


logger = logging.getLogger(__name__)


def _json_default(value: Any) -> Any:
    if isinstance(value, (set, tuple)):
        return list(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def tensor_nbytes(tensor: Any) -> int:
    if tensor is None:
        return 0
    if not hasattr(tensor, "numel") or not hasattr(tensor, "element_size"):
        return 0
    try:
        return int(tensor.numel()) * int(tensor.element_size())
    except Exception:
        return 0


def _safe_cuda_snapshot() -> Dict[str, Optional[int]]:
    out = {
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
        "cuda_max_allocated_bytes": None,
        "cuda_max_reserved_bytes": None,
    }
    if torch is None:
        return out
    try:
        if not torch.cuda.is_available():
            return out
        device_idx = torch.cuda.current_device()
        out["cuda_allocated_bytes"] = int(torch.cuda.memory_allocated(device_idx))
        out["cuda_reserved_bytes"] = int(torch.cuda.memory_reserved(device_idx))
        out["cuda_max_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device_idx))
        out["cuda_max_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device_idx))
    except Exception:
        return out
    return out


class RuntimeProfiler:
    def __init__(
        self,
        *,
        log_dir: str,
        run_name: str,
        interval_sec: float = 0.5,
    ) -> None:
        self.log_dir = os.path.abspath(log_dir)
        self.run_name = run_name
        self.interval_sec = max(0.1, float(interval_sec))
        self.process = psutil.Process(os.getpid())
        self.events_path = os.path.join(self.log_dir, f"{self.run_name}_events.jsonl")
        self.summary_path = os.path.join(self.log_dir, f"{self.run_name}_summary.json")
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._phase_counter = 0
        self._active_phases: Dict[int, Dict[str, Any]] = {}
        self._max_tree_rss_bytes = 0
        self._max_main_rss_bytes = 0
        self._max_cuda_allocated_bytes = 0
        self._event_count = 0

        os.makedirs(self.log_dir, exist_ok=True)
        self._start_sampler()
        self.record_event(
            "profiler_init",
            {
                "log_dir": self.log_dir,
                "run_name": self.run_name,
                "interval_sec": self.interval_sec,
                "pid": os.getpid(),
            },
        )

    def _start_sampler(self) -> None:
        self._thread = threading.Thread(target=self._sample_loop, name="runtime-profiler", daemon=True)
        self._thread.start()

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                snap = self.snapshot()
                with self._lock:
                    self._max_tree_rss_bytes = max(self._max_tree_rss_bytes, int(snap["rss_tree_bytes"]))
                    self._max_main_rss_bytes = max(self._max_main_rss_bytes, int(snap["rss_main_bytes"]))
                    cuda_alloc = snap.get("cuda_allocated_bytes")
                    if isinstance(cuda_alloc, int):
                        self._max_cuda_allocated_bytes = max(self._max_cuda_allocated_bytes, cuda_alloc)
                    for phase_state in self._active_phases.values():
                        phase_state["peak_tree_rss_bytes"] = max(
                            int(phase_state["peak_tree_rss_bytes"]),
                            int(snap["rss_tree_bytes"]),
                        )
                        phase_state["peak_main_rss_bytes"] = max(
                            int(phase_state["peak_main_rss_bytes"]),
                            int(snap["rss_main_bytes"]),
                        )
                        if isinstance(cuda_alloc, int):
                            phase_state["peak_cuda_allocated_bytes"] = max(
                                int(phase_state["peak_cuda_allocated_bytes"]),
                                int(cuda_alloc),
                            )
            except Exception:
                logger.exception("Runtime profiler sampling failed")
            self._stop_event.wait(self.interval_sec)

    def snapshot(self) -> Dict[str, Any]:
        main_mem = self.process.memory_info()
        children = []
        rss_tree = int(main_mem.rss)
        for child in self.process.children(recursive=True):
            try:
                child_mem = child.memory_info()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
            try:
                child_name = child.name()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                child_name = "<exited>"
            try:
                child_status = child.status()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                child_status = "<exited>"
            children.append(
                {
                    "pid": int(child.pid),
                    "name": child_name,
                    "rss_bytes": int(child_mem.rss),
                    "status": child_status,
                }
            )
            rss_tree += int(child_mem.rss)

        vm = psutil.virtual_memory()
        snap: Dict[str, Any] = {
            "ts": time.time(),
            "rss_main_bytes": int(main_mem.rss),
            "vms_main_bytes": int(main_mem.vms),
            "rss_tree_bytes": int(rss_tree),
            "num_children": len(children),
            "children": children,
            "system_available_bytes": int(vm.available),
            "system_used_bytes": int(vm.used),
        }
        snap.update(_safe_cuda_snapshot())
        return snap

    def record_event(self, tag: str, payload: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "tag": str(tag),
            "ts": time.time(),
            "payload": payload or {},
        }
        with open(self.events_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False, default=_json_default) + "\n")
        self._event_count += 1

    @contextlib.contextmanager
    def phase(self, name: str, meta: Optional[Dict[str, Any]] = None) -> Iterator[None]:
        meta = dict(meta or {})
        start_time = time.time()
        start_snap = self.snapshot()
        with self._lock:
            self._phase_counter += 1
            phase_id = self._phase_counter
            self._active_phases[phase_id] = {
                "name": name,
                "meta": meta,
                "start_time": start_time,
                "start_snapshot": start_snap,
                "peak_tree_rss_bytes": int(start_snap["rss_tree_bytes"]),
                "peak_main_rss_bytes": int(start_snap["rss_main_bytes"]),
                "peak_cuda_allocated_bytes": int(start_snap["cuda_allocated_bytes"] or 0),
            }

        self.record_event(
            "phase_start",
            {
                "phase_name": name,
                "meta": meta,
                "snapshot": start_snap,
            },
        )
        try:
            yield
        finally:
            end_time = time.time()
            end_snap = self.snapshot()
            with self._lock:
                phase_state = self._active_phases.pop(phase_id, None)
            if phase_state is None:
                phase_state = {
                    "peak_tree_rss_bytes": int(end_snap["rss_tree_bytes"]),
                    "peak_main_rss_bytes": int(end_snap["rss_main_bytes"]),
                    "peak_cuda_allocated_bytes": int(end_snap["cuda_allocated_bytes"] or 0),
                    "start_snapshot": start_snap,
                }
            self.record_event(
                "phase_end",
                {
                    "phase_name": name,
                    "meta": meta,
                    "duration_sec": end_time - start_time,
                    "start_snapshot": phase_state["start_snapshot"],
                    "end_snapshot": end_snap,
                    "delta_tree_rss_bytes": int(end_snap["rss_tree_bytes"]) - int(start_snap["rss_tree_bytes"]),
                    "delta_main_rss_bytes": int(end_snap["rss_main_bytes"]) - int(start_snap["rss_main_bytes"]),
                    "peak_tree_rss_bytes": int(phase_state["peak_tree_rss_bytes"]),
                    "peak_main_rss_bytes": int(phase_state["peak_main_rss_bytes"]),
                    "peak_cuda_allocated_bytes": int(phase_state["peak_cuda_allocated_bytes"]),
                },
            )

    def close(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self.interval_sec * 2.0))
        final_snap = self.snapshot()
        summary = {
            "run_name": self.run_name,
            "events_path": self.events_path,
            "event_count": self._event_count,
            "max_tree_rss_bytes": int(max(self._max_tree_rss_bytes, final_snap["rss_tree_bytes"])),
            "max_main_rss_bytes": int(max(self._max_main_rss_bytes, final_snap["rss_main_bytes"])),
            "max_cuda_allocated_bytes": int(self._max_cuda_allocated_bytes),
            "final_snapshot": final_snap,
        }
        with open(self.summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2, default=_json_default)
        self.record_event("profiler_close", summary)


_GLOBAL_PROFILER: Optional[RuntimeProfiler] = None


def init_runtime_profiler(
    *,
    log_dir: str,
    run_name: str,
    interval_sec: float = 0.5,
) -> RuntimeProfiler:
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is not None:
        _GLOBAL_PROFILER.close()
    _GLOBAL_PROFILER = RuntimeProfiler(log_dir=log_dir, run_name=run_name, interval_sec=interval_sec)
    return _GLOBAL_PROFILER


def get_runtime_profiler() -> Optional[RuntimeProfiler]:
    return _GLOBAL_PROFILER


def close_runtime_profiler() -> None:
    global _GLOBAL_PROFILER
    if _GLOBAL_PROFILER is not None:
        _GLOBAL_PROFILER.close()
        _GLOBAL_PROFILER = None


@contextlib.contextmanager
def profile_phase(name: str, meta: Optional[Dict[str, Any]] = None) -> Iterator[None]:
    profiler = get_runtime_profiler()
    if profiler is None:
        yield
        return
    with profiler.phase(name, meta):
        yield


def record_profile_event(tag: str, payload: Optional[Dict[str, Any]] = None) -> None:
    profiler = get_runtime_profiler()
    if profiler is None:
        return
    profiler.record_event(tag, payload)

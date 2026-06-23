"""Auto-start and manage vLLM servers for the deep research pipeline.

Examines ``--agentic-model``, ``--llm-model``, and ``--post-*-reranker`` CLI
args to determine which vLLM servers (if any) are required, allocates GPUs,
launches them as background processes, waits for health, and tears them down
on exit.

Usage (inside run_deepresearch_pipeline.py)::

    manager = VLLMServerManager()
    gpu_ids = manager.auto_start(args, total_gpus=8)
    # gpu_ids = list of GPU IDs available for pipeline workers
    try:
        ...  # run pipeline
    finally:
        manager.shutdown()
"""

from __future__ import annotations

import atexit
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import urllib.request
import urllib.error

logger = logging.getLogger(__name__)

# ── Paths ────────────────────────────────────────────────────────────────────
_HF_HOME = "/mnt/sagemaker-nvme/huggingface"
_DOWNLOAD_DIR = "/mnt/sagemaker-nvme/huggingface/hub"
_LOG_DIR = Path("/tmp/vllm_server_logs")

# ── S3 model cache ──────────────────────────────────────────────────────────
# SageMaker VPCs may block HuggingFace downloads.  Models listed here are
# fetched from S3 instead.  Upload once from a machine with HF access:
#
#   aws s3 sync ~/.cache/huggingface/hub/models--Qwen--Qwen3-32B/snapshots/<hash>/ \
#       s3://a204383-ml-workspace-practicallawqw7t-use1/agentic_retrieval/models/Qwen--Qwen3-32B/ \
#       --exclude "*.bin"
#
_S3_BUCKET = "a204383-ml-workspace-practicallawqw7t-use1"
_S3_MODELS_PREFIX = "agentic_retrieval/models"
_LOCAL_MODELS_DIR = Path("/mnt/sagemaker-nvme/models")

_S3_MODEL_CACHE: Dict[str, str] = {
    "Qwen/Qwen3-32B": f"s3://{_S3_BUCKET}/{_S3_MODELS_PREFIX}/Qwen--Qwen3-32B",
    "zai-org/GLM-4.7-Flash": f"s3://{_S3_BUCKET}/{_S3_MODELS_PREFIX}/zai-org--GLM-4.7-Flash",
}


# ═══════════════════════════════════════════════════════════════════════════════
# Server spec dataclass
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ServerSpec:
    """Specification for a single vLLM server instance."""
    model: str
    port: int
    tp_size: int
    max_model_len: Optional[int] = 65536
    max_num_seqs: Optional[int] = 64
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    extra_args: List[str] = field(default_factory=list)
    dtype: Optional[str] = None                # e.g. "float16" for rank1
    enable_prefix_caching: bool = False
    # LoRA support (rank_r1)
    enable_lora: bool = False
    lora_modules: Optional[str] = None         # "alias=path"
    max_lora_rank: Optional[int] = None
    label: str = ""                            # human-readable name for logs


# ═══════════════════════════════════════════════════════════════════════════════
# Static mapping: CLI args → required server specs
# ═══════════════════════════════════════════════════════════════════════════════

# --- LLM servers (port 6008) ------------------------------------------------
_LLM_SERVER_SPECS: Dict[str, ServerSpec] = {
    # key = agentic_model name (for self-managed agents)
    #        or "finetuned:{agentic_model}" (for finetuned variants)
    "oss:gpt-oss-20b": ServerSpec(
        model="openai/gpt-oss-20b",
        port=6008, tp_size=1,
        max_model_len=131072, max_num_seqs=16,
        enforce_eager=True,
        label="GPT-OSS-20B",
    ),
    "oss:gpt-oss-120b": ServerSpec(
        model="openai/gpt-oss-120b",
        port=6008, tp_size=8,
        max_model_len=131072, max_num_seqs=16,
        enforce_eager=True,
        label="GPT-OSS-120B",
    ),
    "glm": ServerSpec(
        model="zai-org/GLM-4.7-Flash",
        port=6008, tp_size=4,
        max_model_len=65536, max_num_seqs=16,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_prefix_caching=True,
        extra_args=["--enable-auto-tool-choice", "--tool-call-parser", "glm47"],
        label="GLM-4.7-Flash",
    ),
    "tongyi": ServerSpec(
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        port=6008, tp_size=4,
        max_model_len=131072, max_num_seqs=16,
        gpu_memory_utilization=0.90,
        enforce_eager=False,
        enable_prefix_caching=True,
        extra_args=["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        label="Tongyi-DeepResearch-30B",
    ),
    "webweaver": ServerSpec(
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        port=6008, tp_size=4,
        max_model_len=131072, max_num_seqs=16,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        extra_args=["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        label="WebWeaver (Tongyi-30B)",
    ),
    "finetuned:cpm_report": ServerSpec(
        model="openbmb/AgentCPM-Report",
        port=6008, tp_size=2,
        max_model_len=65536, max_num_seqs=64,
        gpu_memory_utilization=0.9,
        label="AgentCPM-Report",
    ),
    "finetuned:cpm_explore": ServerSpec(
        model="openbmb/AgentCPM-Explore",
        port=6008, tp_size=1,
        max_model_len=32768, max_num_seqs=64,
        gpu_memory_utilization=0.9,
        label="AgentCPM-Explore",
    ),
    "finetuned:drtulu": ServerSpec(
        model="rl-research/DR-Tulu-8B",
        port=6008, tp_size=1,
        max_model_len=65536, max_num_seqs=64,
        gpu_memory_utilization=0.9,
        label="DR-Tulu-8B",
    ),
    "finetuned:webweaver": ServerSpec(
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        port=6008, tp_size=4,
        max_model_len=131072, max_num_seqs=16,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        extra_args=["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        label="WebWeaver finetuned (Tongyi-30B)",
    ),
    "finetuned:tongyi": ServerSpec(
        model="Alibaba-NLP/Tongyi-DeepResearch-30B-A3B",
        port=6008, tp_size=4,
        max_model_len=131072, max_num_seqs=16,
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        extra_args=["--enable-auto-tool-choice", "--tool-call-parser", "hermes"],
        label="Tongyi finetuned",
    ),
}

# --- Reranker servers (port 8000) -------------------------------------------
_RERANKER_SERVER_SPECS: Dict[str, ServerSpec] = {
    "rank1": ServerSpec(
        model="jhu-clsp/rank1-7b",
        port=8000, tp_size=1,
        max_model_len=4096, max_num_seqs=64,
        gpu_memory_utilization=0.9,
        dtype="float16",
        label="Rank1-7B reranker",
    ),
    "qwen3_reranker": ServerSpec(
        model="Qwen/Qwen3-Reranker-4B",
        port=8000, tp_size=1,
        max_model_len=8192, max_num_seqs=64,
        enable_prefix_caching=True,
        label="Qwen3-Reranker-4B",
    ),
}

# --- Judge server (port 6009) — accuracy evaluation via Qwen3-32B -----------
# Parameters aligned with the original AgentIR evaluation setup
# (https://github.com/texttron/AgentIR/blob/main/evaluation/evaluate_bcp.py).
JUDGE_SERVER_SPEC = ServerSpec(
    model="Qwen/Qwen3-32B",
    port=6009, tp_size=1,
    max_model_len=16384, max_num_seqs=64,
    gpu_memory_utilization=0.90,
    label="Qwen3-32B (judge)",
)

# Agents whose LLM connection is self-managed (they create their own OpenAI client)
_SELF_MANAGED_AGENTS = frozenset({"oss", "tongyi", "glm", "cpm_explore"})


# ═══════════════════════════════════════════════════════════════════════════════
# Manager
# ═══════════════════════════════════════════════════════════════════════════════

class VLLMServerManager:
    """Lifecycle manager for vLLM server processes."""

    def __init__(self):
        self._processes: List[Tuple[ServerSpec, subprocess.Popen]] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def auto_start(
        self,
        args,
        total_gpus: int,
        *,
        health_timeout: int = 900,      # 15 min (large models + downloads)
    ) -> Optional[List[int]]:
        """Resolve needed servers, allocate GPUs, launch, wait for health.

        Args:
            args: Parsed CLI namespace from run_deepresearch_pipeline.py.
            total_gpus: Total number of GPUs on the machine.
            health_timeout: Max seconds to wait for each server to become healthy.

        Returns:
            List of GPU IDs available for pipeline workers, or None if no
            vLLM servers were needed (all GPUs available, existing behavior).
        """
        specs = self._resolve_needed_servers(args)
        if not specs:
            print("[vLLM Manager] No vLLM servers needed for this configuration.")
            return None

        # Check GPU budget
        vllm_gpus_needed = sum(s.tp_size for s in specs)
        if vllm_gpus_needed >= total_gpus:
            print(f"\n[vLLM Manager] ERROR: Need {vllm_gpus_needed} GPUs for vLLM "
                  f"servers but only {total_gpus} available.")
            print(f"  Servers requested:")
            for s in specs:
                print(f"    - {s.label}: TP={s.tp_size} on port {s.port}")
            print(f"\n  Options:")
            print(f"    - Use a larger instance (e.g. ml.g6e.48xlarge for 8 GPUs)")
            print(f"    - Use a cloud LLM (e.g. --llm-model claude-sonnet-4-5) to free GPU budget")
            print(f"    - Remove the vLLM reranker (--post-retrieval-reranker null)")
            sys.exit(1)

        # Allocate GPUs: vLLM servers get leftmost, workers get the rest
        gpu_offset = 0
        server_gpu_assignments: List[Tuple[ServerSpec, List[int]]] = []
        for spec in specs:
            gpu_ids = list(range(gpu_offset, gpu_offset + spec.tp_size))
            server_gpu_assignments.append((spec, gpu_ids))
            gpu_offset += spec.tp_size

        worker_gpu_ids = list(range(gpu_offset, total_gpus))

        # Print allocation summary
        import vllm as _vllm
        print(f"\n{'=' * 70}")
        print(f"[vLLM Manager] GPU Allocation ({total_gpus} GPUs total) — vLLM {_vllm.__version__}")
        print(f"{'=' * 70}")
        for spec, gpus in server_gpu_assignments:
            print(f"  vLLM server : {spec.label}")
            print(f"    Model     : {spec.model}")
            print(f"    Port      : {spec.port}")
            print(f"    GPUs      : {gpus} (TP={spec.tp_size})")
        print(f"  Pipeline workers: GPUs {worker_gpu_ids} ({len(worker_gpu_ids)} workers)")
        print(f"{'=' * 70}\n")

        if len(worker_gpu_ids) == 0:
            print("[vLLM Manager] WARNING: No GPUs left for pipeline workers.")
            print("  The pipeline will run on CPU for retrieval (slow).")

        # Launch servers
        for spec, gpus in server_gpu_assignments:
            self._launch_server(spec, gpus)

        # Register cleanup
        atexit.register(self.shutdown)
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

        # Wait for health
        for spec, _ in server_gpu_assignments:
            self._wait_for_health(spec, timeout=health_timeout)

        return worker_gpu_ids

    @staticmethod
    def _kill_proc_tree(proc: subprocess.Popen, label: str) -> None:
        """Kill a vLLM server and all its worker processes.

        vLLM spawns worker sub-processes; ``proc.terminate()`` only signals
        the main process.  Because we launch with ``preexec_fn=os.setsid``,
        the entire tree shares a process group we can kill at once.
        """
        def _killpg(sig):
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, OSError):
                if sig == signal.SIGTERM:
                    proc.terminate()
                else:
                    proc.kill()

        print(f"  Stopping {label} (PID {proc.pid})...")
        _killpg(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"  Force-killing {label} (PID {proc.pid})...")
            _killpg(signal.SIGKILL)
            proc.wait(timeout=5)

        log_fh = getattr(proc, "_log_file", None) or getattr(getattr(proc, "_spec", None), "_log_file", None)
        if log_fh is not None:
            try:
                log_fh.close()
            except Exception:
                pass

    def shutdown(self):
        """Terminate all managed vLLM server processes."""
        if not self._processes:
            return

        print(f"\n[vLLM Manager] Shutting down {len(self._processes)} server(s)...")
        for spec, proc in self._processes:
            if proc.poll() is None:
                self._kill_proc_tree(proc, spec.label)
        self._processes.clear()
        print("[vLLM Manager] All servers stopped.")

    def shutdown_and_release_gpus(self, total_gpus: int, *, timeout: int = 120) -> None:
        """Shut down all servers, kill orphan port processes, and free GPUs.

        Call this after query processing to guarantee GPUs are available for
        the correctness judge (or any subsequent workload).
        """
        # Collect ports before shutdown clears self._processes
        ports_to_clean = {s.port for s, _ in self._processes}
        for spec_dict in (_LLM_SERVER_SPECS, _RERANKER_SERVER_SPECS):
            for spec in spec_dict.values():
                ports_to_clean.add(spec.port)

        # 1) Kill all managed server processes
        self.shutdown()

        # 2) Kill any orphan processes still bound to query-processing ports
        for port in sorted(ports_to_clean):
            self._kill_port_orphans(port)

        # 3) Force Python-side GPU memory release
        self._force_gpu_cleanup()

        # 4) Wait until nvidia-smi confirms GPUs are free
        gpu_ids = list(range(total_gpus))
        self._wait_for_gpu_release(gpu_ids, min_free_fraction=0.85, timeout=timeout)

    @staticmethod
    def _kill_port_orphans(port: int) -> None:
        """Kill any remaining processes bound to *port* (orphan vLLM workers)."""
        try:
            result = subprocess.run(
                ["fuser", "-n", "tcp", str(port)],
                capture_output=True, text=True, timeout=5,
            )
            pids = result.stdout.strip().split()
            if not pids:
                return
            for pid_str in pids:
                pid = int(pid_str.strip())
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  [vLLM Manager] Killed orphan PID {pid} on port {port}")
                except (ProcessLookupError, PermissionError):
                    pass
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def start_judge_server(
        self,
        total_gpus: int,
        *,
        health_timeout: int = 900,
    ) -> List[str]:
        """Start one or more judge vLLM servers (Qwen3-32B).

        Intended to be called **after** :meth:`shutdown` has freed the
        query-processing server GPUs.  When the model fits on a single GPU,
        launches one server per available GPU for maximum throughput.
        Otherwise falls back to a single TP-sharded server.

        Returns:
            List of ``"http://localhost:<port>/v1"`` base URLs for each
            judge server that was started.
        """
        spec = JUDGE_SERVER_SPEC
        tp_size = self._compute_judge_tp_size(total_gpus, spec.gpu_memory_utilization)

        num_servers = total_gpus // tp_size

        import vllm as _vllm
        print(f"\n{'=' * 70}")
        print(f"[vLLM Manager] Starting {num_servers} judge server(s) — vLLM {_vllm.__version__}")
        print(f"{'=' * 70}")
        print(f"  Model   : {spec.model}")
        print(f"  Servers : {num_servers}  (TP={tp_size} each)")
        print(f"  Ports   : {spec.port}–{spec.port + num_servers - 1}")
        print(f"{'=' * 70}\n")

        self._force_gpu_cleanup()

        base_urls: List[str] = []
        for i in range(num_servers):
            gpu_ids = list(range(i * tp_size, i * tp_size + tp_size))
            port = spec.port + i
            server_spec = ServerSpec(
                model=spec.model,
                port=port,
                tp_size=tp_size,
                max_model_len=spec.max_model_len,
                max_num_seqs=spec.max_num_seqs,
                gpu_memory_utilization=spec.gpu_memory_utilization,
                label=f"{spec.label} [{i}]",
            )
            print(f"  Launching judge server {i}: port={port}, GPUs={gpu_ids}")
            self._wait_for_gpu_release(
                gpu_ids,
                min_free_fraction=spec.gpu_memory_utilization - 0.05,
                timeout=60,
            )
            self._launch_server(server_spec, gpu_ids)
            base_urls.append(f"http://localhost:{port}/v1")

        for i in range(num_servers):
            port = spec.port + i
            server_spec = ServerSpec(
                model=spec.model, port=port, tp_size=tp_size,
                max_model_len=spec.max_model_len, max_num_seqs=spec.max_num_seqs,
                gpu_memory_utilization=spec.gpu_memory_utilization,
                label=f"{spec.label} [{i}]",
            )
            self._wait_for_health(server_spec, timeout=health_timeout)

        return base_urls

    def shutdown_judge_server(self) -> None:
        """Terminate all judge server processes."""
        judge_ports = set(
            range(JUDGE_SERVER_SPEC.port, JUDGE_SERVER_SPEC.port + 16)
        )
        judge_procs = [(s, p) for s, p in self._processes if s.port in judge_ports]
        for spec, proc in judge_procs:
            if proc.poll() is None:
                self._kill_proc_tree(proc, spec.label)
        self._processes = [(s, p) for s, p in self._processes if s.port not in judge_ports]

    # ── TP size selection ─────────────────────────────────────────────────

    @staticmethod
    def _compute_judge_tp_size(
        total_gpus: int,
        gpu_memory_utilization: float,
        model_gb: float = 70.0,
    ) -> int:
        """Choose the smallest TP size that fits the judge model.

        Queries nvidia-smi for per-GPU memory.  Falls back to TP=total_gpus
        if the query fails.
        """
        import math
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.total",
                 "--format=csv,noheader,nounits", "-i", "0"],
                capture_output=True, text=True, timeout=10,
            )
            per_gpu_mb = int(result.stdout.strip().split("\n")[0].strip())
            per_gpu_gb = per_gpu_mb / 1024
            usable_per_gpu = per_gpu_gb * gpu_memory_utilization
            tp = max(1, math.ceil(model_gb / usable_per_gpu))
            tp = min(tp, total_gpus)
            print(f"[vLLM Manager] Per-GPU memory: {per_gpu_gb:.1f} GB "
                  f"(usable {usable_per_gpu:.1f} GB) → judge TP={tp}")
            return tp
        except Exception:
            tp = min(4, total_gpus)
            print(f"[vLLM Manager] Could not query GPU memory — "
                  f"defaulting judge TP={tp}")
            return tp

    # ── Resolve which servers are needed ────────────────────────────────────

    def _resolve_needed_servers(self, args) -> List[ServerSpec]:
        """Determine which vLLM servers are required based on CLI args."""
        specs: List[ServerSpec] = []
        seen_ports: set = set()

        # --- LLM server ---
        llm_spec = self._resolve_llm_server(args)
        if llm_spec is not None:
            specs.append(llm_spec)
            seen_ports.add(llm_spec.port)

        # --- Reranker servers ---
        reranker_names = set()
        if getattr(args, "post_retrieval_reranker", "null") != "null":
            reranker_names.add(args.post_retrieval_reranker)
        if getattr(args, "post_fusion_reranker", "null") != "null":
            reranker_names.add(args.post_fusion_reranker)

        for name in reranker_names:
            if name in _RERANKER_SERVER_SPECS:
                spec = _RERANKER_SERVER_SPECS[name]
                if spec.port not in seen_ports:
                    specs.append(spec)
                    seen_ports.add(spec.port)

        return specs

    def _resolve_llm_server(self, args) -> Optional[ServerSpec]:
        """Resolve the LLM server spec from agentic_model + llm_model."""
        agentic_model = getattr(args, "agentic_model", "")
        llm_model = getattr(args, "llm_model", "")

        # Self-managed agents always need their own vLLM server
        if agentic_model in _SELF_MANAGED_AGENTS:
            # OSS agent: model variant is determined by --llm-model
            if agentic_model == "oss":
                key = f"oss:{llm_model}"
                if key in _LLM_SERVER_SPECS:
                    return _LLM_SERVER_SPECS[key]
                # Default to oss:gpt-oss-20b if llm_model not recognized
                return _LLM_SERVER_SPECS.get("oss:gpt-oss-20b")
            # tongyi, glm
            if agentic_model in _LLM_SERVER_SPECS:
                return _LLM_SERVER_SPECS[agentic_model]

        # Finetuned models served via vLLM — match by HF model name
        key = f"finetuned:{agentic_model}"
        if key in _LLM_SERVER_SPECS:
            spec = _LLM_SERVER_SPECS[key]
            if llm_model == spec.model:
                return spec

        # webweaver with non-finetuned llm_model still self-manages
        if agentic_model == "webweaver" and llm_model not in (
            "claude-sonnet-4-5", "gpt-4.1", "gpt-4.1-mini", "qwen3-max",
            "qwen3-235B-A22B", "gpt-5.1",
        ):
            return _LLM_SERVER_SPECS.get("webweaver")

        # Cloud models (claude-*, gpt-*, qwen3-max) → no vLLM needed
        return None

    # ── S3 model cache ─────────────────────────────────────────────────────

    @staticmethod
    def _ensure_model_local(model_name: str) -> str:
        """Return a local path to model weights, downloading from S3 if needed.

        If the model is not in ``_S3_MODEL_CACHE``, returns ``model_name``
        unchanged (vLLM will download from HuggingFace as usual).
        """
        if model_name not in _S3_MODEL_CACHE:
            return model_name

        local_dir = _LOCAL_MODELS_DIR / model_name.replace("/", "--")

        # Already downloaded to our local cache
        if local_dir.exists() and any(local_dir.glob("*.safetensors")):
            print(f"[vLLM Manager] Model cached at {local_dir}")
            return str(local_dir)

        # Check HF cache (works locally where the model was downloaded before)
        hf_cache = Path(_DOWNLOAD_DIR) / f"models--{model_name.replace('/', '--')}"
        if hf_cache.exists() and any(hf_cache.glob("snapshots/*/*.safetensors")):
            print(f"[vLLM Manager] Model found in HF cache: {hf_cache}")
            return model_name

        # Download from S3
        s3_uri = _S3_MODEL_CACHE[model_name]
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[vLLM Manager] Downloading {model_name} from S3...")
        print(f"  S3 source : {s3_uri}")
        print(f"  Local dest: {local_dir}")

        result = subprocess.run(
            ["aws", "s3", "sync", s3_uri, str(local_dir), "--quiet"],
            capture_output=True, text=True, timeout=3600,
        )
        if result.returncode != 0:
            print(f"[vLLM Manager] WARNING: S3 download failed: {result.stderr}")
            print(f"  Falling back to HuggingFace download...")
            return model_name

        print(f"[vLLM Manager] Model downloaded successfully ({local_dir})")
        return str(local_dir)

    # ── Launch a server process ─────────────────────────────────────────────

    def _launch_server(self, spec: ServerSpec, gpu_ids: List[int]):
        """Start a vLLM server as a background subprocess."""
        # Check if port is already in use (server already running)
        if self._port_in_use(spec.port):
            print(f"[vLLM Manager] Port {spec.port} already in use — "
                  f"assuming {spec.label} is already running, skipping launch.")
            return

        # Resolve model path (download from S3 if needed on SageMaker)
        model_path = self._ensure_model_local(spec.model)

        # Build vllm serve command
        cmd = [
            sys.executable, "-m", "vllm.entrypoints.openai.api_server",
            "--model", model_path,
            "--port", str(spec.port),
        ]
        # When model_path is a local directory (S3 cache), vLLM registers
        # the model under that path.  Clients request spec.model (the HF
        # name), so we need --served-model-name to bridge the mismatch.
        if model_path != spec.model:
            cmd.extend(["--served-model-name", spec.model])
        cmd += [
            "--tensor-parallel-size", str(spec.tp_size),
            "--download-dir", _DOWNLOAD_DIR,
            "--trust-remote-code",
            "--gpu-memory-utilization", str(spec.gpu_memory_utilization),
        ]
        if spec.max_model_len is not None:
            cmd.extend(["--max-model-len", str(spec.max_model_len)])
        if spec.max_num_seqs is not None:
            cmd.extend(["--max-num-seqs", str(spec.max_num_seqs)])
        if spec.enforce_eager:
            cmd.append("--enforce-eager")
        if spec.dtype:
            cmd.extend(["--dtype", spec.dtype])
        if spec.enable_prefix_caching:
            cmd.append("--enable-prefix-caching")
        if spec.enable_lora:
            cmd.append("--enable-lora")
            if spec.lora_modules:
                cmd.extend(["--lora-modules", spec.lora_modules])
            if spec.max_lora_rank is not None:
                cmd.extend(["--max-lora-rank", str(spec.max_lora_rank)])
        cmd.extend(spec.extra_args)

        # Environment
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)
        env["HF_HOME"] = _HF_HOME

        # vLLM V1 engine relies on /dev/shm for multiprocessing; SageMaker
        # containers typically have only ~2 GB which causes silent crashes.
        try:
            shm = os.statvfs("/dev/shm")
            shm_gb = (shm.f_frsize * shm.f_blocks) / (1024 ** 3)
            if shm_gb < 8:
                env.setdefault("VLLM_USE_V1", "0")
                print(f"[vLLM Manager] /dev/shm is {shm_gb:.1f} GB — disabling vLLM V1 engine")
        except OSError:
            pass

        # Log files
        _LOG_DIR.mkdir(parents=True, exist_ok=True)
        log_path = _LOG_DIR / f"vllm_{spec.port}.log"

        print(f"[vLLM Manager] Launching {spec.label} on GPUs {gpu_ids}, port {spec.port}...")
        print(f"  Log: {log_path}")

        log_file = open(log_path, "w")
        proc = subprocess.Popen(
            cmd,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            env=env,
            preexec_fn=os.setsid,  # new process group for clean kill
        )
        self._processes.append((spec, proc))
        # Keep log file handle on the spec for cleanup
        spec._log_file = log_file  # type: ignore[attr-defined]

        print(f"  PID: {proc.pid}")

    # ── Health check ────────────────────────────────────────────────────────

    def _wait_for_health(self, spec: ServerSpec, timeout: int = 900):
        """Poll the vLLM health endpoint until it responds or timeout."""
        url = f"http://localhost:{spec.port}/health"
        start = time.time()
        interval = 2.0
        last_status = ""

        print(f"[vLLM Manager] Waiting for {spec.label} on port {spec.port}...")

        while time.time() - start < timeout:
            # Check if process died
            for s, proc in self._processes:
                if s.port == spec.port and proc.poll() is not None:
                    print(f"\n[vLLM Manager] ERROR: {spec.label} process died "
                          f"(exit code {proc.returncode}).")
                    log_path = _LOG_DIR / f"vllm_{spec.port}.log"
                    print(f"  Check log: {log_path}")
                    if log_path.exists():
                        full_log = log_path.read_text()
                        print(f"  === Full vLLM log ({len(full_log.splitlines())} lines) ===")
                        print(full_log)
                        print(f"  === End of vLLM log ===")
                    sys.exit(1)

            try:
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=5) as resp:
                    if resp.status == 200:
                        elapsed = time.time() - start
                        print(f"[vLLM Manager] {spec.label} is healthy! "
                              f"(took {elapsed:.0f}s)")
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                pass

            # Progress indicator
            elapsed = int(time.time() - start)
            status = f"  ... waiting ({elapsed}s / {timeout}s)"
            if status != last_status and elapsed % 30 == 0 and elapsed > 0:
                print(status)
                last_status = status

            time.sleep(interval)
            # Back off gradually
            interval = min(interval * 1.2, 10.0)

        print(f"\n[vLLM Manager] TIMEOUT: {spec.label} did not become healthy "
              f"within {timeout}s.")
        print(f"  Check log: {_LOG_DIR / f'vllm_{spec.port}.log'}")
        self.shutdown()
        sys.exit(1)

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _force_gpu_cleanup() -> None:
        """Force Python-side GPU memory release (cache + garbage collection)."""
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                print("[vLLM Manager] Forced GPU cache cleanup on all devices.")
        except ImportError:
            pass

    @staticmethod
    def _wait_for_gpu_release(
        gpu_ids: List[int],
        min_free_fraction: float = 0.85,
        timeout: int = 120,
    ) -> None:
        """Poll nvidia-smi until GPU memory is substantially free."""
        ids_str = ",".join(str(g) for g in gpu_ids)
        start = time.time()
        print(f"[vLLM Manager] Waiting for GPU memory release on devices {gpu_ids} "
              f"(need {min_free_fraction:.0%} free)...")
        _last_print = 0.0
        while time.time() - start < timeout:
            try:
                result = subprocess.run(
                    ["nvidia-smi",
                     "--query-gpu=memory.free,memory.total",
                     "--format=csv,noheader,nounits",
                     "-i", ids_str],
                    capture_output=True, text=True, timeout=10,
                )
                lines = result.stdout.strip().split("\n")
                all_free = True
                for idx, line in enumerate(lines):
                    free_mb, total_mb = (int(x.strip()) for x in line.split(","))
                    frac = free_mb / total_mb
                    elapsed = time.time() - start
                    if elapsed - _last_print >= 10:
                        used_mb = total_mb - free_mb
                        print(f"  GPU {gpu_ids[idx]}: {free_mb}MB free / {total_mb}MB total "
                              f"({frac:.1%} free, {used_mb}MB in use)")
                        _last_print = elapsed
                    if frac < min_free_fraction:
                        all_free = False
                if all_free:
                    elapsed = time.time() - start
                    print(f"[vLLM Manager] GPUs {gpu_ids} memory released ({elapsed:.0f}s)")
                    return
            except Exception:
                pass
            time.sleep(3)
        print(f"[vLLM Manager] WARNING: GPUs {gpu_ids} still have memory in use "
              f"after {timeout}s — proceeding anyway")

    @staticmethod
    def _port_in_use(port: int) -> bool:
        """Check if a port is already bound."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

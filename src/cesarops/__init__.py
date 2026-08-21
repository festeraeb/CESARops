"""
CESAROPS — Coastal/Estuarine SAR Operations System

Satellite sensor fusion, GPU/TPU anomaly detection, and AI-directed
scanning for underwater target identification.

Core components:
  engine       — CUDA/TPU processing pipeline
  orchestrator — Multi-sensor probe orchestrator
  agent_runner — Agent execution runner
  ai_director  — Qwen LLM director
  database     — Census database interface
  tpu_client   — Remote TPU client
  triple_lock  — Triple-lock fusion detection

Subpackages:
  drift        — SAR drift modeling and search-and-rescue operations
  scanner      — Satellite data acquisition (NASA CMR, HLS, SWOT, SAR, Landsat)
"""

__version__ = "0.3.0"
__all__ = [
    "engine",
    "orchestrator",
    "agent_runner",
    "ai_director",
    "database",
    "search_planner",
    "tpu_client",
    "tpu_server",
    "triple_lock",
    "llm_context",
    "remote_dispatch",
    "tile_selector",
    "tile_geometry",
    # SAR drift modeling subpackage
    "drift",
    # Satellite scanner subpackage
    "scanner",
]

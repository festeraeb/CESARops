"""
cesarops.drift — Search and Rescue Drift Modeling

Ensemble particle drift simulation for Great Lakes SAR operations.
Integrates real buoy/ERDDAP data sources, ML-enhanced predictions,
and GLOS Seagull environmental conditions.

Quick start:
    from cesarops.drift import SimpleFastDriftEngine, MLDriftPredictor

    engine = SimpleFastDriftEngine()
    results = engine.simulate_drift_ensemble(
        release_lat=42.995, release_lon=-87.845,
        release_time=datetime.utcnow(),
        duration_hours=72,
        environmental_data=[...],
        n_particles=1000,
    )

Modules:
    engine            — SimpleFastDriftEngine: multi-worker ensemble simulation
    engine_numba      — FastDriftEngine: Numba JIT-compiled variant (optional)
    analyzer          — AdvancedDriftAnalyzer: ML + Rust-core integration
    ml_predictor      — MLDriftPredictor: Random Forest / regression models
    drifter_collector — RealDrifterCollector: ERDDAP/GLOS live drifter data
    training_pipeline — DrifterTrainingPipeline: NOAA GDP training data
    weather_fetcher   — HistoricalWeatherFetcher: NDBC/ERDDAP historical wx
    glos_analyzer     — GlosSeagullAnalyzer: GLOS Seagull buoy analysis
    sarops_core       — EnhancedOceanDrift + Great Lakes env fetching
    sar_cases         — SARCase, TeamAlert, CollaborativeOps dataclasses
"""

from .engine import SimpleFastDriftEngine
from .ml_predictor import MLDriftPredictor
from .drifter_collector import RealDrifterCollector
from .glos_analyzer import GlosSeagullAnalyzer

__all__ = [
    "SimpleFastDriftEngine",
    "MLDriftPredictor",
    "RealDrifterCollector",
    "GlosSeagullAnalyzer",
    # lazy imports (heavy deps)
    "engine",
    "engine_numba",
    "analyzer",
    "training_pipeline",
    "weather_fetcher",
    "sarops_core",
    "sar_cases",
]

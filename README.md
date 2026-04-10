# CESAROPS

**Coastal/Estuarine SAR Operations System**

Satellite sensor fusion, GPU/TPU anomaly detection, and AI-directed scanning for underwater target identification on the Great Lakes.

---

## Structure

```
cesarops/
  src/cesarops/          Python package (engine, orchestrator, AI director, TPU, database)
  slicer/                Rust crate — zero-copy GeoTIFF tile slicer
  config/                Sensor configs, schema, knowledge base
  docs/                  Agent context and instructions
  tests/                 Test suite
  pyproject.toml         Python package manifest
  Cargo.toml             Rust workspace (includes slicer/)
```

## Python Package

```bash
pip install -e ".[gpu]"       # With CUDA GPU support
pip install -e ".[dev]"       # With test tools
```

### Key modules

| Module | Purpose |
|---|---|
| `cesarops.engine` | CUDA/TPU processing pipeline |
| `cesarops.orchestrator` | Multi-sensor probe orchestrator |
| `cesarops.agent_runner` | Agent execution entrypoint |
| `cesarops.ai_director` | Qwen LLM director |
| `cesarops.database` | Census DB interface |
| `cesarops.tpu_client` | Remote TPU (Xenon) client |
| `cesarops.triple_lock` | Triple-lock fusion detection |
| `cesarops.remote_dispatch` | Pi→Xenon dispatch |

## Rust Slicer

```bash
cd slicer
cargo build --release
./target/release/slicer --help
```

The slicer reads GeoTIFF files using memory-mapped I/O, bakes coordinates into each tile, and outputs sliced tile JSON for distributed processing.

## Hardware

- **Local GPU**: NVIDIA M2200 / P1000 (CuPy CUDA)
- **Remote TPU**: Coral TPU on Xenon (10.0.0.40:5001)
- **Remote dispatch**: Pi slice → Xenon process via SSH

## Configuration

Copy and edit `.env.example` → `.env`:

```
QWEN_API_KEY=...
EARTHDATA_TOKEN=...
SENTINEL_HUB_CLIENT_ID=...
SENTINEL_HUB_CLIENT_SECRET=...
```

Sensor config: `config/sensor_config.json`  
DB schema: `config/schema.sql`  

"""
cesarops.scanner — Satellite data acquisition and cross-referencing.

Modules:
    cmr_search          NASA CMR live granule query (HLS, SAR, SWOT, ICESat-2)
    universal_downloader Multi-source satellite downloader (ASF, Copernicus, PO.DAAC, USGS, HLS)
    background_probe    Continuous background probe & ML parameter tuning
    swot_ssh_extractor  SWOT SSH anomaly extractor (PO.DAAC Expert granules)
    crossref_scans      Cross-reference multi-year scan outputs for persistent anomalies
"""

from .cmr_search import main as cmr_search
from .universal_downloader import main as download
from .background_probe import main as background_probe
from .swot_ssh_extractor import main as swot_extract
from .crossref_scans import main as crossref

__all__ = [
    "cmr_search",
    "download",
    "background_probe",
    "swot_extract",
    "crossref",
]

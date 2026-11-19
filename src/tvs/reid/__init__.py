"""Vehicle re-identification inference components."""

from .config import ReIDConfig
from .embedder import VehicleEmbedder
from .model import VehicleReIDModel

__all__ = ["ReIDConfig", "VehicleEmbedder", "VehicleReIDModel"]

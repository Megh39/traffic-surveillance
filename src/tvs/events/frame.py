from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Frame:
    frame_id: int
    elapsed_seconds: float
    image: np.ndarray
    source_path: Path
    fps_effective: float

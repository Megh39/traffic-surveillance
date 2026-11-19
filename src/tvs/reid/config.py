from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ReIDConfig:
    """Runtime settings for the pretrained vehicle Re-ID model."""

    model_path: Path
    input_width: int = 208
    input_height: int = 208
    intra_op_num_threads: int = 2
    inter_op_num_threads: int = 1
    normalize_embeddings: bool = True

    def validate(self) -> None:
        if self.input_width != 208 or self.input_height != 208:
            raise ValueError(
                "vehicle-reid-0001 expects a 208x208 input. "
                "Change this only if you replace the model."
            )
        if self.intra_op_num_threads < 1:
            raise ValueError("intra_op_num_threads must be at least 1")
        if self.inter_op_num_threads < 1:
            raise ValueError("inter_op_num_threads must be at least 1")
        if not Path(self.model_path).is_file():
            raise FileNotFoundError(
                f"Vehicle Re-ID ONNX model not found: {self.model_path}. "
                "Run scripts/download_reid_model.py first."
            )

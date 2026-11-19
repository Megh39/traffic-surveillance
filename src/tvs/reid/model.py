from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .config import ReIDConfig


class VehicleReIDModel:
    """CPU-only ONNX inference wrapper for Open Model Zoo vehicle-reid-0001.

    Expected input: BGR OpenCV image.
    Model preprocessing follows the published sample: resize to 208x208,
    convert BGR to RGB, transpose to NCHW, float32, with values in [0, 255].
    The returned feature is L2-normalized by default.
    """

    def __init__(self, config: ReIDConfig):
        config.validate()
        self.config = config

        options = ort.SessionOptions()
        options.intra_op_num_threads = config.intra_op_num_threads
        options.inter_op_num_threads = config.inter_op_num_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )

        self.session = ort.InferenceSession(
            str(Path(config.model_path)),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )

        providers = self.session.get_providers()
        if not providers or providers[0] != "CPUExecutionProvider":
            raise RuntimeError(
                f"CPUExecutionProvider was not selected: {providers}"
            )

        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        if len(inputs) != 1 or len(outputs) < 1:
            raise RuntimeError("Unexpected ONNX model input/output signature")

        self.input_name = inputs[0].name
        self.output_name = outputs[0].name
        self.embedding_dimension = 512

        expected_shape = inputs[0].shape
        # The model card specifies [1, 3, 208, 208]. Validate concrete
        # dimensions when ONNX Runtime exposes them as integers.
        expected = (1, 3, 208, 208)
        for actual, wanted in zip(expected_shape, expected):
            if isinstance(actual, int) and actual != wanted:
                raise RuntimeError(
                    f"Unexpected model input shape {expected_shape}; "
                    f"expected {expected}"
                )

    def preprocess(self, bgr_image: np.ndarray) -> np.ndarray:
        if (
            bgr_image is None
            or not isinstance(bgr_image, np.ndarray)
            or bgr_image.ndim != 3
            or bgr_image.shape[2] != 3
            or bgr_image.size == 0
        ):
            raise ValueError("Expected a non-empty three-channel BGR image")

        resized = cv2.resize(
            bgr_image,
            (self.config.input_width, self.config.input_height),
            interpolation=cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        chw = np.transpose(rgb, (2, 0, 1)).astype(np.float32, copy=False)
        return np.expand_dims(np.ascontiguousarray(chw), axis=0)

    def extract(self, bgr_image: np.ndarray) -> np.ndarray:
        tensor = self.preprocess(bgr_image)
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: tensor},
        )
        embedding = np.asarray(outputs[0], dtype=np.float32).reshape(-1)

        if embedding.size != self.embedding_dimension:
            raise RuntimeError(
                f"Expected {self.embedding_dimension}-D embedding, "
                f"received {embedding.size}-D output"
            )
        if not np.all(np.isfinite(embedding)):
            raise RuntimeError("Model produced NaN or infinite embedding values")

        if self.config.normalize_embeddings:
            norm = float(np.linalg.norm(embedding))
            if norm <= 1e-12:
                raise RuntimeError("Model produced a zero-norm embedding")
            embedding = embedding / norm

        return embedding.astype(np.float32, copy=False)

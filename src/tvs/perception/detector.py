
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import torch
import yaml
from ultralytics import YOLO

from src.tvs.events.detection import (
    Detection,
    map_coco_vehicle_class,
)
from src.tvs.events.frame import Frame
from src.tvs.perception.boxes import xyxy_to_tlwh


@dataclass(frozen=True, slots=True)
class DetectorConfig:
    weights_path: Path
    device: str
    half: bool
    image_size: int
    confidence_threshold: float
    publish_confidence_threshold: float
    iou_threshold: float
    class_agnostic_nms: bool
    vehicle_class_ids: tuple[int, ...]

    @classmethod
    def from_yaml(
        cls,
        config_path: Path,
        project_root: Path,
    ) -> "DetectorConfig":
        config_path = Path(config_path)
        project_root = Path(project_root)

        if not config_path.is_file():
            raise FileNotFoundError(
                f"Detector configuration not found: {config_path}"
            )

        with config_path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError(
                "Detector configuration must be a YAML mapping"
            )

        try:
            model = config["model"]
            inference = config["inference"]
            classes = config["classes"]

            weights = Path(model["weights"])
            weights_path = (
                weights if weights.is_absolute()
                else project_root / weights
            )

            detector_config = cls(
                weights_path=weights_path.resolve(),
                device=str(model["device"]).lower(),
                half=bool(model["half"]),
                image_size=int(inference["image_size"]),
                confidence_threshold=float(
                    inference["confidence_threshold"]
                ),
                publish_confidence_threshold=float(
                    inference["publish_confidence_threshold"]
                ),
                iou_threshold=float(inference["iou_threshold"]),
                class_agnostic_nms=bool(
                    inference["class_agnostic_nms"]
                ),
                vehicle_class_ids=tuple(
                    int(class_id)
                    for class_id in classes["vehicle_class_ids"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid detector configuration: {exc}"
            ) from exc

        detector_config.validate()
        return detector_config

    def validate(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "0"}:
            raise ValueError(
                "device must be one of: auto, cpu, cuda, 0"
            )

        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

        for name, value in (
            ("confidence_threshold", self.confidence_threshold),
            (
                "publish_confidence_threshold",
                self.publish_confidence_threshold,
            ),
            ("iou_threshold", self.iou_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1"
                )

        if (
            self.publish_confidence_threshold
            < self.confidence_threshold
        ):
            raise ValueError(
                "publish_confidence_threshold must be greater than "
                "or equal to confidence_threshold"
            )

        if not self.vehicle_class_ids:
            raise ValueError(
                "vehicle_class_ids cannot be empty"
            )

        if any(class_id < 0 for class_id in self.vehicle_class_ids):
            raise ValueError(
                "vehicle class IDs cannot be negative"
            )

        if len(set(self.vehicle_class_ids)) != len(
            self.vehicle_class_ids
        ):
            raise ValueError(
                "vehicle class IDs must be unique"
            )


class VehicleDetector:
    """
    Loads YOLO and converts its predictions into Detection objects.

    Model loading happens once during initialization.
    """

    def __init__(self, config: DetectorConfig):
        self.config = config

        if not config.weights_path.is_file():
            raise FileNotFoundError(
                f"YOLO model weights not found: {config.weights_path}"
            )

        self.device = self._resolve_device(config.device)
        self.use_half = config.half and self.device != "cpu"

        self.discarded_non_vehicle_count = 0
        self.discarded_vehicle_class_count = 0
        self.clipped_box_count = 0
        self.degenerate_box_count = 0
        self.inference_failure_count = 0

        self.last_inference_latency_ms = 0.0
        self.effective_image_size = config.image_size
        self.last_inference_device = self.device
        self.last_inference_image_size = config.image_size
        print(f"Loading YOLO model: {config.weights_path}")
        print(f"Execution device  : {self.device}")
        print(f"FP16 enabled      : {self.use_half}")

        self.model = YOLO(str(config.weights_path))

        print("YOLO model loaded successfully.")

    @staticmethod
    def _resolve_device(requested_device: str) -> str:
        cuda_available = torch.cuda.is_available()

        if requested_device == "cpu":
            return "cpu"

        if requested_device in {"cuda", "0"}:
            if not cuda_available:
                raise RuntimeError(
                    "CUDA was requested, but no CUDA-compatible GPU "
                    "is available."
                )
            return "0"

        if requested_device == "auto":
            return "0" if cuda_available else "cpu"

        raise ValueError(
            f"Unsupported execution device: {requested_device}"
        )

    def _run_prediction(
        self,
        frame: Frame,
        device: str,
        image_size: int,
    ):
        """Execute YOLO inference using the supplied runtime settings."""
        use_half = self.config.half and device != "cpu"

        return self.model.predict(
            source=frame.image,
            imgsz=image_size,
            conf=self.config.confidence_threshold,
            iou=self.config.iou_threshold,
            device=device,
            quantize=16 if use_half else 32,
            agnostic_nms=self.config.class_agnostic_nms,
            verbose=False,
        )

    @staticmethod
    def _is_cuda_oom(exc: Exception) -> bool:
        """Identify CUDA out-of-memory exceptions."""
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)

        if oom_type is not None and isinstance(exc, oom_type):
            return True

        message = str(exc).casefold()

        return (
            "out of memory" in message
            and ("cuda" in message or "gpu" in message)
        )

    def _recover_from_cuda_oom(
        self,
        frame: Frame,
        failed_image_size: int,
    ):
        """
        Recover from CUDA OOM using reduced CUDA resolution,
        followed by CPU inference.
        """
        print("CUDA out of memory. Clearing GPU cache.")

        torch.cuda.empty_cache()

        reduced_size = max(160, failed_image_size // 2)

        # First fallback: retry CUDA with reduced resolution.
        if reduced_size < failed_image_size:
            print(
                f"Retrying CUDA inference at "
                f"{reduced_size}x{reduced_size}."
            )

            try:
                results = self._run_prediction(
                    frame,
                    device="0",
                    image_size=reduced_size,
                )

                self.effective_image_size = reduced_size
                self.last_inference_device = "0"
                self.last_inference_image_size = reduced_size

                print(
                    f"CUDA fallback successful at "
                    f"{reduced_size}x{reduced_size}."
                )

                return results

            except Exception as exc:
                print(
                    f"Reduced-resolution CUDA inference failed: {exc}"
                )
                torch.cuda.empty_cache()

        # Second fallback: use CPU with FP32.
        cpu_size = reduced_size

        print(
            f"Falling back to CPU inference at "
            f"{cpu_size}x{cpu_size}."
        )

        for attempt in range(2):
            try:
                results = self._run_prediction(
                    frame,
                    device="cpu",
                    image_size=cpu_size,
                )

                self.effective_image_size = cpu_size
                self.last_inference_device = "cpu"
                self.last_inference_image_size = cpu_size

                print(
                    f"CPU fallback successful at "
                    f"{cpu_size}x{cpu_size}."
                )

                return results

            except Exception as exc:
                if attempt == 0:
                    print(
                        f"CPU inference failed. Retrying once: {exc}"
                    )
                else:
                    print(
                        f"CPU inference failed after retry: {exc}"
                    )

        return None

    def detect(self, frame: Frame) -> list[Detection]:
        """
        Run YOLO inference on one frame and return vehicle detections.

        All valid detections above the inference threshold are retained
        for the tracker. The publishing threshold is applied separately.
        """
        image_height, image_width = frame.image.shape[:2]
        # Run inference using the most recently successful settings.
        device = self.last_inference_device
        image_size = self.effective_image_size

        results = None
        start_time = perf_counter()

        try:
            results = self._run_prediction(
                frame,
                device=device,
                image_size=image_size,
            )

            self.last_inference_device = device
            self.last_inference_image_size = image_size

        except Exception as exc:
            if device != "cpu" and self._is_cuda_oom(exc):
                results = self._recover_from_cuda_oom(
                    frame,
                    failed_image_size=image_size,
                )

            else:
                print(
                    f"YOLO inference failed. Retrying once: {exc}"
                )

                try:
                    results = self._run_prediction(
                        frame,
                        device=device,
                        image_size=image_size,
                    )

                    self.last_inference_device = device
                    self.last_inference_image_size = image_size

                except Exception as retry_exc:
                    if (
                        device != "cpu"
                        and self._is_cuda_oom(retry_exc)
                    ):
                        results = self._recover_from_cuda_oom(
                            frame,
                            failed_image_size=image_size,
                        )
                    else:
                        print(
                            "YOLO inference failed after retry: "
                            f"{retry_exc}"
                        )

        self.last_inference_latency_ms = (
            perf_counter() - start_time
        ) * 1000

        if results is None:
            self.inference_failure_count += 1
            return []

        if not results:
            return []


        result = results[0]

        if result.boxes is None:
            return []

        detections: list[Detection] = []

        for box in result.boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())

            source_label = self.model.names[class_id]

            vehicle_type = map_coco_vehicle_class(
                class_id,
                source_label,
            )

            if vehicle_type is None:
                self.discarded_non_vehicle_count += 1
                continue

            if class_id not in self.config.vehicle_class_ids:
                self.discarded_vehicle_class_count += 1
                continue

            x1, y1, x2, y2 = (
                float(value)
                for value in box.xyxy[0].tolist()
            )

            # Clip bounding-box coordinates to image dimensions.
            clipped_box = (
                max(0.0, min(x1, float(image_width))),
                max(0.0, min(y1, float(image_height))),
                max(0.0, min(x2, float(image_width))),
                max(0.0, min(y2, float(image_height))),
            )

            if clipped_box != (x1, y1, x2, y2):
                self.clipped_box_count += 1

            x1, y1, x2, y2 = clipped_box

            # Reject boxes that become invalid after clipping.
            if x2 <= x1 or y2 <= y1:
                self.degenerate_box_count += 1
                continue

            bbox = xyxy_to_tlwh(clipped_box)

            detections.append(
                Detection(
                    frame_id=frame.frame_id,
                    bbox=bbox,
                    class_id=class_id,
                    source_label=source_label,
                    vehicle_type=vehicle_type,
                    confidence=confidence,
                )
            )

        return detections

    def get_publishable_detections(
        self,
        detections: list[Detection],
    ) -> list[Detection]:
        """
        Select detections meeting the higher publishing threshold.

        This does not modify the original detections passed to ByteTrack.
        """
        return [
            detection
            for detection in detections
            if detection.confidence
            >= self.config.publish_confidence_threshold
        ]
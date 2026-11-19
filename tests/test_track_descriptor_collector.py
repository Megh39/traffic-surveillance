import unittest
from types import SimpleNamespace
import numpy as np

from src.tvs.reid.track_descriptor_collector import CollectorConfig, TrackDescriptorCollector


class FakeEmbedder:
    def __init__(self):
        self.boxes = []

    def from_frame_bbox(self, frame_bgr, bbox_xyxy):
        self.boxes.append(bbox_xyxy)
        vector = np.zeros(512, dtype=np.float32)
        vector[0] = 1.0
        return vector


def frame(n):
    return SimpleNamespace(frame_id=n, image=np.zeros((480, 640, 3), dtype=np.uint8))


def observation(n, track=7, confidence=0.9, bbox=(100, 50, 80, 40)):
    return SimpleNamespace(frame_id=n, published_track_id=track, confidence=confidence, bbox=bbox)


class CollectorTests(unittest.TestCase):
    def test_conversion_and_aggregation(self):
        embedder = FakeEmbedder()
        collector = TrackDescriptorCollector(embedder, CollectorConfig(minimum_samples_per_track=2))
        self.assertTrue(collector.add(frame(1), observation(1)))
        self.assertTrue(collector.add(frame(2), observation(2)))
        self.assertEqual(embedder.boxes, [(100.0, 50.0, 180.0, 90.0)] * 2)
        descriptor = collector.finalize(7)
        self.assertEqual(descriptor.shape, (512,))
        self.assertAlmostEqual(float(np.linalg.norm(descriptor)), 1.0, places=6)

    def test_rejects_low_confidence_and_duplicate_frame(self):
        collector = TrackDescriptorCollector(FakeEmbedder())
        self.assertFalse(collector.add(frame(1), observation(1, confidence=0.1)))
        self.assertTrue(collector.add(frame(2), observation(2)))
        self.assertFalse(collector.add(frame(2), observation(2)))
        self.assertEqual(collector.sample_count(7), 1)

    def test_insufficient_samples(self):
        collector = TrackDescriptorCollector(FakeEmbedder())
        collector.add(frame(1), observation(1))
        self.assertIsNone(collector.finalize(7))

    def test_unpublished_track(self):
        collector = TrackDescriptorCollector(FakeEmbedder())
        self.assertFalse(collector.add(frame(1), observation(1, track=None)))


if __name__ == "__main__":
    unittest.main()

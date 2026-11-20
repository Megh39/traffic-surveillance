from __future__ import annotations

import unittest
from contextlib import contextmanager
from uuid import uuid4

import numpy as np

from src.tvs.reid.run_persistence import persist_finalized_track_descriptors


class FakeCursor:
    def __init__(self, connection):
        self.connection = connection
        self.result = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params=None):
        normalized = " ".join(sql.split())
        if normalized.startswith("SELECT vehicle_type"):
            self.result = ("car",)
        elif "INSERT INTO vehicle_reid_descriptors" in normalized:
            self.result = None if self.connection.already_exists else (1,)
            self.connection.already_exists = True
        elif "INSERT INTO persistent_vehicles" in normalized:
            self.result = (77,)
        else:
            self.result = None

    def fetchone(self):
        return self.result


class FakeConnection:
    def __init__(self):
        self.already_exists = False

    def cursor(self):
        return FakeCursor(self)

    @contextmanager
    def transaction(self):
        yield


class RunPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.connection = FakeConnection()
        self.run_id = uuid4()
        self.track_map = {10: 501}
        self.descriptors = {10: np.ones(512, dtype=np.float32)}

    def test_creates_new_persistent_identity_when_no_candidate(self):
        result = persist_finalized_track_descriptors(
            self.connection, self.run_id, self.track_map, self.descriptors,
            candidate_finder=lambda *args, **kwargs: [],
        )
        self.assertEqual(result["persisted_descriptors"], 1)
        self.assertEqual(result["created_new_identities"], 1)
        self.assertEqual(result["candidate_count"], 0)

    def test_records_candidate_without_auto_merging(self):
        candidate = {
            "persistent_vehicle_id": 88,
            "similarity": 0.71,
            "source_run_id": str(uuid4()),
            "source_published_track_id": 4,
        }
        result = persist_finalized_track_descriptors(
            self.connection, self.run_id, self.track_map, self.descriptors,
            candidate_finder=lambda *args, **kwargs: [candidate],
        )
        self.assertEqual(result["persisted_descriptors"], 1)
        self.assertEqual(result["created_new_identities"], 0)
        self.assertEqual(result["candidate_count"], 1)
        self.assertEqual(result["candidates"][0]["candidate_vehicle_id"], 88)

    def test_skips_descriptor_without_database_track_mapping(self):
        result = persist_finalized_track_descriptors(
            self.connection, self.run_id, {}, self.descriptors,
            candidate_finder=lambda *args, **kwargs: [],
        )
        self.assertEqual(result["persisted_descriptors"], 0)
        self.assertEqual(result["skipped_missing_track"], 1)

    def test_rejects_invalid_candidate_limit(self):
        with self.assertRaises(ValueError):
            persist_finalized_track_descriptors(
                self.connection,
                self.run_id,
                self.track_map,
                self.descriptors,
                candidate_limit=0,
            )

    def test_clamps_floating_point_similarity_overshoot(self):
        candidate = {
            "persistent_vehicle_id": 88,
            "similarity": 1.0000001,
            "source_run_id": str(uuid4()),
            "source_published_track_id": 4,
        }
        result = persist_finalized_track_descriptors(
            self.connection, self.run_id, self.track_map, self.descriptors,
            candidate_finder=lambda *args, **kwargs: [candidate],
        )
        self.assertEqual(result["candidates"][0]["similarity"], 1.0)

    def test_rejects_non_finite_similarity(self):
        candidate = {
            "persistent_vehicle_id": 88,
            "similarity": float("nan"),
            "source_run_id": str(uuid4()),
            "source_published_track_id": 4,
        }
        with self.assertRaises(ValueError):
            persist_finalized_track_descriptors(
                self.connection, self.run_id, self.track_map, self.descriptors,
                candidate_finder=lambda *args, **kwargs: [candidate],
            )


if __name__ == "__main__":
    unittest.main()

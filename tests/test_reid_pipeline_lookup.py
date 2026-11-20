from __future__ import annotations

import unittest
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import yaml

from src.tvs.db.track_id_lookup import get_track_id_map


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.params = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, sql, params):
        self.params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.fake_cursor = FakeCursor(rows)

    def cursor(self):
        return self.fake_cursor


class TrackIdLookupIntegrationTests(unittest.TestCase):
    def test_returns_mapping_for_the_requested_run(self):
        run_id = uuid4()
        connection = FakeConnection([(12, 900), (13, 901)])
        self.assertEqual(
            get_track_id_map(connection, run_id),
            {12: 900, 13: 901},
        )
        self.assertEqual(connection.fake_cursor.params, (run_id,))

    def test_empty_run_returns_empty_mapping(self):
        self.assertEqual(get_track_id_map(FakeConnection([]), uuid4()), {})


if __name__ == "__main__":
    unittest.main()

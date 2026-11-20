import unittest
from uuid import uuid4

from src.tvs.db.track_id_lookup import get_track_id_map


class FakeCursor:
    def __init__(self, rows):
        self.rows = rows
        self.executed_sql = None
        self.executed_params = None
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True

    def execute(self, sql, params):
        self.executed_sql = sql
        self.executed_params = params

    def fetchall(self):
        return self.rows


class FakeConnection:
    def __init__(self, rows):
        self.fake_cursor = FakeCursor(rows)
        self.cursor_calls = 0
        self.closed = False

    def cursor(self):
        self.cursor_calls += 1
        return self.fake_cursor

    def close(self):
        self.closed = True


class TrackIdLookupTests(unittest.TestCase):
    def test_returns_published_to_database_id_mapping(self):
        run_id = uuid4()
        connection = FakeConnection([(12, 901), (15, 905), (18, 911)])

        result = get_track_id_map(connection, run_id)

        self.assertEqual(result, {12: 901, 15: 905, 18: 911})
        self.assertEqual(connection.fake_cursor.executed_params, (run_id,))
        self.assertIn("FROM vehicle_tracks", connection.fake_cursor.executed_sql)
        self.assertIn("WHERE run_id = %s", connection.fake_cursor.executed_sql)
        self.assertTrue(connection.fake_cursor.closed)
        self.assertFalse(connection.closed)

    def test_returns_empty_mapping_when_run_has_no_tracks(self):
        connection = FakeConnection([])

        result = get_track_id_map(connection, uuid4())

        self.assertEqual(result, {})

    def test_converts_database_integer_values_to_int(self):
        connection = FakeConnection([("7", "1007")])

        result = get_track_id_map(connection, uuid4())

        self.assertEqual(result, {7: 1007})
        self.assertIs(type(next(iter(result.keys()))), int)
        self.assertIs(type(next(iter(result.values()))), int)


if __name__ == "__main__":
    unittest.main()

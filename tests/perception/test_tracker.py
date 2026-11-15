from pathlib import Path

import numpy as np

from src.tvs.events.detection import Detection
from src.tvs.events.frame import Frame
from src.tvs.perception.tracker import Tracker, TrackerConfig


FPS = 30.0


def make_tracker() -> Tracker:
    config = TrackerConfig(
        track_high_thresh=0.25,
        track_low_thresh=0.05,
        new_track_thresh=0.25,
        match_thresh=0.80,
        fuse_score=True,
        lost_buffer_seconds=1.0,
        minimum_box_area=0,
        confirmation_frames=3,
        minimum_track_points=5,
        minimum_displacement_pixels=10,
    )
    return Tracker(config)


def make_frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        elapsed_seconds=(frame_id - 1) / FPS,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        source_path=Path(f"frame_{frame_id:05d}.jpg"),
        fps_effective=FPS,
    )


def make_detection(
    frame_id: int,
    x: float,
    confidence: float = 0.90,
) -> Detection:
    return Detection(
        frame_id=frame_id,
        bbox=(x, 100.0, 40.0, 30.0),
        class_id=2,
        source_label="car",
        vehicle_type="car",
        confidence=confidence,
    )


def process_frame(
    tracker: Tracker,
    frame_id: int,
    x: float | None,
):
    frame = make_frame(frame_id)
    detections = (
        []
        if x is None
        else [make_detection(frame_id, x)]
    )
    return tracker.update(detections, frame)


def test_linear_movement_keeps_one_published_id():
    tracker = make_tracker()
    observations = []

    for frame_id in range(1, 6):
        x = 100.0 + (frame_id - 1) * 5.0
        observations.extend(
            process_frame(tracker, frame_id, x)
        )

    assert [item.frame_id for item in observations] == [3, 4, 5]
    assert {item.published_track_id for item in observations} == {1}
    assert len({item.internal_track_id for item in observations}) == 1


def test_three_frame_gap_recovers_same_published_id():
    tracker = make_tracker()
    observations = []

    positions = {
        1: 100.0,
        2: 105.0,
        3: 110.0,
        7: 130.0,
        8: 135.0,
        9: 140.0,
    }

    for frame_id in range(1, 10):
        observations.extend(
            process_frame(
                tracker,
                frame_id,
                positions.get(frame_id),
            )
        )

    assert observations
    assert {item.published_track_id for item in observations} == {1}
    assert {item.internal_track_id for item in observations} == {1}
    assert {item.frame_id for item in observations} == {3, 7, 8, 9}


def test_two_frame_flicker_is_never_published():
    tracker = make_tracker()
    observations = []

    for frame_id in range(1, 6):
        x = 100.0 + (frame_id - 1) * 5.0 if frame_id <= 2 else None
        observations.extend(
            process_frame(tracker, frame_id, x)
        )

    assert observations == []


def test_track_after_buffer_expiry_gets_new_published_id():
    tracker = make_tracker()
    observations_by_frame = {}

    for frame_id in range(1, 40):
        if frame_id <= 3:
            x = 100.0 + (frame_id - 1) * 5.0
        elif frame_id <= 34:
            x = None
        else:
            x = 300.0 + (frame_id - 35) * 5.0

        observations_by_frame[frame_id] = process_frame(
            tracker,
            frame_id,
            x,
        )

    assert [item.published_track_id for item in observations_by_frame[3]] == [1]
    assert observations_by_frame[35] == []
    assert observations_by_frame[36] == []

    new_track = observations_by_frame[37]
    assert len(new_track) == 1
    assert new_track[0].published_track_id == 2
    assert new_track[0].internal_track_id == 2

    assert observations_by_frame[38][0].published_track_id == 2
    assert observations_by_frame[39][0].published_track_id == 2


def test_association_statistics_count_new_and_matched_detections():
    tracker = make_tracker()

    process_frame(tracker, 1, 100.0)
    first_stats = tracker.association_stats

    assert first_stats is not None
    assert first_stats.frame_id == 1
    assert first_stats.matched_tracks == 0
    assert first_stats.unmatched_tracks == 0
    assert first_stats.unmatched_detections == 1

    process_frame(tracker, 2, 105.0)
    second_stats = tracker.association_stats

    assert second_stats is not None
    assert second_stats.frame_id == 2
    assert second_stats.matched_tracks == 1
    assert second_stats.unmatched_tracks == 0
    assert second_stats.unmatched_detections == 0
def test_crossing_vehicles_retain_distinct_published_ids():
    tracker = make_tracker()
    observations_by_frame = {}

    for frame_id in range(1, 8):
        vehicle_a_x = 100.0 + (frame_id - 1) * 20.0
        vehicle_b_x = 230.0 - (frame_id - 1) * 20.0

        frame = make_frame(frame_id)
        detections = [
            make_detection(frame_id, vehicle_a_x),
            make_detection(frame_id, vehicle_b_x),
        ]

        observations_by_frame[frame_id] = tracker.update(
            detections,
            frame,
        )

    initial_observations = observations_by_frame[3]
    final_observations = observations_by_frame[7]

    assert len(initial_observations) == 2
    assert len(final_observations) == 2

    initial_a = min(
        initial_observations,
        key=lambda item: abs(item.bbox[0] - 140.0),
    )
    initial_b = min(
        initial_observations,
        key=lambda item: abs(item.bbox[0] - 190.0),
    )

    final_a = min(
        final_observations,
        key=lambda item: abs(item.bbox[0] - 220.0),
    )
    final_b = min(
        final_observations,
        key=lambda item: abs(item.bbox[0] - 110.0),
    )

    assert initial_a.published_track_id != initial_b.published_track_id
    assert final_a.published_track_id == initial_a.published_track_id
    assert final_b.published_track_id == initial_b.published_track_id
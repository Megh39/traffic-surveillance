from pathlib import Path

import numpy as np
import pytest

from tvs.events.frame import Frame
from tvs.perception.tracker import TrackObservation
from tvs.perception.renderer import TrackRenderer


def make_frame(frame_id: int) -> Frame:
    return Frame(
        frame_id=frame_id,
        elapsed_seconds=(frame_id - 1) / 30.0,
        image=np.zeros((720, 1280, 3), dtype=np.uint8),
        source_path=Path(f"frame_{frame_id:05d}.jpg"),
        fps_effective=30.0,
    )


def make_observation(
    frame_id: int,
    track_id: int | None,
    x: float = 100.0,
) -> TrackObservation:
    return TrackObservation(
        frame_id=frame_id,
        internal_track_id=track_id if track_id is not None else 99,
        published_track_id=track_id,
        bbox=(x, 100.0, 40.0, 30.0),
        class_id=2,
        source_label="car",
        vehicle_type="car",
        confidence=0.90,
        detection_index=0,
    )


def test_renderer_draws_bounding_box_and_id():
    renderer = TrackRenderer()
    frame = make_frame(1)
    observation = make_observation(1, 1)

    rendered = renderer.render(frame, [observation])

    assert rendered.shape == frame.image.shape
    assert rendered.dtype == frame.image.dtype
    assert np.any(rendered != frame.image)


def test_renderer_maintains_trajectory_history():
    renderer = TrackRenderer(trail_length=5)

    for frame_id in range(1, 4):
        observation = make_observation(
            frame_id,
            1,
            x=100.0 + (frame_id - 1) * 10.0,
        )
        renderer.render(make_frame(frame_id), [observation])

    trajectory = renderer.trajectories[1]

    assert len(trajectory) == 3
    assert list(trajectory) == [
        (120, 115),
        (130, 115),
        (140, 115),
    ]


def test_renderer_keeps_separate_trajectories_for_vehicles():
    renderer = TrackRenderer()

    observations = [
        make_observation(1, 1, x=100.0),
        make_observation(1, 2, x=300.0),
    ]

    renderer.render(make_frame(1), observations)

    assert set(renderer.trajectories.keys()) == {1, 2}
    assert len(renderer.trajectories[1]) == 1
    assert len(renderer.trajectories[2]) == 1
    assert renderer.trajectories[1] != renderer.trajectories[2]


def test_renderer_ignores_unpublished_tracks():
    renderer = TrackRenderer()
    frame = make_frame(1)
    observation = make_observation(1, None)

    rendered = renderer.render(frame, [observation])

    assert np.array_equal(rendered, frame.image)
    assert renderer.trajectories == {}


def test_renderer_does_not_modify_original_frame():
    renderer = TrackRenderer()
    frame = make_frame(1)
    original = frame.image.copy()

    renderer.render(
        frame,
        [make_observation(1, 1)],
    )

    assert np.array_equal(frame.image, original)


def test_renderer_rejects_observation_from_another_frame():
    renderer = TrackRenderer()

    with pytest.raises(ValueError, match="supplied frame"):
        renderer.render(
            make_frame(1),
            [make_observation(2, 1)],
        )


def test_renderer_limits_trajectory_length():
    renderer = TrackRenderer(trail_length=3)

    for frame_id in range(1, 6):
        renderer.render(
            make_frame(frame_id),
            [
                make_observation(
                    frame_id,
                    1,
                    x=100.0 + frame_id * 5.0,
                )
            ],
        )

    assert len(renderer.trajectories[1]) == 3


def test_renderer_reset_clears_trajectory_history():
    renderer = TrackRenderer()

    renderer.render(
        make_frame(1),
        [make_observation(1, 1)],
    )

    assert 1 in renderer.trajectories

    renderer.reset()

    assert renderer.trajectories == {}

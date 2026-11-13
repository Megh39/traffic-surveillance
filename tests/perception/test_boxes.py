import pytest

from tvs.perception.boxes import (
    tlwh_to_xyxy,
    xyxy_to_tlwh,
)


class TestXYXYToTLWH:

    def test_standard_box(self):
        assert xyxy_to_tlwh((100, 150, 300, 250)) == (
            100.0, 150.0, 200.0, 100.0
        )

    def test_fractional_coordinates(self):
        assert xyxy_to_tlwh((10.5, 20.5, 110.5, 70.5)) == (
            10.5, 20.5, 100.0, 50.0
        )

    def test_negative_top_left_coordinates_are_preserved(self):
        assert xyxy_to_tlwh((-10, -5, 100, 80)) == (
            -10.0, -5.0, 110.0, 85.0
        )

    def test_zero_width_is_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_tlwh((100, 100, 100, 200))

    def test_zero_height_is_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_tlwh((100, 100, 200, 100))

    def test_inverted_coordinates_are_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_tlwh((200, 200, 100, 100))

    def test_invalid_length_is_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_tlwh((10, 20, 30))

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(ValueError):
            xyxy_to_tlwh((10, 20, float("nan"), 100))


class TestTLWHToXYXY:

    def test_standard_box(self):
        assert tlwh_to_xyxy((100, 150, 200, 100)) == (
            100.0, 150.0, 300.0, 250.0
        )

    def test_fractional_coordinates(self):
        assert tlwh_to_xyxy((10.5, 20.5, 100.0, 50.0)) == (
            10.5, 20.5, 110.5, 70.5
        )

    def test_zero_width_is_rejected(self):
        with pytest.raises(ValueError):
            tlwh_to_xyxy((100, 100, 0, 100))

    def test_zero_height_is_rejected(self):
        with pytest.raises(ValueError):
            tlwh_to_xyxy((100, 100, 100, 0))

    def test_invalid_length_is_rejected(self):
        with pytest.raises(ValueError):
            tlwh_to_xyxy((10, 20, 30))

    def test_non_finite_values_are_rejected(self):
        with pytest.raises(ValueError):
            tlwh_to_xyxy((10, 20, float("inf"), 100))


class TestBoundingBoxRoundTrip:

    @pytest.mark.parametrize(
        "box",
        [
            (100, 150, 300, 250),
            (0, 0, 640, 480),
            (10.5, 20.5, 110.5, 70.5),
            (-10, -5, 100, 80),
        ],
    )
    def test_xyxy_tlwh_xyxy_round_trip(self, box):
        assert tlwh_to_xyxy(xyxy_to_tlwh(box)) == tuple(
            float(value) for value in box
        )

    @pytest.mark.parametrize(
        "box",
        [
            (100, 150, 200, 100),
            (0, 0, 640, 480),
            (10.5, 20.5, 100.0, 50.0),
        ],
    )
    def test_tlwh_xyxy_tlwh_round_trip(self, box):
        assert xyxy_to_tlwh(tlwh_to_xyxy(box)) == tuple(
            float(value) for value in box
        )

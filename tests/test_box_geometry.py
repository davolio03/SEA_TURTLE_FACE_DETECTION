"""Tests for bounding-box geometry utilities."""

import unittest

import numpy as np

from turtle_detection.box_geometry import (
    clip_xyxy_to_image,
    intersection_over_union_xyxy,
    normalized_xywh_to_pixel_xyxy,
    pixel_xyxy_to_normalized_xywh,
    valid_normalized_xywh_mask,
    valid_xyxy_mask,
    validate_normalized_xywh,
    validate_xyxy,
)


class BoxGeometryTests(unittest.TestCase):
    def test_normalized_xywh_to_pixel_xyxy(self) -> None:
        result = normalized_xywh_to_pixel_xyxy([0.1, 0.2, 0.4, 0.4], 200, 100)

        np.testing.assert_allclose(result, [20, 20, 100, 60])

    def test_box_conversion_round_trip_for_batch(self) -> None:
        normalized = np.array(
            [
                [0.1, 0.2, 0.4, 0.4],
                [0.0, 0.0, 1.0, 1.0],
            ]
        )

        pixels = normalized_xywh_to_pixel_xyxy(normalized, 512, 384)
        restored = pixel_xyxy_to_normalized_xywh(pixels, 512, 384)

        np.testing.assert_allclose(restored, normalized)

    def test_clip_xyxy_to_image(self) -> None:
        result = clip_xyxy_to_image([-5, 10, 220, 120], 200, 100)

        np.testing.assert_allclose(result, [0, 10, 200, 100])

    def test_normalized_xywh_validity(self) -> None:
        boxes = np.array(
            [
                [0.1, 0.2, 0.4, 0.4],
                [-0.1, 0.2, 0.4, 0.4],
                [0.1, 0.2, 0.0, 0.4],
                [0.8, 0.2, 0.3, 0.4],
                [0.1, np.nan, 0.4, 0.4],
            ]
        )

        np.testing.assert_array_equal(
            valid_normalized_xywh_mask(boxes),
            [True, False, False, False, False],
        )

    def test_xyxy_validity_with_image_bounds(self) -> None:
        boxes = np.array(
            [
                [0, 0, 200, 100],
                [20, 20, 20, 60],
                [-1, 0, 20, 20],
                [0, 0, 201, 100],
            ]
        )

        np.testing.assert_array_equal(
            valid_xyxy_mask(boxes, 200, 100),
            [True, False, False, False],
        )

    def test_validators_reject_invalid_boxes(self) -> None:
        with self.assertRaises(ValueError):
            validate_normalized_xywh([0.9, 0.1, 0.2, 0.2])
        with self.assertRaises(ValueError):
            validate_xyxy([10, 10, 10, 20])

    def test_intersection_over_union_xyxy(self) -> None:
        cases = [
            ([0, 0, 2, 2], [0, 0, 2, 2], 1.0),
            ([0, 0, 1, 1], [1, 1, 2, 2], 0.0),
            ([0, 0, 2, 2], [1, 1, 3, 3], 1 / 7),
        ]
        for first, second, expected in cases:
            with self.subTest(first=first, second=second):
                result = intersection_over_union_xyxy(first, second)
                self.assertAlmostEqual(float(result), expected)

    def test_intersection_over_union_supports_broadcasting(self) -> None:
        boxes = np.array([[0, 0, 2, 2], [0, 0, 1, 1]])

        result = intersection_over_union_xyxy(boxes, [0, 0, 2, 2])

        np.testing.assert_allclose(result, [1.0, 0.25])

    def test_intersection_over_union_rejects_degenerate_boxes(self) -> None:
        with self.assertRaises(ValueError):
            intersection_over_union_xyxy([0, 0, 0, 1], [0, 0, 1, 1])

    def test_conversions_reject_invalid_image_dimensions(self) -> None:
        cases = [(0, 100), (100, -1), (np.nan, 100)]
        for width, height in cases:
            with self.subTest(width=width, height=height):
                with self.assertRaises(ValueError):
                    normalized_xywh_to_pixel_xyxy(
                        [0, 0, 1, 1],
                        width,
                        height,
                    )


if __name__ == "__main__":
    unittest.main()

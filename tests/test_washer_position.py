import math
import unittest
from dataclasses import replace

from src.inspection.position import analyze_washer_position
from src.inspection.service import Detection


def detection(name, x, y=50, *, width=10, height=10):
    return Detection(name, 0.9, (x - width / 2, y - height / 2, x + width / 2, y + height / 2))


def assembly(washers=(70, 150)):
    return [detection("bolt", 10), detection("cap_nut", 210), detection("spacer", 110)] + [
        detection("washer", x) for x in washers
    ]


def transform(items, angle, scale=1, mirror=False):
    c, s = math.cos(angle), math.sin(angle)
    changed = []
    for item in items:
        x1, y1, x2, y2 = item.xyxy
        points = []
        for x, y in ((x1, y1), (x1, y2), (x2, y1), (x2, y2)):
            x = -x if mirror else x
            points.append((scale * (c * x - s * y) + 500, scale * (s * x + c * y) + 500))
        changed.append(replace(item, xyxy=(
            min(x for x, _ in points), min(y for _, y in points),
            max(x for x, _ in points), max(y for _, y in points),
        )))
    return changed


class TestWasherPosition(unittest.TestCase):
    def test_normal_has_one_washer_on_each_side_and_normalized_axis(self):
        result = analyze_washer_position(assembly())
        self.assertEqual((result.status, result.reason), ("resolved", "both_sides_present"))
        self.assertEqual([item.side for item in result.washers], ["before_spacer", "after_spacer"])
        self.assertEqual(result.axis.project((10, 50)), 0)
        self.assertEqual(result.axis.project((210, 50)), 1)
        self.assertEqual(result.spacer_center, 0.5)
        self.assertIsNone(result.missing_defect_type)

    def test_remaining_washer_identifies_the_opposite_missing_side(self):
        for remaining, expected in ((70, "washer_after_spacer_missing"), (150, "washer_before_spacer_missing")):
            with self.subTest(remaining=remaining):
                result = analyze_washer_position(assembly((remaining,)))
                self.assertEqual(result.missing_defect_type, expected)
                self.assertEqual(result.reason, "single_side_missing")

    def test_rotation_reflection_scale_and_translation_do_not_swap_sides(self):
        for angle in (0, math.pi / 4, math.pi / 2, math.pi, -math.pi / 3):
            for scale in (0.2, 1, 4):
                for mirror in (False, True):
                    with self.subTest(angle=angle, scale=scale, mirror=mirror):
                        result = analyze_washer_position(transform(assembly((70,)), angle, scale, mirror))
                        self.assertEqual(result.missing_defect_type, "washer_after_spacer_missing")
                        self.assertAlmostEqual(result.washers[0].center, 0.3)

    def test_missing_or_duplicate_anchors_never_select_an_arbitrary_detection(self):
        items = assembly((70,))
        for name in ("bolt", "cap_nut", "spacer"):
            for changed in ([item for item in items if item.class_name != name], items + [next(item for item in items if item.class_name == name)]):
                with self.subTest(name=name, count=len(changed)):
                    result = analyze_washer_position(changed)
                    self.assertEqual(result.reason, "anchor_count")
                    self.assertIsNone(result.missing_defect_type)

    def test_absent_or_excess_washers_are_not_interpreted_as_single_side_missing(self):
        for positions in ((), (60, 70, 150)):
            result = analyze_washer_position(assembly(positions))
            self.assertEqual(result.reason, "washer_count")
            self.assertIsNone(result.missing_defect_type)

    def test_missing_invalid_or_zero_area_coordinates_are_unresolved(self):
        for box in (None, (1, 2, 3), (float("nan"), 1, 2, 3), (1, 1, float("inf"), 3), (3, 2, 1, 4), (1, 1, 1, 2)):
            items = assembly((70,))
            items[-1] = replace(items[-1], xyxy=box)
            result = analyze_washer_position(items)
            self.assertEqual(result.status, "unresolved")
            self.assertIsNone(result.missing_defect_type)

    def test_coincident_or_overlapping_anchor_boxes_are_unresolved(self):
        items = assembly((70,))
        items[1] = detection("cap_nut", 10)
        self.assertEqual(analyze_washer_position(items).reason, "axis_unavailable")
        items[1] = detection("cap_nut", 15)
        self.assertEqual(analyze_washer_position(items).reason, "anchors_overlap")

    def test_spacer_must_be_between_anchors_and_cross_the_axis(self):
        for spacer in (detection("spacer", 240), detection("spacer", 110, 100)):
            items = assembly((70,))
            items[2] = spacer
            self.assertEqual(analyze_washer_position(items).reason, "spacer_outside_axis")

    def test_box_background_crossing_spacer_center_does_not_override_center_order(self):
        for washer in (detection("washer", 105, width=20), detection("washer", 105, width=10)):
            items = assembly(()) + [washer]
            result = analyze_washer_position(items)
            self.assertEqual(result.washers[0].side, "before_spacer")
            self.assertEqual(result.missing_defect_type, "washer_after_spacer_missing")

    def test_coincident_projected_centers_remain_ambiguous(self):
        for angle in (0, math.pi / 4, math.pi / 2):
            result = analyze_washer_position(transform(assembly((110,)), angle))
            self.assertEqual(result.washers[0].side, "ambiguous")
            self.assertIsNone(result.missing_defect_type)

    def test_diagonal_anchor_background_overlap_is_not_physical_axis_overlap(self):
        # 枠の角の射影は重なるが、軸が通る区間は十分に離れている。
        items = [detection("bolt", 0, 0, width=20, height=220),
                 detection("cap_nut", 100, 100, width=20, height=220),
                 detection("spacer", 50, 50), detection("washer", 25, 25)]
        self.assertEqual(analyze_washer_position(items).missing_defect_type, "washer_after_spacer_missing")

    def test_washer_outside_axis_or_beyond_anchors_is_not_used(self):
        for washer in (detection("washer", 70, 90), detection("washer", -20)):
            result = analyze_washer_position(assembly(()) + [washer])
            self.assertEqual(result.washers[0].side, "outside_axis")
            self.assertIsNone(result.missing_defect_type)

    def test_two_washers_on_same_side_do_not_invent_a_new_quantity_defect(self):
        result = analyze_washer_position(assembly((60, 80)))
        self.assertEqual(result.reason, "washers_on_same_side")
        self.assertIsNone(result.missing_defect_type)

    def test_detection_order_and_unknown_classes_do_not_change_geometry(self):
        items = assembly()
        original = list(items)
        result = analyze_washer_position(iter(reversed(items + [Detection("unrelated", 0.9)])))
        self.assertEqual(result.reason, "both_sides_present")
        self.assertEqual([item.center for item in result.washers], [0.3, 0.7])
        self.assertEqual(items, original)


if __name__ == "__main__":
    unittest.main()

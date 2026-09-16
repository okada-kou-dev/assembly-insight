"""数量・判定・欠品側の評価。元の集計定義を保持する。"""

# Copyright (C) 2026 Okada Kou. SPDX-License-Identifier: AGPL-3.0-only
from collections import Counter
from dataclasses import asdict
from training.data import CLASSES, DIRECTIONS, STATE_COUNTS, STATE_DEFECTS
SIDE_STATES = ("washer_front_missing", "washer_rear_missing")


def summarize(records):
    """実測件数を分母付きで集計する。精度の合格率は設定しない。"""
    ids = set()
    for record in records:
        if record["id"] in ids:
            raise ValueError("duplicate evaluation id")
        ids.add(record["id"])
        if record["direction"] not in DIRECTIONS or record["state"] not in STATE_COUNTS:
            raise ValueError("unknown direction/state")
        counts = record["counts"]
        if set(counts) != set(CLASSES) or any(type(n) is not int or n < 0 for n in counts.values()):
            raise ValueError("counts must contain four non-negative integers")
        if record["inspection_result"] not in ("OK", "NG"):
            raise ValueError("inspection errors must not be converted into product NG")

    def group(rows):
        totals = {key: 0 for key in ("images", "quantity_matches", "ok_ng_matches", "decision_matches",
                                     "false_ok", "false_ng", "side_eligible", "side_resolved", "side_correct",
                                     "wrong_side", "side_unresolved")}
        confusion, reasons = Counter(), Counter()
        for row in rows:
            state = row["state"]
            expected_counts = dict(zip(CLASSES, STATE_COUNTS[state]))
            expected_defect = STATE_DEFECTS[state]
            expected_result = "OK" if state == "normal" else "NG"
            totals["images"] += 1
            totals["quantity_matches"] += row["counts"] == expected_counts
            totals["ok_ng_matches"] += row["inspection_result"] == expected_result
            totals["decision_matches"] += (row["inspection_result"] == expected_result and row["defect_type"] == expected_defect)
            totals["false_ok"] += expected_result == "NG" and row["inspection_result"] == "OK"
            totals["false_ng"] += expected_result == "OK" and row["inspection_result"] == "NG"
            position = row["position"]
            reasons[position.get("reason") or "unavailable"] += 1
            confusion[(state, row["inspection_result"], row["defect_type"])] += 1
            if state in SIDE_STATES:
                totals["side_eligible"] += 1
                resolved = position.get("status") == "resolved" and position.get("missing_defect_type") in {
                    STATE_DEFECTS[s] for s in SIDE_STATES
                }
                totals["side_resolved"] += resolved
                totals["side_correct"] += resolved and position["missing_defect_type"] == expected_defect
                totals["wrong_side"] += resolved and position["missing_defect_type"] != expected_defect
                totals["side_unresolved"] += not resolved
        totals["rates"] = {
            key: {"numerator": totals[key], "denominator": totals[denominator],
                  "percent": round(totals[key] * 100 / totals[denominator], 1) if totals[denominator] else None}
            for key, denominator in (("quantity_matches", "images"), ("ok_ng_matches", "images"),
                                     ("decision_matches", "images"), ("side_correct", "side_eligible"))
        }
        totals["position_reasons"] = dict(reasons)
        totals["confusion"] = [{"expected_state": state, "actual_result": result, "actual_defect": defect, "count": count}
                               for (state, result, defect), count in confusion.items()]
        return totals

    return {"overall": group(records),
            "by_direction": {direction: group([r for r in records if r["direction"] == direction]) for direction in DIRECTIONS},
            "by_direction_state": [{"direction": direction, "state": state,
                                    **group([r for r in records if r["direction"] == direction and r["state"] == state])}
                                   for direction in DIRECTIONS for state in STATE_COUNTS],
            "acceptance_threshold": None,
            "notes": "side metrics apply only to single-washer-missing states; zero denominators are unavailable"}


def serialize_inspection(sample, inspection, image_sha256, seconds):
    position = inspection.washer_position
    return {"id": sample["id"], "direction": sample["direction"], "state": sample["state"], "split": sample["split"],
            "image": sample["image"], "image_sha256": image_sha256,
            "counts": dict(inspection.counts), "inspection_result": inspection.decision.inspection_result,
            "defect_type": inspection.decision.defect_type,
            "position": asdict(position) if position is not None else {"status": "unavailable", "reason": None},
            "detections": [asdict(detection) for detection in inspection.detections],
            "inspection_seconds": seconds}


def check_acceptance(record):
    expected_counts = dict(zip(CLASSES, STATE_COUNTS[record['state']]))
    expected_result = 'OK' if record['state'] == 'normal' else 'NG'
    record['checks'] = {'counts': record['counts'] == expected_counts,
                        'ok_ng': record['inspection_result'] == expected_result,
                        'defect_type': record['defect_type'] == STATE_DEFECTS[record['state']]}
    record['all_matches'] = all(record['checks'].values())
    return record

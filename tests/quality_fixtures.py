"""時間帯・日別推移・欠品傾向を検証する合成データ。実画像の判定には使わない。"""

from datetime import date, datetime, time


def build_dummy_records() -> list[dict]:
    hours = (8, 9, 10, 11, 13, 14, 15, 17, 19, 21)
    spacer, cap, washer = "spacer_missing", "cap_nut_missing", "washer_missing"
    scenarios = (
        (None, None, spacer, None, None, cap, None, None, washer, None),
        (None, spacer, None, None, None, cap, None, None, washer, washer),
        (None, None, None, cap, None, washer, None, washer, washer, washer),
        (None, cap, None, washer, cap, washer, None, washer, washer, washer),
        (None, None, None, None, cap, None, None, washer, washer, None),
    )
    records = []
    for day, defects in enumerate(scenarios, start=1):
        for hour, defect in zip(hours, defects, strict=True):
            records.append({
                "image_path": f"fixtures/sample_{len(records) + 1:03d}.jpg",
                "captured_at": datetime.combine(date(2026, 9, day), time(hour)),
                "inspection_result": "NG" if defect is not None else "OK",
                "defect_type": defect,
                "confidence": None,
            })
    return records

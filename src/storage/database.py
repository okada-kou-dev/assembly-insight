import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path


def initialize_database(db_path: str | Path) -> None:
    db_path = Path(db_path)

    db_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with closing(sqlite3.connect(db_path)) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS inspections (
                image_id INTEGER PRIMARY KEY AUTOINCREMENT,
                image_path TEXT NOT NULL,
                captured_at TEXT NOT NULL,
                inspection_result TEXT NOT NULL
                    CHECK (inspection_result IN ('OK', 'NG')),
                defect_type TEXT,
                confidence REAL
            )
            """
        )

        connection.commit()


def save_inspection(
    db_path: str | Path,
    image_path: str | Path,
    captured_at: datetime,
    inspection_result: str,
    defect_type: str | None,
    confidence: float | None,
) -> int:
    initialize_database(db_path)

    with closing(sqlite3.connect(db_path)) as connection:
        cursor = connection.execute(
            """
            INSERT INTO inspections (
                image_path,
                captured_at,
                inspection_result,
                defect_type,
                confidence
            )
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                str(image_path),
                captured_at.isoformat(timespec="seconds"),
                inspection_result,
                defect_type,
                confidence,
            ),
        )

        connection.commit()

        image_id = cursor.lastrowid

    if image_id is None:
        raise RuntimeError("Failed to create inspection record")

    return int(image_id)


def list_inspections(
    db_path: str | Path,
) -> list[dict]:
    initialize_database(db_path)

    with closing(sqlite3.connect(db_path)) as connection:
        connection.row_factory = sqlite3.Row

        rows = connection.execute(
            """
            SELECT
                image_id,
                image_path,
                captured_at,
                inspection_result,
                defect_type,
                confidence
            FROM inspections
            ORDER BY captured_at ASC, image_id ASC
            """
        ).fetchall()

    return [dict(row) for row in rows]

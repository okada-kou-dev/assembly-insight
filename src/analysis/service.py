from collections.abc import Callable
from pathlib import Path
from typing import Any

from src.analysis.comment_summary import build_quality_summary
from src.analysis.statistics import confidence_rows, residual_analysis
from src.analysis.evidence import build_quality_evidence
from src.analysis.quality import (
    build_quality_trend_data,
    calculate_daily_quality_metrics,
    calculate_quality_metrics,
    calculate_time_band_quality_metrics,
)
from src.llm.client import generate_free_quality_comment
from src.storage.database import list_inspections

ANALYSIS_SCHEMA_VERSION = 2
SMALL_SAMPLE_THRESHOLD = 30


QualityCommentGenerator = Callable[
    [dict[str, Any]],
    str,
]


def build_llm_context(
    *,
    total_inspections: int,
    data_type: str,
    evidence: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    ローカルLLMへ渡す品質分析コンテキストを生成する。

    データ種別・標本数の注意・Python計算済みevidenceを渡す。
    """
    if not data_type:
        raise ValueError("data_type must not be empty.")

    context: dict[str, Any] = {
        "data_type": data_type,
        "sample_size_limited": (total_inspections <= SMALL_SAMPLE_THRESHOLD),
    }

    if evidence is not None:
        context["evidence"] = evidence

    return context


def analyze_quality(
    db_path: str | Path,
    *,
    data_type: str = "test_data",
    comment_generator: QualityCommentGenerator | None = None,
    generate_comment: bool = True,
) -> dict[str, Any]:
    """
    SQLite履歴から品質分析とローカルLLMコメントをまとめて生成する。

    品質数値はPythonで決定論的に計算し、
    自由文生成には分母付きのPython集計を渡し、Evidenceも結果に保持する。
    """
    return analyze_records(
        list_inspections(db_path), source_db_path=str(db_path), data_type=data_type,
        comment_generator=comment_generator,
        generate_comment=generate_comment,
    )


def analyze_records(
    inspections: list[dict], *, source_db_path: str, data_type: str = "test_data",
    comment_generator: QualityCommentGenerator | None = None,
    generate_comment: bool = True,
) -> dict[str, Any]:
    """取得済みの履歴を分析する。プレビューも同じ集計経路を使用する。"""

    overall = calculate_quality_metrics(inspections)

    daily = calculate_daily_quality_metrics(inspections)

    time_bands = calculate_time_band_quality_metrics(inspections)
    intervals = confidence_rows(time_bands)
    residuals = residual_analysis(time_bands)

    trend = build_quality_trend_data(inspections)


    evidence = build_quality_evidence(inspections)

    llm_context = build_llm_context(
        total_inspections=overall["total_inspections"],
        data_type=data_type,
        evidence=evidence,
    )
    llm_context["quality_summary"] = build_quality_summary(
        inspections,
        data_type=data_type,
    )

    if not generate_comment or overall["total_inspections"] == 0:
        llm_comment = None

    else:
        generator = (
            comment_generator
            if comment_generator is not None
            else generate_free_quality_comment
        )

        llm_comment = generator(llm_context)

    return {
        "source_db_path": source_db_path,
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "confidence_intervals": intervals,
        "residual_analysis": residuals,
        "overall": overall,
        "daily": daily,
        "time_bands": time_bands,
        "trend": trend,
        "llm_context": llm_context,
        "llm_comment": llm_comment,
    }

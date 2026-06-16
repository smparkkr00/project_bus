from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import empty_frame, quantile_or_nan


COMMUTE_HOURS = [7, 8, 9]
EVENING_HOURS = [17, 18, 19, 20]
COMMUTE_FOCUS_THRESHOLD = 25.0
EVENING_FOCUS_THRESHOLD = 25.0


STOP_SUMMARY_COLUMNS = [
    "stop_name",
    "district",
    "total_boardings",
    "total_alightings",
    "total_users",
    "route_count",
    "boardings_per_route",
    "commute_boardings",
    "evening_boardings",
    "commute_focus_pct",
    "evening_focus_pct",
    "peak_hour",
    "boarding_alighting_gap",
    "imbalance_index",
    "stop_type",
    "latitude",
    "longitude",
]


def _peak_hour(hourly_summary: pd.DataFrame) -> pd.DataFrame:
    """정류소별 승차 인원이 가장 많은 시간대를 찾는다."""
    if hourly_summary.empty:
        return empty_frame(["stop_name", "district", "peak_hour"])
    grouped = (
        hourly_summary.groupby(["stop_name", "district", "hour"], dropna=False)["boardings"]
        .sum()
        .reset_index()
        .sort_values(["stop_name", "district", "boardings"], ascending=[True, True, False])
    )
    peak = grouped.drop_duplicates(["stop_name", "district"])
    peak["peak_hour"] = peak["hour"].astype("Int64").astype("string") + "시"
    return peak[["stop_name", "district", "peak_hour"]]


def classify_stop(row: pd.Series, low_use_cutoff: float) -> str:
    """출근/퇴근 집중도와 이용량에 따라 정류소 유형을 나눈다."""
    total_boardings = row.get("total_boardings")
    commute_pct = row.get("commute_focus_pct")
    evening_pct = row.get("evening_focus_pct")
    if pd.notna(total_boardings) and pd.notna(low_use_cutoff) and total_boardings <= low_use_cutoff:
        return "저이용형"
    if pd.notna(commute_pct) and pd.notna(evening_pct):
        commute_high = commute_pct >= COMMUTE_FOCUS_THRESHOLD
        evening_high = evening_pct >= EVENING_FOCUS_THRESHOLD
        if commute_high and evening_high:
            return "출퇴근형"
        if commute_high and commute_pct > evening_pct:
            return "출근형"
        if evening_high and evening_pct > commute_pct:
            return "퇴근형"
    return "생활형"


def build_stop_summary(monthly_summary: pd.DataFrame, hourly_summary: pd.DataFrame) -> pd.DataFrame:
    """월별 이용량과 시간대별 이용량을 합쳐 정류소 요약 지표를 만든다."""
    if monthly_summary.empty:
        return empty_frame(STOP_SUMMARY_COLUMNS)

    monthly_group = (
        monthly_summary.groupby(["stop_name", "district"], dropna=False)
        .agg(
            total_boardings=("boardings", "sum"),
            total_alightings=("alightings", "sum"),
            total_users=("total_users", "sum"),
            route_count=("route_count", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
        )
        .reset_index()
    )
    monthly_group["boardings_per_route"] = monthly_group["total_boardings"] / monthly_group["route_count"].where(
        monthly_group["route_count"] > 0
    )
    monthly_group["boarding_alighting_gap"] = monthly_group["total_boardings"] - monthly_group["total_alightings"]
    monthly_group["imbalance_index"] = (
        (monthly_group["total_boardings"] - monthly_group["total_alightings"]).abs()
        / (monthly_group["total_boardings"] + monthly_group["total_alightings"]).replace(0, np.nan)
    )

    if hourly_summary.empty:
        hourly_features = empty_frame(
            ["stop_name", "district", "hourly_boardings", "commute_boardings", "evening_boardings"]
        )
        peak = empty_frame(["stop_name", "district", "peak_hour"])
    else:
        hourly_features = (
            hourly_summary.assign(
                commute_part=lambda df: df["boardings"].where(df["hour"].isin(COMMUTE_HOURS), 0),
                evening_part=lambda df: df["boardings"].where(df["hour"].isin(EVENING_HOURS), 0),
            )
            .groupby(["stop_name", "district"], dropna=False)
            .agg(
                hourly_boardings=("boardings", "sum"),
                commute_boardings=("commute_part", "sum"),
                evening_boardings=("evening_part", "sum"),
            )
            .reset_index()
        )
        peak = _peak_hour(hourly_summary)

    result = monthly_group.merge(hourly_features, on=["stop_name", "district"], how="left")
    result = result.merge(peak, on=["stop_name", "district"], how="left")
    result["commute_focus_pct"] = result["commute_boardings"] / result["hourly_boardings"].replace(0, np.nan) * 100
    result["evening_focus_pct"] = result["evening_boardings"] / result["hourly_boardings"].replace(0, np.nan) * 100
    low_use_cutoff = quantile_or_nan(result["total_boardings"], 0.25)
    result["stop_type"] = result.apply(lambda row: classify_stop(row, low_use_cutoff), axis=1)

    for col in STOP_SUMMARY_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    return result[STOP_SUMMARY_COLUMNS].sort_values("total_boardings", ascending=False)


def attach_stop_type(summary: pd.DataFrame, detail: pd.DataFrame) -> pd.DataFrame:
    """월별/시간대별 자료에도 정류소 유형을 붙여 필터링할 수 있게 한다."""
    if detail.empty or summary.empty:
        return detail
    columns = ["stop_name", "district", "stop_type"]
    return detail.merge(summary[columns], on=["stop_name", "district"], how="left")


def add_yoy_metrics(monthly_summary: pd.DataFrame) -> pd.DataFrame:
    """전년 동월 대비 승차 인원 증감률을 계산한다."""
    if monthly_summary.empty:
        return monthly_summary

    base = (
        monthly_summary.groupby(["stop_name", "year", "month"], dropna=False)
        .agg(boardings=("boardings", "sum"))
        .reset_index()
    )
    prior = base.rename(columns={"boardings": "previous_year_boardings"}).copy()
    prior["year"] = prior["year"] + 1
    merged = monthly_summary.merge(prior, on=["stop_name", "year", "month"], how="left")
    merged["yoy_change"] = merged["boardings"] - merged["previous_year_boardings"]
    merged["yoy_rate"] = merged["yoy_change"] / merged["previous_year_boardings"].replace(0, np.nan) * 100
    return merged


def build_imbalance_candidates(
    stop_summary: pd.DataFrame,
    demand_quantile: float = 0.75,
    route_quantile: float = 0.50,
    load_quantile: float = 0.90,
) -> pd.DataFrame:
    """수요는 높고 노선 공급은 상대적으로 적은 추가 검토 후보를 추린다."""
    if stop_summary.empty:
        return empty_frame(STOP_SUMMARY_COLUMNS)

    demand_cutoff = quantile_or_nan(stop_summary["total_boardings"], demand_quantile)
    route_cutoff = quantile_or_nan(stop_summary["route_count"], route_quantile)
    load_cutoff = quantile_or_nan(stop_summary["boardings_per_route"], load_quantile)

    candidates = stop_summary[
        (stop_summary["total_boardings"] >= demand_cutoff)
        & (stop_summary["route_count"] <= route_cutoff)
        & (stop_summary["boardings_per_route"] >= load_cutoff)
    ].copy()
    return candidates.sort_values("boardings_per_route", ascending=False)


def build_data_check_report(file_reports: list[dict], monthly: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """전처리 과정에서 확인한 파일별 상태를 표로 정리한다."""
    rows = list(file_reports)
    rows.append(
        {
            "source_file": "processed/monthly_summary.csv",
            "dataset_type": "processed",
            "encoding": "utf-8-sig",
            "rows": len(monthly),
            "columns": len(monthly.columns),
            "missing_cells": int(monthly.isna().sum().sum()) if not monthly.empty else 0,
            "duplicated_rows": int(monthly.duplicated().sum()) if not monthly.empty else 0,
            "status": "created",
        }
    )
    rows.append(
        {
            "source_file": "processed/hourly_summary.csv",
            "dataset_type": "processed",
            "encoding": "utf-8-sig",
            "rows": len(hourly),
            "columns": len(hourly.columns),
            "missing_cells": int(hourly.isna().sum().sum()) if not hourly.empty else 0,
            "duplicated_rows": int(hourly.duplicated().sum()) if not hourly.empty else 0,
            "status": "created",
        }
    )
    return pd.DataFrame(rows)


def create_analysis_summary(
    stop_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
    hourly_summary: pd.DataFrame,
    weather_monthly: pd.DataFrame,
) -> str:
    """발표 준비용 요약 Markdown을 만든다."""
    top_stop = "확인 불가"
    if not stop_summary.empty:
        first = stop_summary.sort_values("total_boardings", ascending=False).iloc[0]
        top_stop = f"{first['stop_name']} ({int(first['total_boardings']):,}명)"

    months = "없음"
    if not monthly_summary.empty:
        months = f"{monthly_summary['month_label'].min()} ~ {monthly_summary['month_label'].max()}"

    return "\n".join(
        [
            "# 대구 시내버스 정류소 이용 수요 및 노선 공급 분석 요약",
            "",
            f"- 분석 월 범위: {months}",
            f"- 월별 이용량 행 수: {len(monthly_summary):,}",
            f"- 시간대별 이용량 행 수: {len(hourly_summary):,}",
            f"- 분석 정류소 수: {stop_summary['stop_name'].nunique() if not stop_summary.empty else 0:,}",
            f"- 전체 승차 인원이 가장 많은 정류소: {top_stop}",
            f"- 날씨 월별 자료 행 수: {len(weather_monthly):,}",
            "",
            "## 해석 시 유의사항",
            "",
            "- 분석 결과는 노선 부족을 확정하는 것이 아니라 추가 검토 후보를 찾는 용도입니다.",
            "- 경유 노선 수는 실제 배차 횟수나 차량 규모를 뜻하지 않습니다.",
            "- 하차 태그 누락, 현금 승차 제외, 계절성, 행사, 노선 개편 등 외부 요인이 이용량에 영향을 줄 수 있습니다.",
            "- 월별 날씨와 이용량의 관계는 상관 경향이며 인과관계로 표현하지 않습니다.",
        ]
    )


from __future__ import annotations

import numpy as np
import pandas as pd

from .column_mapper import WEATHER_CANDIDATES, find_column
from .utils import add_year_month_columns, empty_frame, parse_year_month, safe_to_number


WEATHER_COLUMNS = [
    "year_month",
    "year",
    "month",
    "month_label",
    "avg_temp",
    "max_temp",
    "min_temp",
    "rainfall",
    "humidity",
    "wind",
    "sunshine",
    "snow",
]

WEATHER_LABELS = {
    "avg_temp": "평균기온",
    "max_temp": "평균최고기온",
    "min_temp": "평균최저기온",
    "rainfall": "월합강수량",
    "humidity": "평균상대습도",
    "wind": "평균풍속",
    "sunshine": "합계 일조시간",
    "snow": "최심적설",
}


def standardize_weather(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """일별 또는 월별 날씨 자료를 월별 지표로 집계한다."""
    columns = list(df.columns)
    date_col = find_column(columns, WEATHER_CANDIDATES["date"])
    if not date_col:
        return empty_frame(WEATHER_COLUMNS), empty_frame(["standard_column", "original_column"])

    weather = pd.DataFrame()
    weather["year_month"] = parse_year_month(df[date_col])
    mapping_rows = [{"standard_column": "year_month", "original_column": date_col}]

    for standard_col, candidates in WEATHER_CANDIDATES.items():
        if standard_col == "date":
            continue
        original_col = find_column(columns, candidates, contains=True)
        if original_col:
            weather[standard_col] = safe_to_number(df[original_col])
            mapping_rows.append({"standard_column": standard_col, "original_column": original_col})

    if weather.drop(columns=["year_month"], errors="ignore").empty:
        return empty_frame(WEATHER_COLUMNS), pd.DataFrame(mapping_rows)

    agg_dict = {}
    for col in weather.columns:
        if col == "year_month":
            continue
        if col in {"rainfall", "sunshine"}:
            agg_dict[col] = "sum"
        elif col == "snow":
            agg_dict[col] = "max"
        else:
            agg_dict[col] = "mean"

    monthly = weather.dropna(subset=["year_month"]).groupby("year_month", dropna=False).agg(agg_dict).reset_index()
    monthly = add_year_month_columns(monthly)
    for col in WEATHER_COLUMNS:
        if col not in monthly.columns:
            monthly[col] = pd.NA
    return monthly[WEATHER_COLUMNS], pd.DataFrame(mapping_rows)


def merge_bus_weather(monthly_summary: pd.DataFrame, weather_monthly: pd.DataFrame) -> pd.DataFrame:
    """정류소별 행에 날씨를 반복해서 붙이지 않고, 월별 전체 이용량과 날씨를 결합한다."""
    if monthly_summary.empty or weather_monthly.empty:
        return empty_frame(["year_month", "year", "month", "month_label", "boardings", "alightings", "total_users"])

    bus_monthly = (
        monthly_summary.groupby(["year_month", "year", "month", "month_label"], dropna=False)
        .agg(
            boardings=("boardings", "sum"),
            alightings=("alightings", "sum"),
            total_users=("total_users", "sum"),
            stop_count=("stop_name", "nunique"),
        )
        .reset_index()
    )
    merged = bus_monthly.merge(weather_monthly, on=["year_month", "year", "month", "month_label"], how="inner")
    return merged.sort_values("year_month")


def interpret_correlation(value: float | None) -> str:
    """상관계수의 절댓값을 쉬운 말로 해석한다."""
    if value is None or pd.isna(value):
        return "계산 불가"
    strength = abs(float(value))
    if strength < 0.2:
        return "약함"
    if strength < 0.4:
        return "다소 약함"
    if strength < 0.6:
        return "보통"
    if strength < 0.8:
        return "강함"
    return "매우 강함"


def spearman_without_scipy(left: pd.Series, right: pd.Series) -> float:
    """scipy가 없어도 순위값의 Pearson 상관으로 Spearman 상관을 계산한다."""
    ranked = pd.DataFrame({"left": left, "right": right}).dropna().rank()
    if len(ranked) < 3:
        return np.nan
    return ranked["left"].corr(ranked["right"], method="pearson")


def calculate_weather_correlations(bus_weather: pd.DataFrame) -> pd.DataFrame:
    """월별 버스 이용량과 날씨 지표의 Pearson, Spearman 상관계수를 계산한다."""
    rows = []
    if bus_weather.empty or "boardings" not in bus_weather.columns:
        return empty_frame(["weather_variable", "weather_label", "observations", "pearson", "spearman", "interpretation"])

    for variable, label in WEATHER_LABELS.items():
        if variable not in bus_weather.columns:
            continue
        subset = bus_weather[["boardings", variable]].dropna()
        if len(subset) < 3:
            pearson = np.nan
            spearman = np.nan
        else:
            pearson = subset["boardings"].corr(subset[variable], method="pearson")
            spearman = spearman_without_scipy(subset["boardings"], subset[variable])
        rows.append(
            {
                "weather_variable": variable,
                "weather_label": label,
                "observations": len(subset),
                "pearson": pearson,
                "spearman": spearman,
                "interpretation": interpret_correlation(pearson),
            }
        )
    return pd.DataFrame(rows)

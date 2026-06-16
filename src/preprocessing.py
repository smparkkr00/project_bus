from __future__ import annotations

import pandas as pd

from .column_mapper import (
    ALIGHTING_CANDIDATES,
    BOARDING_CANDIDATES,
    DATE_CANDIDATES,
    DISTRICT_CANDIDATES,
    KIND_CANDIDATES,
    LATITUDE_CANDIDATES,
    LONGITUDE_CANDIDATES,
    ROUTE_COUNT_CANDIDATES,
    STOP_ID_CANDIDATES,
    STOP_NAME_CANDIDATES,
    find_column,
    hour_columns,
)
from .utils import (
    add_year_month_columns,
    clean_stop_id,
    empty_frame,
    mode_or_first,
    parse_year_month,
    safe_to_number,
)


MONTHLY_COLUMNS = [
    "year_month",
    "year",
    "month",
    "month_label",
    "stop_name",
    "district",
    "boardings",
    "alightings",
    "total_users",
    "route_count",
    "boardings_per_route",
    "latitude",
    "longitude",
]

HOURLY_COLUMNS = [
    "year_month",
    "year",
    "month",
    "month_label",
    "stop_id",
    "stop_name",
    "district",
    "hour",
    "boardings",
    "alightings",
    "total_users",
    "route_count",
    "latitude",
    "longitude",
]

LOCATION_COLUMNS = ["stop_name", "district", "latitude", "longitude", "route_count", "routes"]


def standardize_location(df: pd.DataFrame) -> pd.DataFrame:
    """정류소 위치정보 파일을 표준 컬럼으로 바꾼다."""
    columns = list(df.columns)
    stop_name_col = find_column(columns, STOP_NAME_CANDIDATES)
    district_col = find_column(columns, DISTRICT_CANDIDATES)
    lat_col = find_column(columns, LATITUDE_CANDIDATES)
    lon_col = find_column(columns, LONGITUDE_CANDIDATES)
    route_count_col = find_column(columns, ROUTE_COUNT_CANDIDATES)
    routes_col = find_column(columns, ["경유노선", "노선", "routes"], contains=True)

    if not stop_name_col:
        return empty_frame(LOCATION_COLUMNS)

    result = pd.DataFrame()
    result["stop_name"] = df[stop_name_col].astype("string").str.strip()
    result["district"] = df[district_col].astype("string").str.strip() if district_col else pd.NA
    result["latitude"] = safe_to_number(df[lat_col]) if lat_col else pd.NA
    result["longitude"] = safe_to_number(df[lon_col]) if lon_col else pd.NA
    result["route_count"] = safe_to_number(df[route_count_col]) if route_count_col else pd.NA
    result["routes"] = df[routes_col].astype("string").str.strip() if routes_col else pd.NA

    result = result.dropna(subset=["stop_name"]).drop_duplicates()
    result = result[
        result["latitude"].between(35.0, 37.5, inclusive="both")
        & result["longitude"].between(127.0, 130.0, inclusive="both")
    ]
    result = result[result["route_count"].isna() | (result["route_count"] >= 0)]
    return result[LOCATION_COLUMNS]


def make_location_lookup(location: pd.DataFrame) -> pd.DataFrame:
    """정류소명이 중복될 수 있어 발표용 분석에 쓸 대표 위치를 만든다."""
    if location.empty:
        return empty_frame(LOCATION_COLUMNS)
    grouped = (
        location.groupby("stop_name", dropna=False)
        .agg(
            district=("district", mode_or_first),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            route_count=("route_count", "max"),
            routes=("routes", lambda values: ", ".join(sorted({str(v) for v in values.dropna() if str(v)}))[:500]),
        )
        .reset_index()
    )
    return grouped[LOCATION_COLUMNS]


def merge_location(df: pd.DataFrame, location_lookup: pd.DataFrame) -> pd.DataFrame:
    """이용량 데이터에 구군, 좌표, 경유노선수를 붙인다."""
    if df.empty:
        return df
    result = df.copy()
    if location_lookup.empty:
        for col in ["district", "latitude", "longitude", "route_count"]:
            if col not in result.columns:
                result[col] = pd.NA
        return result

    location_cols = ["stop_name", "district", "latitude", "longitude", "route_count"]
    merged = result.merge(location_lookup[location_cols], on="stop_name", how="left", suffixes=("", "_loc"))
    for col in ["district", "latitude", "longitude", "route_count"]:
        loc_col = f"{col}_loc"
        if loc_col in merged.columns:
            if col in result.columns:
                merged[col] = merged[col].combine_first(merged[loc_col])
            else:
                merged[col] = merged[loc_col]
            merged = merged.drop(columns=[loc_col])
    return merged


def standardize_monthly_usage(df: pd.DataFrame, location_lookup: pd.DataFrame) -> pd.DataFrame:
    """정류소별 월별 이용자수 파일을 표준 형식으로 정리한다."""
    columns = list(df.columns)
    date_col = find_column(columns, DATE_CANDIDATES)
    stop_name_col = find_column(columns, STOP_NAME_CANDIDATES)
    boarding_col = find_column(columns, BOARDING_CANDIDATES)
    alighting_col = find_column(columns, ALIGHTING_CANDIDATES)

    if not date_col or not stop_name_col or not boarding_col:
        return empty_frame(MONTHLY_COLUMNS)

    result = pd.DataFrame()
    result["year_month"] = parse_year_month(df[date_col])
    result["stop_name"] = df[stop_name_col].astype("string").str.strip()
    result["boardings"] = safe_to_number(df[boarding_col])
    result["alightings"] = safe_to_number(df[alighting_col]) if alighting_col else pd.NA
    result = result.dropna(subset=["year_month", "stop_name"])
    result = result[(result["boardings"].isna()) | (result["boardings"] >= 0)]
    result = result[(result["alightings"].isna()) | (result["alightings"] >= 0)]

    result = (
        result.groupby(["year_month", "stop_name"], dropna=False)
        .agg(boardings=("boardings", "sum"), alightings=("alightings", "sum"))
        .reset_index()
    )
    result["total_users"] = result["boardings"].fillna(0) + result["alightings"].fillna(0)
    result = merge_location(result, location_lookup)
    result = add_year_month_columns(result)
    result["boardings_per_route"] = result["boardings"] / result["route_count"].where(result["route_count"] > 0)
    for col in MONTHLY_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    return result[MONTHLY_COLUMNS]


def standardize_hourly_usage(df: pd.DataFrame, location_lookup: pd.DataFrame) -> pd.DataFrame:
    """시간대별 wide 형식 승하차 자료를 long 형식으로 바꾼다."""
    columns = list(df.columns)
    date_col = find_column(columns, DATE_CANDIDATES)
    stop_name_col = find_column(columns, STOP_NAME_CANDIDATES)
    stop_id_col = find_column(columns, STOP_ID_CANDIDATES)
    kind_col = find_column(columns, KIND_CANDIDATES)
    hours = hour_columns(columns)

    if not date_col or not stop_name_col or not kind_col or not hours:
        return empty_frame(HOURLY_COLUMNS)

    base_cols = [date_col, stop_name_col, kind_col]
    if stop_id_col:
        base_cols.append(stop_id_col)

    melted = df[base_cols + hours].copy()
    for hour_col in hours:
        melted[hour_col] = safe_to_number(melted[hour_col])

    long_df = melted.melt(id_vars=base_cols, value_vars=hours, var_name="hour_label", value_name="passengers")
    long_df["year_month"] = parse_year_month(long_df[date_col])
    long_df["stop_name"] = long_df[stop_name_col].astype("string").str.strip()
    long_df["stop_id"] = clean_stop_id(long_df[stop_id_col]) if stop_id_col else pd.NA
    long_df["hour"] = safe_to_number(long_df["hour_label"].astype("string").str.replace("시", "", regex=False)).astype("Int64")
    long_df["kind"] = long_df[kind_col].astype("string").str.strip()
    long_df["passengers"] = pd.to_numeric(long_df["passengers"], errors="coerce")
    long_df = long_df.dropna(subset=["year_month", "stop_name", "hour"])
    long_df = long_df[long_df["passengers"].isna() | (long_df["passengers"] >= 0)]

    pivot = (
        long_df.pivot_table(
            index=["year_month", "stop_id", "stop_name", "hour"],
            columns="kind",
            values="passengers",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    pivot["boardings"] = pivot["승차"] if "승차" in pivot.columns else 0
    pivot["alightings"] = pivot["하차"] if "하차" in pivot.columns else 0
    pivot["total_users"] = pivot["boardings"].fillna(0) + pivot["alightings"].fillna(0)
    result = merge_location(pivot, location_lookup)
    result = add_year_month_columns(result)
    for col in HOURLY_COLUMNS:
        if col not in result.columns:
            result[col] = pd.NA
    return result[HOURLY_COLUMNS]


from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.analysis import (
    add_yoy_metrics,
    attach_stop_type,
    build_data_check_report,
    build_imbalance_candidates,
    build_stop_summary,
    create_analysis_summary,
)
from src.column_mapper import (
    ALIGHTING_CANDIDATES,
    BOARDING_CANDIDATES,
    DATE_CANDIDATES,
    DISTRICT_CANDIDATES,
    LATITUDE_CANDIDATES,
    LONGITUDE_CANDIDATES,
    ROUTE_COUNT_CANDIDATES,
    STOP_ID_CANDIDATES,
    STOP_NAME_CANDIDATES,
    WEATHER_CANDIDATES,
    classify_file,
    find_column,
)
from src.data_loader import list_csv_files, read_csv_safely
from src.preprocessing import (
    HOURLY_COLUMNS,
    LOCATION_COLUMNS,
    MONTHLY_COLUMNS,
    make_location_lookup,
    standardize_hourly_usage,
    standardize_location,
    standardize_monthly_usage,
)
from src.utils import PROCESSED_DIR, RAW_DIR, empty_frame, ensure_output_dirs, save_csv
from src.visualization import generate_static_figures
from src.weather_analysis import (
    WEATHER_COLUMNS,
    calculate_weather_correlations,
    merge_bus_weather,
    standardize_weather,
)


def _mapping_rows(path: Path, dataset_type: str, df: pd.DataFrame) -> list[dict]:
    """실제 컬럼과 표준 컬럼의 연결 상태를 기록한다."""
    column_sets = {
        "stop_id": STOP_ID_CANDIDATES,
        "stop_name": STOP_NAME_CANDIDATES,
        "date": DATE_CANDIDATES,
        "district": DISTRICT_CANDIDATES,
        "boardings": BOARDING_CANDIDATES,
        "alightings": ALIGHTING_CANDIDATES,
        "latitude": LATITUDE_CANDIDATES,
        "longitude": LONGITUDE_CANDIDATES,
        "route_count": ROUTE_COUNT_CANDIDATES,
    }
    rows = []
    for standard, candidates in column_sets.items():
        original = find_column(list(df.columns), candidates, contains=True)
        if original:
            rows.append(
                {
                    "source_file": str(path),
                    "dataset_type": dataset_type,
                    "standard_column": standard,
                    "original_column": original,
                }
            )
    for standard, candidates in WEATHER_CANDIDATES.items():
        original = find_column(list(df.columns), candidates, contains=True)
        if original:
            rows.append(
                {
                    "source_file": str(path),
                    "dataset_type": dataset_type,
                    "standard_column": f"weather_{standard}",
                    "original_column": original,
                }
            )
    return rows


def main() -> None:
    """원본 CSV를 읽어 Streamlit 앱에서 바로 쓸 정제 CSV를 만든다."""
    ensure_output_dirs()
    csv_files = list_csv_files(RAW_DIR)
    if not csv_files:
        raise FileNotFoundError("data/raw 폴더에 CSV 파일이 없습니다.")

    print(f"[시작] CSV {len(csv_files)}개를 확인합니다.")

    raw_frames: list[tuple[Path, str, str, pd.DataFrame]] = []
    file_reports: list[dict] = []
    mapping_rows: list[dict] = []

    for path in csv_files:
        try:
            df, encoding = read_csv_safely(path)
            dataset_type = classify_file(path, df)
            raw_frames.append((path, encoding, dataset_type, df))
            file_reports.append(
                {
                    "source_file": str(path),
                    "dataset_type": dataset_type,
                    "encoding": encoding,
                    "rows": len(df),
                    "columns": len(df.columns),
                    "missing_cells": int(df.isna().sum().sum()),
                    "duplicated_rows": int(df.duplicated().sum()),
                    "status": "loaded",
                }
            )
            mapping_rows.extend(_mapping_rows(path, dataset_type, df))
            print(f"  - {path.name}: {dataset_type}, {len(df):,}행, {encoding}")
        except Exception as exc:
            file_reports.append(
                {
                    "source_file": str(path),
                    "dataset_type": "unknown",
                    "encoding": "",
                    "rows": 0,
                    "columns": 0,
                    "missing_cells": 0,
                    "duplicated_rows": 0,
                    "status": f"failed: {exc}",
                }
            )
            print(f"  ! {path.name}: 읽기 실패 - {exc}")

    location_parts = [standardize_location(df) for _, _, kind, df in raw_frames if kind == "location"]
    location = pd.concat(location_parts, ignore_index=True) if location_parts else empty_frame(LOCATION_COLUMNS)
    location_lookup = make_location_lookup(location)

    monthly_parts = [
        standardize_monthly_usage(df, location_lookup) for _, _, kind, df in raw_frames if kind == "monthly"
    ]
    hourly_parts = [
        standardize_hourly_usage(df, location_lookup) for _, _, kind, df in raw_frames if kind == "hourly"
    ]
    weather_results = [standardize_weather(df) for _, _, kind, df in raw_frames if kind == "weather"]

    monthly_summary = pd.concat(monthly_parts, ignore_index=True) if monthly_parts else empty_frame(MONTHLY_COLUMNS)
    hourly_summary = pd.concat(hourly_parts, ignore_index=True) if hourly_parts else empty_frame(HOURLY_COLUMNS)

    if not monthly_summary.empty:
        monthly_summary = monthly_summary.drop_duplicates()
    if not hourly_summary.empty:
        hourly_summary = hourly_summary.drop_duplicates()

    stop_summary = build_stop_summary(monthly_summary, hourly_summary)
    monthly_summary = attach_stop_type(stop_summary, monthly_summary)
    hourly_summary = attach_stop_type(stop_summary, hourly_summary)
    monthly_summary = add_yoy_metrics(monthly_summary)
    imbalance_candidates = build_imbalance_candidates(stop_summary)

    weather_frames = [result[0] for result in weather_results if not result[0].empty]
    weather_mapping_frames = [result[1] for result in weather_results if not result[1].empty]
    weather_monthly = pd.concat(weather_frames, ignore_index=True) if weather_frames else empty_frame(WEATHER_COLUMNS)
    if not weather_monthly.empty:
        weather_monthly = (
            weather_monthly.groupby(["year_month", "year", "month", "month_label"], dropna=False)
            .agg(
                avg_temp=("avg_temp", "mean"),
                max_temp=("max_temp", "mean"),
                min_temp=("min_temp", "mean"),
                rainfall=("rainfall", "sum"),
                humidity=("humidity", "mean"),
                wind=("wind", "mean"),
                sunshine=("sunshine", "sum"),
                snow=("snow", "max"),
            )
            .reset_index()
        )
    weather_mapping = pd.concat(weather_mapping_frames, ignore_index=True) if weather_mapping_frames else pd.DataFrame()
    weather_audit = pd.DataFrame(
        [
            {
                "item": "weather_monthly_rows",
                "value": len(weather_monthly),
                "note": "월별 날씨 자료 행 수",
            },
            {
                "item": "bus_weather_merge_rule",
                "value": "year, month",
                "note": "정류소별 행에 날씨를 반복 부착하지 않고 월별 전체 이용량과 결합",
            },
        ]
    )
    bus_weather = merge_bus_weather(monthly_summary, weather_monthly)
    weather_corr = calculate_weather_correlations(bus_weather)
    column_mapping = pd.DataFrame(mapping_rows)
    data_check_report = build_data_check_report(file_reports, monthly_summary, hourly_summary)

    save_csv(stop_summary, PROCESSED_DIR / "stop_summary.csv")
    save_csv(hourly_summary, PROCESSED_DIR / "hourly_summary.csv")
    save_csv(monthly_summary, PROCESSED_DIR / "monthly_summary.csv")
    save_csv(column_mapping, PROCESSED_DIR / "column_mapping.csv")
    save_csv(data_check_report, PROCESSED_DIR / "data_check_report.csv")
    save_csv(weather_monthly, PROCESSED_DIR / "weather_monthly.csv")
    save_csv(weather_mapping, PROCESSED_DIR / "weather_mapping.csv")
    save_csv(weather_audit, PROCESSED_DIR / "weather_audit.csv")
    save_csv(bus_weather, PROCESSED_DIR / "bus_weather_monthly_merged.csv")
    save_csv(weather_corr, PROCESSED_DIR / "weather_correlation_results.csv")
    save_csv(imbalance_candidates, PROCESSED_DIR / "imbalance_candidates.csv")

    summary_text = create_analysis_summary(stop_summary, monthly_summary, hourly_summary, weather_monthly)
    (PROCESSED_DIR / "analysis_summary.md").write_text(summary_text, encoding="utf-8")

    generate_static_figures(stop_summary, hourly_summary, monthly_summary)

    print("[완료] 전처리 결과를 outputs/processed에 저장했습니다.")
    print(f"  - stop_summary: {len(stop_summary):,}행")
    print(f"  - monthly_summary: {len(monthly_summary):,}행")
    print(f"  - hourly_summary: {len(hourly_summary):,}행")
    print(f"  - weather_monthly: {len(weather_monthly):,}행")


if __name__ == "__main__":
    main()


from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .utils import normalize_text


STOP_ID_CANDIDATES = ["정류소ID", "정류소아이디", "정류장ID", "정류소번호", "bs_id", "stop_id"]
STOP_NAME_CANDIDATES = ["정류소명", "정류장명", "버스정류소명", "stop_name"]
BOARDING_CANDIDATES = ["승차", "승차인원", "승차승객수", "boardings"]
ALIGHTING_CANDIDATES = ["하차", "하차인원", "하차승객수", "alightings"]
LATITUDE_CANDIDATES = ["위도", "latitude", "lat"]
LONGITUDE_CANDIDATES = ["경도", "longitude", "lon", "lng"]
ROUTE_COUNT_CANDIDATES = ["경유노선수", "노선수", "버스노선수", "route_count"]
DATE_CANDIDATES = ["년월", "일시", "date", "month"]
DISTRICT_CANDIDATES = ["구군", "구·군", "행정구역", "시군구", "district"]
KIND_CANDIDATES = ["구분", "승하차구분", "type"]


WEATHER_CANDIDATES = {
    "date": ["일시", "년월"],
    "avg_temp": ["평균기온(°C)", "평균기온", "평균 기온"],
    "max_temp": ["최고기온(°C)", "평균최고기온", "평균 최고기온"],
    "min_temp": ["최저기온(°C)", "평균최저기온", "평균 최저기온"],
    "rainfall": ["일강수량(mm)", "월합강수량", "강수량"],
    "humidity": ["평균 상대습도(%)", "평균상대습도", "상대습도"],
    "wind": ["평균 풍속(m/s)", "평균풍속", "풍속"],
    "sunshine": ["합계 일조시간(hr)", "합계 일조시간", "일조시간"],
    "snow": ["일 최심적설(cm)", "최심적설", "적설"],
}


@dataclass
class ColumnMapping:
    source_file: str
    dataset_type: str
    original_column: str
    standard_column: str


def find_column(columns: list[str], candidates: list[str], contains: bool = False) -> str | None:
    """후보명과 실제 컬럼명을 비교해 가장 알맞은 컬럼을 찾는다."""
    normalized = {normalize_text(col): col for col in columns}
    for candidate in candidates:
        key = normalize_text(candidate)
        if key in normalized:
            return normalized[key]
    if contains:
        for candidate in candidates:
            key = normalize_text(candidate)
            for original_key, original_col in normalized.items():
                if key and key in original_key:
                    return original_col
    return None


def hour_columns(columns: list[str]) -> list[str]:
    """05시, 06시처럼 시간대를 뜻하는 컬럼을 찾는다."""
    result = []
    for col in columns:
        text = str(col).strip()
        if re.fullmatch(r"\d{1,2}\s*시", text):
            result.append(col)
    return result


def classify_file(path: Path, df: pd.DataFrame) -> str:
    """파일명과 컬럼명을 보고 데이터 종류를 판정한다."""
    columns = list(df.columns)
    name = path.name.lower()
    has_weather = find_column(columns, WEATHER_CANDIDATES["avg_temp"], contains=True) is not None
    has_lat_lon = find_column(columns, LATITUDE_CANDIDATES) and find_column(columns, LONGITUDE_CANDIDATES)
    has_hour = bool(hour_columns(columns))
    has_stop = find_column(columns, STOP_NAME_CANDIDATES) is not None
    has_boarding = find_column(columns, BOARDING_CANDIDATES) is not None
    has_alighting = find_column(columns, ALIGHTING_CANDIDATES) is not None

    if "obs_asos" in name or has_weather:
        return "weather"
    if has_lat_lon and has_stop:
        return "location"
    if has_hour and has_stop:
        return "hourly"
    if has_stop and (has_boarding or has_alighting):
        return "monthly"
    return "unknown"


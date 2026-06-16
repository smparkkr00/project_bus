from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "outputs" / "processed"
FIGURE_DIR = PROJECT_ROOT / "outputs" / "figures"

ENCODING_CANDIDATES = ["utf-8", "utf-8-sig", "cp949", "euc-kr"]


def ensure_output_dirs() -> None:
    """결과 저장 폴더가 없으면 만든다."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)


def clean_column_name(value: object) -> str:
    """컬럼명의 앞뒤 공백과 보이지 않는 문자를 정리한다."""
    return str(value).replace("\ufeff", "").strip()


def normalize_text(value: object) -> str:
    """비교용 문자열을 공백 없이 소문자로 바꾼다."""
    return clean_column_name(value).replace(" ", "").replace("_", "").lower()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """데이터프레임 컬럼명을 일괄 정리한다."""
    result = df.copy()
    result.columns = [clean_column_name(col) for col in result.columns]
    return result


def safe_to_number(series: pd.Series) -> pd.Series:
    """쉼표가 들어간 숫자 문자열을 안전하게 숫자로 바꾼다."""
    cleaned = (
        series.astype("string")
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.replace("\u2212", "-", regex=False)
    )
    return pd.to_numeric(cleaned, errors="coerce")


def clean_stop_id(series: pd.Series) -> pd.Series:
    """정류소 ID를 문자열로 통일하고 엑셀식 .0 꼬리를 제거한다."""
    cleaned = series.astype("string").str.strip()
    cleaned = cleaned.str.replace(r"\.0$", "", regex=True)
    cleaned = cleaned.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    return cleaned


def parse_year_month(series: pd.Series) -> pd.Series:
    """년월 컬럼을 월 단위 Timestamp로 바꾼다."""
    parsed = pd.to_datetime(series, errors="coerce")
    return parsed.dt.to_period("M").dt.to_timestamp()


def add_year_month_columns(df: pd.DataFrame, date_col: str = "year_month") -> pd.DataFrame:
    """월 컬럼에서 year, month, month_label을 만든다."""
    result = df.copy()
    result[date_col] = pd.to_datetime(result[date_col], errors="coerce")
    result["year"] = result[date_col].dt.year.astype("Int64")
    result["month"] = result[date_col].dt.month.astype("Int64")
    result["month_label"] = result[date_col].dt.strftime("%Y-%m")
    return result


def safe_divide(numerator: pd.Series | float, denominator: pd.Series | float) -> pd.Series | float:
    """0으로 나누는 오류를 피하면서 비율을 계산한다."""
    if isinstance(denominator, pd.Series):
        denom = denominator.replace(0, np.nan)
        return numerator / denom
    if denominator in (0, None) or (isinstance(denominator, float) and math.isnan(denominator)):
        return np.nan
    return numerator / denominator


def mode_or_first(values: Iterable[object]) -> object:
    """가장 자주 나오는 값을 고르고, 없으면 첫 값을 반환한다."""
    series = pd.Series(list(values)).dropna()
    if series.empty:
        return pd.NA
    mode = series.mode(dropna=True)
    if not mode.empty:
        return mode.iloc[0]
    return series.iloc[0]


def quantile_or_nan(series: pd.Series, q: float) -> float:
    """빈 데이터에서 분위수를 구할 때 생기는 오류를 피한다."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return float("nan")
    return float(numeric.quantile(q))


def newest_mtime(paths: Iterable[Path]) -> float | None:
    """파일 목록에서 가장 최근 수정 시간을 구한다."""
    times = [path.stat().st_mtime for path in paths if path.exists()]
    if not times:
        return None
    return max(times)


def save_csv(df: pd.DataFrame, path: Path) -> None:
    """한글 컬럼이 엑셀에서도 잘 보이도록 utf-8-sig로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f"{path.stem}.tmp{path.suffix}")
    df.to_csv(temp_path, index=False, encoding="utf-8-sig")
    temp_path.replace(path)


def empty_frame(columns: list[str]) -> pd.DataFrame:
    """필요한 컬럼을 가진 빈 데이터프레임을 만든다."""
    return pd.DataFrame(columns=columns)

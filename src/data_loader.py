from __future__ import annotations

from pathlib import Path

import pandas as pd

from .utils import ENCODING_CANDIDATES, RAW_DIR, normalize_columns


def list_csv_files(raw_dir: Path = RAW_DIR) -> list[Path]:
    """data/raw 아래의 모든 CSV 파일을 찾는다."""
    if not raw_dir.exists():
        return []
    return sorted(raw_dir.rglob("*.csv"))


def read_csv_safely(path: Path, nrows: int | None = None) -> tuple[pd.DataFrame, str]:
    """여러 인코딩을 순서대로 시도해 CSV를 읽는다."""
    last_error: Exception | None = None
    for encoding in ENCODING_CANDIDATES:
        try:
            df = pd.read_csv(
                path,
                encoding=encoding,
                dtype="string",
                nrows=nrows,
                low_memory=False,
            )
            return normalize_columns(df), encoding
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"{path} 파일을 읽을 수 없습니다: {last_error}")


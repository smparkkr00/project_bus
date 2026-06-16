from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    import pydeck as pdk
except Exception:
    pdk = None

from src.utils import PROCESSED_DIR, RAW_DIR, newest_mtime
from src.weather_analysis import WEATHER_LABELS, interpret_correlation, spearman_without_scipy


PAGE_TITLE = "대구 시내버스 정류소 이용 수요 및 노선 공급 불균형 분석"
REQUIRED_FILES = [
    "stop_summary.csv",
    "hourly_summary.csv",
    "monthly_summary.csv",
    "weather_monthly.csv",
    "bus_weather_monthly_merged.csv",
    "weather_correlation_results.csv",
    "imbalance_candidates.csv",
    "data_check_report.csv",
]

METRIC_LABELS = {
    "boardings": "승차 인원",
    "alightings": "하차 인원",
    "total_users": "전체 이용객",
    "total_boardings": "전체 승차 인원",
    "total_alightings": "전체 하차 인원",
}

PUBLIC_LABELS = {
    "stop_name": "정류소명",
    "district": "구·군",
    "total_boardings": "전체 승차 인원",
    "total_alightings": "전체 하차 인원",
    "total_users": "전체 이용객",
    "route_count": "경유 노선 수",
    "boardings_per_route": "노선당 승차 인원",
    "commute_focus_pct": "출근 시간 집중도",
    "evening_focus_pct": "퇴근 시간 집중도",
    "peak_hour": "최대 혼잡 시간대",
    "stop_type": "정류소 유형",
    "boarding_alighting_gap": "승하차 차이",
    "imbalance_index": "승하차 불균형 지수",
    "month_label": "연월",
    "boardings": "승차 인원",
    "alightings": "하차 인원",
    "yoy_change": "증감 인원",
    "yoy_rate": "증감률",
    "previous_year_boardings": "전년 동월 승차 인원",
}


st.set_page_config(page_title=PAGE_TITLE, layout="wide")

st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.4rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: 0;}
    .metric-row [data-testid="stMetric"] {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 14px 16px;
    }
    .analysis-note {
        color: #4b5563;
        font-size: 0.94rem;
        line-height: 1.55;
    }
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 0.92rem;
    }
    .styled-table th {
        background: #f3f4f6;
        color: #111827;
        text-align: left;
        padding: 9px 10px;
        border-bottom: 1px solid #d1d5db;
    }
    .styled-table td {
        padding: 8px 10px;
        border-bottom: 1px solid #e5e7eb;
    }
    .badge-up, .badge-down, .badge-flat {
        display: inline-block;
        min-width: 58px;
        text-align: center;
        border-radius: 999px;
        padding: 2px 8px;
        font-weight: 650;
    }
    .badge-up {background: #e7f6ec; color: #146c2e;}
    .badge-down {background: #fdecec; color: #a32020;}
    .badge-flat {background: #eef2f7; color: #374151;}
    </style>
    """,
    unsafe_allow_html=True,
)


def file_ready() -> tuple[bool, list[str]]:
    """정제 CSV가 모두 있는지 확인한다."""
    missing = [name for name in REQUIRED_FILES if not (PROCESSED_DIR / name).exists()]
    return len(missing) == 0, missing


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    """정제 CSV를 캐시해서 빠르게 불러온다."""
    path = PROCESSED_DIR / name
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, encoding="utf-8-sig")
    if "year_month" in df.columns:
        df["year_month"] = pd.to_datetime(df["year_month"], errors="coerce")
    return df


def format_int(value: object) -> str:
    """숫자를 천 단위 쉼표가 있는 문자열로 바꾼다."""
    if pd.isna(value):
        return "-"
    return f"{float(value):,.0f}"


def format_pct(value: object) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):.1f}%"


def add_metric_columns(monthly: pd.DataFrame, hourly: pd.DataFrame) -> pd.DataFrame:
    """선택된 필터 범위에서 정류소 요약을 다시 계산한다."""
    if monthly.empty:
        return pd.DataFrame()
    group = (
        monthly.groupby(["stop_name", "district"], dropna=False)
        .agg(
            total_boardings=("boardings", "sum"),
            total_alightings=("alightings", "sum"),
            total_users=("total_users", "sum"),
            route_count=("route_count", "max"),
            latitude=("latitude", "mean"),
            longitude=("longitude", "mean"),
            stop_type=("stop_type", "first"),
        )
        .reset_index()
    )
    group["boardings_per_route"] = group["total_boardings"] / group["route_count"].where(group["route_count"] > 0)
    group["boarding_alighting_gap"] = group["total_boardings"] - group["total_alightings"]
    group["imbalance_index"] = (
        group["boarding_alighting_gap"].abs() / (group["total_boardings"] + group["total_alightings"]).replace(0, np.nan)
    )

    if not hourly.empty:
        features = (
            hourly.assign(
                commute_part=lambda df: df["boardings"].where(df["hour"].isin([7, 8, 9]), 0),
                evening_part=lambda df: df["boardings"].where(df["hour"].isin([17, 18, 19, 20]), 0),
            )
            .groupby(["stop_name", "district"], dropna=False)
            .agg(
                hourly_boardings=("boardings", "sum"),
                commute_boardings=("commute_part", "sum"),
                evening_boardings=("evening_part", "sum"),
            )
            .reset_index()
        )
        peak = (
            hourly.groupby(["stop_name", "district", "hour"], dropna=False)["boardings"]
            .sum()
            .reset_index()
            .sort_values(["stop_name", "district", "boardings"], ascending=[True, True, False])
            .drop_duplicates(["stop_name", "district"])
        )
        peak["peak_hour"] = peak["hour"].astype("Int64").astype("string") + "시"
        group = group.merge(features, on=["stop_name", "district"], how="left")
        group = group.merge(peak[["stop_name", "district", "peak_hour"]], on=["stop_name", "district"], how="left")
        group["commute_focus_pct"] = group["commute_boardings"] / group["hourly_boardings"].replace(0, np.nan) * 100
        group["evening_focus_pct"] = group["evening_boardings"] / group["hourly_boardings"].replace(0, np.nan) * 100
    else:
        group["commute_focus_pct"] = np.nan
        group["evening_focus_pct"] = np.nan
        group["peak_hour"] = pd.NA
    return group


def apply_filters(
    monthly: pd.DataFrame,
    hourly: pd.DataFrame,
    districts: list[str],
    stops: list[str],
    months: list[str],
    hours: list[int],
    stop_types: list[str],
    min_boardings: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """사이드바 필터를 월별 자료와 시간대별 자료에 동일하게 적용한다."""
    m = monthly.copy()
    h = hourly.copy()
    for df_name, df in [("monthly", m), ("hourly", h)]:
        if df.empty:
            continue
        mask = pd.Series(True, index=df.index)
        if districts:
            mask &= df["district"].isin(districts)
        if stops:
            mask &= df["stop_name"].isin(stops)
        if months and "month_label" in df.columns:
            mask &= df["month_label"].isin(months)
        if stop_types and "stop_type" in df.columns:
            mask &= df["stop_type"].isin(stop_types)
        if df_name == "hourly" and hours:
            mask &= df["hour"].isin(hours)
        if df_name == "monthly" and min_boardings > 0:
            mask &= df["boardings"] >= min_boardings
        if df_name == "monthly":
            m = df[mask].copy()
        else:
            h = df[mask].copy()
    return m, h


def plot_or_info(fig, message: str = "표시할 데이터가 없습니다.") -> None:
    """빈 그래프 대신 안내문을 보여준다."""
    if fig is None:
        st.info(message)
    else:
        st.plotly_chart(fig, width="stretch")


def bar_top(df: pd.DataFrame, x: str, y: str, title: str, top_n: int, color: str = "#2f6f9f"):
    if df.empty or x not in df.columns or y not in df.columns:
        return None
    top = df.nlargest(top_n, x).sort_values(x)
    return px.bar(top, x=x, y=y, orientation="h", title=title, labels=PUBLIC_LABELS, color_discrete_sequence=[color])


def line_by_hour(hourly: pd.DataFrame, metric: str, title: str):
    if hourly.empty:
        return None
    by_hour = hourly.groupby("hour", dropna=False)[metric].sum().reset_index()
    if by_hour.empty:
        return None
    return px.line(by_hour, x="hour", y=metric, markers=True, title=title, labels=PUBLIC_LABELS)


def styled_table(df: pd.DataFrame, columns: list[str], limit: int | None = None) -> None:
    """발표용으로 읽기 좋은 HTML 표를 출력한다."""
    if df.empty:
        st.info("표시할 데이터가 없습니다.")
        return
    view = df.copy()
    if limit:
        view = view.head(limit)
    available = [col for col in columns if col in view.columns]
    view = view[available].rename(columns=PUBLIC_LABELS)
    html = view.to_html(index=False, classes="styled-table", escape=False)
    st.markdown(html, unsafe_allow_html=True)


def render_yoy_table(df: pd.DataFrame, title: str, ascending: bool, top_n: int) -> None:
    """전년 동월 대비 증감률 표를 배지 스타일로 보여준다."""
    st.subheader(title)
    if df.empty:
        st.info("선택한 기준월에서 전년 동월 비교가 가능한 정류소가 없습니다.")
        return
    ranked = df.sort_values("yoy_rate", ascending=ascending).head(top_n).copy()
    rows = []
    for rank, (_, row) in enumerate(ranked.iterrows(), start=1):
        rate = row["yoy_rate"]
        badge_class = "badge-up" if rate > 0 else "badge-down" if rate < 0 else "badge-flat"
        rows.append(
            "<tr>"
            f"<td>{rank}</td>"
            f"<td>{row.get('stop_name', '-')}</td>"
            f"<td>{row.get('month_label', '-')}</td>"
            f"<td>{format_int(row.get('boardings'))}</td>"
            f"<td>{format_int(row.get('previous_year_boardings'))}</td>"
            f"<td>{format_int(row.get('yoy_change'))}</td>"
            f"<td><span class='{badge_class}'>{format_pct(rate)}</span></td>"
            "</tr>"
        )
    html = (
        "<table class='styled-table'><thead><tr>"
        "<th>순위</th><th>정류소명</th><th>연월</th><th>승차 인원</th>"
        "<th>전년 동월</th><th>증감 인원</th><th>증감률</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )
    st.markdown(html, unsafe_allow_html=True)


ready, missing_files = file_ready()
if not ready:
    st.title(PAGE_TITLE)
    st.error("정제 CSV가 아직 준비되지 않았습니다.")
    st.markdown("먼저 PowerShell에서 아래 명령을 실행해 주세요.")
    st.code("python prepare_data.py", language="powershell")
    st.caption("Streamlit 앱은 원본 CSV를 즉석에서 처리하지 않고 outputs/processed의 정제 CSV만 읽습니다.")
    st.write("누락 파일:", ", ".join(missing_files))
    st.stop()

processed_times = [PROCESSED_DIR / name for name in REQUIRED_FILES]
raw_newest = newest_mtime(RAW_DIR.rglob("*.csv"))
processed_oldest = min(path.stat().st_mtime for path in processed_times if path.exists())
if raw_newest and raw_newest > processed_oldest:
    st.warning("원본 CSV가 정제 CSV보다 최신입니다. 전처리 결과가 오래되었으므로 `python prepare_data.py`를 다시 실행해 주세요.")

stop_summary = load_csv("stop_summary.csv")
monthly_summary = load_csv("monthly_summary.csv")
hourly_summary = load_csv("hourly_summary.csv")
weather_monthly = load_csv("weather_monthly.csv")
bus_weather = load_csv("bus_weather_monthly_merged.csv")
weather_corr = load_csv("weather_correlation_results.csv")
imbalance_candidates = load_csv("imbalance_candidates.csv")
data_check = load_csv("data_check_report.csv")

st.title(PAGE_TITLE)
st.markdown(
    "<p class='analysis-note'>대구 시내버스 정류소의 시간대별 승하차 데이터를 분석하여 수요가 집중되는 시간과 지역을 확인하고, "
    "이용 수요 대비 경유 노선 수가 상대적으로 적은 정류소 후보를 탐색합니다.</p>",
    unsafe_allow_html=True,
)

district_options = sorted(monthly_summary["district"].dropna().astype(str).unique()) if not monthly_summary.empty else []
stop_options = sorted(monthly_summary["stop_name"].dropna().astype(str).unique()) if not monthly_summary.empty else []
month_options = sorted(monthly_summary["month_label"].dropna().astype(str).unique()) if not monthly_summary.empty else []
hour_options = sorted(hourly_summary["hour"].dropna().astype(int).unique()) if not hourly_summary.empty else []
type_options = sorted(stop_summary["stop_type"].dropna().astype(str).unique()) if not stop_summary.empty else []
default_months = month_options[-1:] if month_options else []

with st.sidebar:
    st.header("필터")
    selected_districts = st.multiselect("구·군 선택", district_options)
    selected_stops = st.multiselect("정류소 선택", stop_options)
    selected_months = st.multiselect("연도·월 선택", month_options, default=default_months)
    selected_hours = st.multiselect("시간대 선택", hour_options, default=hour_options)
    selected_types = st.multiselect("정류소 유형 선택", type_options)
    selected_metric = st.selectbox("승차·하차·전체 이용객 선택", ["boardings", "alightings", "total_users"], format_func=METRIC_LABELS.get)
    top_n = st.slider("TOP N 선택", min_value=5, max_value=30, value=10, step=1)
    min_boardings = st.number_input("최소 승차 인원", min_value=0, value=0, step=1000)
    max_route_count = st.number_input("최대 경유 노선 수", min_value=0, value=10, step=1)
    load_percentile = st.slider("노선당 승차 인원 기준(상위 백분위)", min_value=50, max_value=99, value=90, step=1)
    show_map = st.checkbox("지도 표시 여부", value=True)
    show_static = st.checkbox("정적 그래프 표시 여부", value=False)

filtered_monthly, filtered_hourly = apply_filters(
    monthly_summary,
    hourly_summary,
    selected_districts,
    selected_stops,
    selected_months,
    selected_hours,
    selected_types,
    min_boardings,
)
filtered_stops = add_metric_columns(filtered_monthly, filtered_hourly)

tabs = st.tabs(
    [
        "전체 현황",
        "시간대별 분석",
        "정류소별 분석",
        "수요·공급 불균형 분석",
        "지도 분석",
        "장기 추세 분석",
        "날씨와 버스 이용",
        "데이터 및 분석 한계",
    ]
)

with tabs[0]:
    st.subheader("전체 현황")
    if filtered_monthly.empty:
        st.info("선택한 필터에 해당하는 월별 이용량 데이터가 없습니다.")
    else:
        total_boardings = filtered_monthly["boardings"].sum()
        total_alightings = filtered_monthly["alightings"].sum()
        stop_count = filtered_monthly["stop_name"].nunique()
        top_stop_name = "-"
        route_load_stop = "-"
        peak_hour = "-"
        if not filtered_stops.empty:
            top_row = filtered_stops.sort_values("total_boardings", ascending=False).iloc[0]
            top_stop_name = f"{top_row['stop_name']} ({format_int(top_row['total_boardings'])}명)"
            load_row = filtered_stops.sort_values("boardings_per_route", ascending=False).iloc[0]
            route_load_stop = f"{load_row['stop_name']} ({format_int(load_row['boardings_per_route'])}명/노선)"
        if not filtered_hourly.empty:
            by_hour = filtered_hourly.groupby("hour")["boardings"].sum()
            if not by_hour.empty:
                peak_hour = f"{int(by_hour.idxmax())}시"

        st.markdown("<div class='metric-row'>", unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("전체 승차 인원", format_int(total_boardings))
        c2.metric("전체 하차 인원", format_int(total_alightings))
        c3.metric("분석 대상 정류소 수", format_int(stop_count))
        c4, c5, c6 = st.columns(3)
        c4.metric("가장 이용객이 많은 정류소", top_stop_name)
        c5.metric("가장 혼잡한 시간대", peak_hour)
        c6.metric("노선당 승차 인원이 가장 높은 정류소", route_load_stop)
        st.markdown("</div>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            plot_or_info(bar_top(filtered_stops, "total_boardings", "stop_name", "정류소 승차 인원 TOP N", top_n))
        with col2:
            district = filtered_monthly.groupby("district", dropna=False)["boardings"].sum().reset_index()
            fig = px.bar(district.sort_values("boardings", ascending=False), x="district", y="boardings", title="구·군별 승차 인원", labels=PUBLIC_LABELS)
            plot_or_info(fig)
        col3, col4 = st.columns(2)
        with col3:
            plot_or_info(line_by_hour(filtered_hourly, "boardings", "시간대별 승차 인원"))
        with col4:
            type_count = filtered_stops["stop_type"].value_counts(dropna=False).reset_index()
            type_count.columns = ["stop_type", "count"]
            fig = px.pie(type_count, names="stop_type", values="count", title="정류소 유형별 비율") if not type_count.empty else None
            plot_or_info(fig)

with tabs[1]:
    st.subheader("시간대별 분석")
    if filtered_hourly.empty:
        st.info("선택한 필터에 해당하는 시간대별 데이터가 없습니다.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            plot_or_info(line_by_hour(filtered_hourly, "boardings", "시간대별 전체 승차 인원"))
        with col2:
            by_hour = filtered_hourly.groupby("hour", dropna=False)[["boardings", "alightings"]].sum().reset_index()
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=by_hour["hour"], y=by_hour["boardings"], mode="lines+markers", name="승차"))
            fig.add_trace(go.Scatter(x=by_hour["hour"], y=by_hour["alightings"], mode="lines+markers", name="하차"))
            fig.update_layout(title="시간대별 승차·하차 비교", xaxis_title="시간대", yaxis_title="인원")
            plot_or_info(fig)

        top_names = filtered_stops.nlargest(20, "total_boardings")["stop_name"].tolist() if not filtered_stops.empty else []
        heat = filtered_hourly[filtered_hourly["stop_name"].isin(top_names)]
        col3, col4 = st.columns(2)
        with col3:
            pivot = heat.pivot_table(index="stop_name", columns="hour", values="boardings", aggfunc="sum", fill_value=0)
            fig = px.imshow(pivot, aspect="auto", title="정류소 × 시간대 히트맵", labels={"x": "시간대", "y": "정류소", "color": "승차 인원"}) if not pivot.empty else None
            plot_or_info(fig)
        with col4:
            district_pivot = filtered_hourly.pivot_table(index="district", columns="hour", values="boardings", aggfunc="sum", fill_value=0)
            fig = px.imshow(district_pivot, aspect="auto", title="구·군 × 시간대 히트맵", labels={"x": "시간대", "y": "구·군", "color": "승차 인원"}) if not district_pivot.empty else None
            plot_or_info(fig)

        col5, col6 = st.columns(2)
        with col5:
            commute = filtered_hourly[filtered_hourly["hour"].isin([7, 8, 9])]
            commute_top = commute.groupby("stop_name")["boardings"].sum().reset_index()
            plot_or_info(bar_top(commute_top, "boardings", "stop_name", "출근 시간 이용객 TOP N", top_n, "#2f6f9f"))
        with col6:
            evening = filtered_hourly[filtered_hourly["hour"].isin([17, 18, 19, 20])]
            evening_top = evening.groupby("stop_name")["boardings"].sum().reset_index()
            plot_or_info(bar_top(evening_top, "boardings", "stop_name", "퇴근 시간 이용객 TOP N", top_n, "#a45f3d"))

        compare_stops = selected_stops[:6] if selected_stops else top_names[:5]
        compare = filtered_hourly[filtered_hourly["stop_name"].isin(compare_stops)]
        pattern = compare.groupby(["stop_name", "hour"], dropna=False)["boardings"].sum().reset_index()
        fig = px.line(pattern, x="hour", y="boardings", color="stop_name", markers=True, title="선택 정류소 시간대 패턴 비교", labels=PUBLIC_LABELS) if not pattern.empty else None
        plot_or_info(fig)

with tabs[2]:
    st.subheader("정류소별 분석")
    if filtered_stops.empty:
        st.info("정류소별로 표시할 데이터가 없습니다.")
    else:
        default_stop = selected_stops[0] if selected_stops else filtered_stops.sort_values("total_boardings", ascending=False).iloc[0]["stop_name"]
        station = st.selectbox("상세 분석 정류소", sorted(filtered_stops["stop_name"].dropna().unique()), index=sorted(filtered_stops["stop_name"].dropna().unique()).index(default_stop) if default_stop in sorted(filtered_stops["stop_name"].dropna().unique()) else 0)
        station_summary = filtered_stops[filtered_stops["stop_name"] == station].sort_values("total_boardings", ascending=False).head(1)
        styled_table(
            station_summary,
            [
                "stop_name",
                "district",
                "route_count",
                "total_boardings",
                "total_alightings",
                "boardings_per_route",
                "peak_hour",
                "commute_focus_pct",
                "evening_focus_pct",
                "stop_type",
            ],
        )

        station_hourly = filtered_hourly[filtered_hourly["stop_name"] == station]
        station_monthly = monthly_summary[monthly_summary["stop_name"] == station]
        col1, col2 = st.columns(2)
        with col1:
            by_hour = station_hourly.groupby("hour")[["boardings", "alightings"]].sum().reset_index()
            fig = go.Figure()
            if not by_hour.empty:
                fig.add_trace(go.Scatter(x=by_hour["hour"], y=by_hour["boardings"], mode="lines+markers", name="승차"))
                fig.add_trace(go.Scatter(x=by_hour["hour"], y=by_hour["alightings"], mode="lines+markers", name="하차"))
                fig.update_layout(title="선택 정류소의 시간대별 승하차", xaxis_title="시간대", yaxis_title="인원")
            plot_or_info(fig if not by_hour.empty else None)
        with col2:
            all_avg = filtered_hourly.groupby("hour")["boardings"].mean().reset_index(name="전체 평균")
            district_value = station_summary["district"].iloc[0] if not station_summary.empty else None
            district_avg = filtered_hourly[filtered_hourly["district"] == district_value].groupby("hour")["boardings"].mean().reset_index(name="구·군 평균")
            station_avg = station_hourly.groupby("hour")["boardings"].mean().reset_index(name="선택 정류소")
            merged = all_avg.merge(district_avg, on="hour", how="outer").merge(station_avg, on="hour", how="outer")
            fig = px.line(merged, x="hour", y=["전체 평균", "구·군 평균", "선택 정류소"], markers=True, title="전체·구군 평균과 비교") if not merged.empty else None
            plot_or_info(fig)
        trend = station_monthly.groupby("month_label")["boardings"].sum().reset_index()
        fig = px.line(trend, x="month_label", y="boardings", markers=True, title="월별 이용량 추세", labels=PUBLIC_LABELS) if not trend.empty else None
        plot_or_info(fig)

with tabs[3]:
    st.subheader("수요·공급 불균형 분석")
    if filtered_stops.empty:
        st.info("분석할 정류소 데이터가 없습니다.")
    else:
        scatter_stops = filtered_stops.copy()
        for col in ["route_count", "total_boardings", "boardings_per_route"]:
            scatter_stops[col] = pd.to_numeric(scatter_stops[col], errors="coerce")
        scatter_stops["boardings_per_route"] = scatter_stops["boardings_per_route"].fillna(0).clip(lower=0)
        scatter_stops = scatter_stops.dropna(subset=["route_count", "total_boardings"])
        fig = px.scatter(
            scatter_stops,
            x="route_count",
            y="total_boardings",
            size="boardings_per_route",
            color="stop_type",
            hover_data={
                "stop_name": True,
                "district": True,
                "total_boardings": ":,.0f",
                "route_count": ":,.0f",
                "boardings_per_route": ":,.0f",
                "peak_hour": True,
                "stop_type": True,
            },
            title="경유 노선 수와 전체 승차 인원",
            labels=PUBLIC_LABELS,
        )
        route_median = scatter_stops["route_count"].median()
        boarding_median = scatter_stops["total_boardings"].median()
        if pd.notna(route_median) and pd.notna(boarding_median):
            fig.add_vline(x=route_median, line_dash="dash", line_color="#6b7280")
            fig.add_hline(y=boarding_median, line_dash="dash", line_color="#6b7280")
            fig.add_vrect(x0=0, x1=route_median, y0=boarding_median, y1=scatter_stops["total_boardings"].max(), fillcolor="#f59e0b", opacity=0.12, line_width=0)
        plot_or_info(fig)

        load_cutoff = filtered_stops["boardings_per_route"].quantile(load_percentile / 100)
        demand_cutoff = filtered_stops["total_boardings"].quantile(0.75)
        candidates = filtered_stops[
            (filtered_stops["total_boardings"] >= demand_cutoff)
            & (filtered_stops["route_count"] <= max_route_count)
            & (filtered_stops["boardings_per_route"] >= load_cutoff)
        ].sort_values("boardings_per_route", ascending=False)
        col1, col2 = st.columns(2)
        with col1:
            plot_or_info(bar_top(filtered_stops, "boardings_per_route", "stop_name", "노선당 승차 인원 TOP N", top_n, "#a45f3d"))
        with col2:
            st.markdown("#### 추가 검토 후보 정류소")
            st.caption("이 결과는 실제 노선 부족을 확정하는 것이 아니라 추가 검토가 필요한 후보 정류소입니다.")
            styled_table(
                candidates,
                [
                    "stop_name",
                    "district",
                    "total_boardings",
                    "total_alightings",
                    "total_users",
                    "route_count",
                    "boardings_per_route",
                    "commute_focus_pct",
                    "evening_focus_pct",
                    "peak_hour",
                    "stop_type",
                ],
                top_n,
            )
            st.download_button(
                "추가 검토 후보 CSV 다운로드",
                data=candidates.to_csv(index=False, encoding="utf-8-sig"),
                file_name="imbalance_candidates_filtered.csv",
                mime="text/csv",
            )

with tabs[4]:
    st.subheader("지도 분석")
    if not show_map:
        st.info("사이드바에서 지도 표시 여부를 켜면 지도를 볼 수 있습니다.")
    elif pdk is None:
        st.info("pydeck이 설치되어 있지 않아 지도 분석을 표시할 수 없습니다.")
    elif filtered_stops.empty or filtered_stops[["latitude", "longitude"]].dropna().empty:
        st.info("위치정보 데이터가 없어 지도 분석을 표시할 수 없습니다.")
    else:
        map_data = filtered_stops.dropna(subset=["latitude", "longitude"]).copy()
        map_data["radius"] = float(np.sqrt(60) * 4 + 35)
        load = pd.to_numeric(map_data["boardings_per_route"], errors="coerce").fillna(0).clip(lower=0)
        low_load = load.quantile(0.05)
        high_load = load.quantile(0.95)
        if pd.isna(high_load) or high_load <= low_load:
            intensity = pd.Series(0.0, index=map_data.index)
        else:
            intensity = ((load.clip(low_load, high_load) - low_load) / (high_load - low_load)).clip(0, 1)
        alpha = 0.3 + intensity * 0.7
        red = (123 + intensity * 132).round().astype(int)
        map_data["color"] = [
            [int(red_value), 0, 0, int(round(alpha_value * 255))]
            for red_value, alpha_value in zip(red, alpha)
        ]
        layer = pdk.Layer(
            "ScatterplotLayer",
            data=map_data,
            get_position="[longitude, latitude]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )
        view_state = pdk.ViewState(
            latitude=float(map_data["latitude"].mean()),
            longitude=float(map_data["longitude"].mean()),
            zoom=10,
            pitch=0,
        )
        tooltip = {
            "html": "<b>{stop_name}</b><br/>구·군: {district}<br/>승차 인원: {total_boardings}<br/>경유 노선 수: {route_count}<br/>노선당 승차 인원: {boardings_per_route}<br/>정류소 유형: {stop_type}",
            "style": {"backgroundColor": "#111827", "color": "white"},
        }
        st.pydeck_chart(pdk.Deck(layers=[layer], initial_view_state=view_state, tooltip=tooltip))

with tabs[5]:
    st.subheader("장기 추세 분석")
    yoy_available = monthly_summary.dropna(subset=["previous_year_boardings", "yoy_rate"]).copy()
    if yoy_available.empty:
        st.info("전년 동월 비교가 가능한 월별 데이터가 없습니다.")
    else:
        comparable_months = sorted(yoy_available["month_label"].dropna().unique())
        default_comparable = comparable_months[-1]
        selected_base_month = st.selectbox("비교 기준월", comparable_months, index=comparable_months.index(default_comparable))
        base = yoy_available[yoy_available["month_label"] == selected_base_month].copy()
        valid_increase = base[base["previous_year_boardings"] >= 1000]
        valid_decrease = base[(base["previous_year_boardings"] >= 1000) & (base["boardings"] >= 1000)]
        col1, col2 = st.columns(2)
        with col1:
            render_yoy_table(valid_increase, "증가율 TOP N", ascending=False, top_n=top_n)
        with col2:
            render_yoy_table(valid_decrease, "감소율 TOP N", ascending=True, top_n=top_n)
        st.caption("전년 동월 승차 인원이 1,000명 미만인 경우 증감률이 과장될 수 있어 순위에서 제외했습니다. 감소율 표에서는 현재 월 승차 인원도 1,000명 이상인 정류소만 표시합니다.")

        trend = filtered_monthly.groupby("month_label")["boardings"].sum().reset_index()
        fig = px.line(trend, x="month_label", y="boardings", markers=True, title="월별 이용량 추세", labels=PUBLIC_LABELS) if not trend.empty else None
        plot_or_info(fig)

with tabs[6]:
    st.subheader("날씨와 버스 이용")
    if bus_weather.empty or weather_monthly.empty:
        st.info("월별 날씨와 버스 이용량 결합 데이터가 없습니다.")
    else:
        weather_vars = [col for col in WEATHER_LABELS if col in bus_weather.columns]
        selected_weather = st.selectbox("날씨 변수 선택", weather_vars, format_func=lambda key: WEATHER_LABELS.get(key, key))
        weather_target = st.selectbox("분석 대상", ["대구 전체"] + stop_options)
        if weather_target == "대구 전체":
            weather_view = bus_weather.copy()
        else:
            station_monthly = monthly_summary[monthly_summary["stop_name"] == weather_target]
            station_bus = (
                station_monthly.groupby(["year_month", "year", "month", "month_label"], dropna=False)
                .agg(boardings=("boardings", "sum"), alightings=("alightings", "sum"), total_users=("total_users", "sum"))
                .reset_index()
            )
            weather_view = station_bus.merge(weather_monthly, on=["year_month", "year", "month", "month_label"], how="inner")

        if len(weather_view) < 6:
            st.warning("관측 월 수가 적어 상관관계 해석의 신뢰성이 낮습니다.")
        corr_subset = weather_view[["boardings", selected_weather]].dropna()
        pearson = corr_subset["boardings"].corr(corr_subset[selected_weather], method="pearson") if len(corr_subset) >= 3 else np.nan
        spearman = spearman_without_scipy(corr_subset["boardings"], corr_subset[selected_weather]) if len(corr_subset) >= 3 else np.nan
        c1, c2, c3 = st.columns(3)
        c1.metric("Pearson 상관계수", "-" if pd.isna(pearson) else f"{pearson:.3f}")
        c2.metric("Spearman 상관계수", "-" if pd.isna(spearman) else f"{spearman:.3f}")
        c3.metric("상관 강도", interpret_correlation(pearson))
        st.caption("상관관계는 인과관계가 아닙니다. 예를 들어 '비 때문에 감소했다'가 아니라 '강수량이 높은 달에 이용량이 감소하는 경향이 관찰되었다'처럼 해석해야 합니다.")

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=weather_view["month_label"], y=weather_view["boardings"], mode="lines+markers", name="승차 인원"))
            fig.add_trace(go.Scatter(x=weather_view["month_label"], y=weather_view[selected_weather], mode="lines+markers", name=WEATHER_LABELS.get(selected_weather, selected_weather), yaxis="y2"))
            fig.update_layout(title="월별 버스 이용량과 날씨 추세", yaxis_title="승차 인원", yaxis2=dict(title=WEATHER_LABELS.get(selected_weather, selected_weather), overlaying="y", side="right"))
            plot_or_info(fig if not weather_view.empty else None)
        with col2:
            fig = px.scatter(weather_view, x=selected_weather, y="boardings", title=f"{WEATHER_LABELS.get(selected_weather, selected_weather)}와 승차 인원 산점도", labels={selected_weather: WEATHER_LABELS.get(selected_weather, selected_weather), "boardings": "승차 인원"}) if len(weather_view.dropna(subset=[selected_weather, "boardings"])) >= 2 else None
            plot_or_info(fig)

        if "month" in weather_view.columns:
            season_map = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄", 6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
            season = weather_view.assign(season=weather_view["month"].map(season_map))
            fig = px.box(season, x="season", y="boardings", title="계절별 버스 이용량 비교", labels={"season": "계절", "boardings": "승차 인원"}) if not season.empty else None
            plot_or_info(fig)

        corr_rows = []
        for variable in weather_vars:
            subset = weather_view[["boardings", variable]].dropna()
            if len(subset) >= 3:
                corr_rows.append({"날씨 변수": WEATHER_LABELS.get(variable, variable), "Pearson": subset["boardings"].corr(subset[variable]), "Spearman": spearman_without_scipy(subset["boardings"], subset[variable])})
        corr_table = pd.DataFrame(corr_rows)
        if not corr_table.empty:
            st.markdown("#### 날씨 변수별 상관관계")
            st.dataframe(corr_table, width="stretch")
            st.download_button("날씨 분석 결과 CSV 다운로드", data=corr_table.to_csv(index=False, encoding="utf-8-sig"), file_name="weather_correlation_filtered.csv", mime="text/csv")

with tabs[7]:
    st.subheader("데이터 및 분석 한계")
    limitations = [
        "하차 태그를 하지 않은 승객이 있을 수 있어 실제 하차 인원과 차이가 날 수 있습니다.",
        "현금 승차가 데이터에서 제외될 수 있습니다.",
        "경유 노선 수가 많다고 실제 배차 횟수가 많은 것은 아닙니다.",
        "버스 배차 간격과 차량 크기 데이터가 없습니다.",
        "이용객 수만으로 실제 차량 내부 혼잡도를 확정할 수 없습니다.",
        "특정 정류소의 이용량이 높은 이유는 주변 학교, 상권, 병원, 환승센터 등 추가 정보가 필요합니다.",
        "분석 결과는 노선 부족을 확정하는 것이 아니라 추가 검토 후보를 찾는 것입니다.",
        "데이터 수집 기간에 따라 계절성과 일시적 이벤트의 영향을 받을 수 있습니다.",
        "공휴일과 방학, 노선 개편, 지역 행사, 유가 변화, 코로나19 등의 외부 요인이 이용량에 영향을 줄 수 있습니다.",
        "월별 자료에서는 개별 강수일의 즉각적인 영향을 확인하기 어렵습니다.",
    ]
    for item in limitations:
        st.markdown(f"- {item}")

    st.markdown("#### 전처리 점검 요약")
    if data_check.empty:
        st.info("전처리 점검 파일이 없습니다.")
    else:
        visible_cols = [col for col in ["source_file", "dataset_type", "encoding", "rows", "columns", "status"] if col in data_check.columns]
        st.dataframe(data_check[visible_cols], width="stretch")

    if show_static:
        st.markdown("#### 저장된 정적 그래프")
        figure_dir = Path("outputs") / "figures"
        figures = sorted(figure_dir.glob("*.png"))
        if not figures:
            st.info("저장된 PNG 그래프가 없습니다. `python prepare_data.py`를 실행하면 생성됩니다.")
        else:
            for fig_path in figures:
                st.image(str(fig_path), caption=fig_path.name, width="stretch")

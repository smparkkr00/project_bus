from __future__ import annotations

from pathlib import Path

import pandas as pd

from .analysis import COMMUTE_HOURS, EVENING_HOURS
from .utils import FIGURE_DIR


def _load_plotting():
    """matplotlib/seaborn이 설치되어 있을 때만 정적 그래프를 만든다."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from matplotlib import font_manager, rcParams

        selected_font = None
        for font_path in [
            "C:/Windows/Fonts/malgun.ttf",
            "C:/Windows/Fonts/malgunbd.ttf",
            "C:/Windows/Fonts/NanumGothic.ttf",
        ]:
            if Path(font_path).exists():
                font_manager.fontManager.addfont(font_path)
                selected_font = font_manager.FontProperties(fname=font_path).get_name()
                rcParams["font.family"] = selected_font
                break
        else:
            font_names = [font.name for font in font_manager.fontManager.ttflist]
            for font_name in ["Malgun Gothic", "NanumGothic", "Noto Sans CJK KR", "AppleGothic"]:
                if font_name in font_names:
                    selected_font = font_name
                    rcParams["font.family"] = font_name
                    break
        rcParams["axes.unicode_minus"] = False
        sns.set_theme(style="whitegrid", font=selected_font)
        if selected_font:
            rcParams["font.family"] = selected_font
        return plt, sns
    except Exception as exc:
        print(f"[정적 그래프 건너뜀] matplotlib/seaborn을 불러올 수 없습니다: {exc}")
        return None, None


def _save_current(plt, filename: str) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def generate_static_figures(
    stop_summary: pd.DataFrame,
    hourly_summary: pd.DataFrame,
    monthly_summary: pd.DataFrame,
) -> None:
    """PPT 발표에 바로 넣을 수 있는 PNG 그래프를 저장한다."""
    plt, sns = _load_plotting()
    if plt is None or sns is None:
        return

    if not stop_summary.empty:
        stop_plot = stop_summary.copy()
        for col in ["total_boardings", "total_alightings", "route_count", "boardings_per_route", "commute_focus_pct"]:
            if col in stop_plot.columns:
                stop_plot[col] = pd.to_numeric(stop_plot[col], errors="coerce").astype(float)

        top = stop_plot.nlargest(10, "total_boardings")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top, y="stop_name", x="total_boardings", color="#2f6f9f")
        plt.title("정류소별 승차 인원 TOP 10")
        plt.xlabel("승차 인원")
        plt.ylabel("정류소")
        _save_current(plt, "01_top10_stop_boardings.png")

        district = stop_plot.groupby("district", dropna=False)["total_boardings"].sum().reset_index()
        plt.figure(figsize=(10, 5))
        sns.barplot(data=district.sort_values("total_boardings", ascending=False), x="district", y="total_boardings")
        plt.title("구·군별 승차 인원")
        plt.xlabel("구·군")
        plt.ylabel("승차 인원")
        plt.xticks(rotation=30)
        _save_current(plt, "02_district_boardings.png")

        plt.figure(figsize=(9, 5))
        scatter_data = stop_plot.dropna(subset=["route_count", "total_boardings"]).copy()
        scatter_data["boardings_per_route"] = scatter_data["boardings_per_route"].fillna(0.0)
        sns.scatterplot(
            data=scatter_data,
            x="route_count",
            y="total_boardings",
            size="boardings_per_route",
            hue="stop_type",
            sizes=(20, 250),
            alpha=0.75,
        )
        plt.title("경유 노선 수와 승차 인원")
        plt.xlabel("경유 노선 수")
        plt.ylabel("승차 인원")
        _save_current(plt, "05_route_count_scatter.png")

        top_load = stop_plot.nlargest(10, "boardings_per_route")
        plt.figure(figsize=(10, 6))
        sns.barplot(data=top_load, y="stop_name", x="boardings_per_route", color="#a45f3d")
        plt.title("노선당 승차 인원 TOP 10")
        plt.xlabel("노선당 승차 인원")
        plt.ylabel("정류소")
        _save_current(plt, "06_boardings_per_route_top10.png")

        plt.figure(figsize=(8, 5))
        sns.countplot(data=stop_plot, x="stop_type", order=stop_plot["stop_type"].value_counts().index)
        plt.title("정류소 유형별 개수")
        plt.xlabel("정류소 유형")
        plt.ylabel("정류소 수")
        _save_current(plt, "07_stop_type_count.png")

        commute_compare = stop_plot[stop_plot["stop_type"].isin(["출근형", "퇴근형", "출퇴근형"])]
        if not commute_compare.empty:
            plt.figure(figsize=(9, 5))
            sns.boxplot(data=commute_compare, x="stop_type", y="commute_focus_pct")
            plt.title("출근형·퇴근형 정류소 비교")
            plt.xlabel("정류소 유형")
            plt.ylabel("출근 시간 집중도(%)")
            _save_current(plt, "08_commute_evening_type_compare.png")

    if not hourly_summary.empty:
        by_hour = hourly_summary.groupby("hour", dropna=False)["boardings"].sum().reset_index()
        plt.figure(figsize=(10, 5))
        sns.lineplot(data=by_hour, x="hour", y="boardings", marker="o")
        plt.title("시간대별 전체 승차 인원")
        plt.xlabel("시간대")
        plt.ylabel("승차 인원")
        _save_current(plt, "03_hourly_boardings.png")

        top_names = stop_plot.nlargest(20, "total_boardings")["stop_name"].tolist() if not stop_plot.empty else []
        heat = hourly_summary[hourly_summary["stop_name"].isin(top_names)]
        if not heat.empty:
            pivot = heat.pivot_table(index="stop_name", columns="hour", values="boardings", aggfunc="sum", fill_value=0)
            pivot = pivot.apply(pd.to_numeric, errors="coerce").fillna(0).astype(float)
            plt.figure(figsize=(12, 8))
            sns.heatmap(pivot, cmap="YlGnBu")
            plt.title("정류소 × 시간대 승차 인원 히트맵")
            plt.xlabel("시간대")
            plt.ylabel("정류소")
            _save_current(plt, "04_stop_hour_heatmap.png")

    if not monthly_summary.empty:
        trend = monthly_summary.groupby("month_label", dropna=False)["boardings"].sum().reset_index()
        plt.figure(figsize=(12, 5))
        sns.lineplot(data=trend, x="month_label", y="boardings", marker="o")
        plt.title("월별 장기 승차 인원 추세")
        plt.xlabel("연월")
        plt.ylabel("승차 인원")
        plt.xticks(rotation=45)
        _save_current(plt, "09_monthly_trend.png")

        yoy = monthly_summary.dropna(subset=["yoy_rate"]).copy()
        yoy = yoy[(yoy["previous_year_boardings"] >= 1000) & (yoy["boardings"] >= 1000)]
        if not yoy.empty:
            latest_month = yoy["month_label"].max()
            latest = yoy[yoy["month_label"] == latest_month]
            change = pd.concat([latest.nlargest(10, "yoy_rate"), latest.nsmallest(10, "yoy_rate")])
            change["direction"] = change["yoy_rate"].apply(lambda value: "증가" if value > 0 else "감소")
            plt.figure(figsize=(11, 7))
            sns.barplot(data=change, y="stop_name", x="yoy_rate", hue="direction", dodge=False)
            plt.title(f"이용량 증가율 및 감소율 TOP 10 ({latest_month})")
            plt.xlabel("전년 동월 대비 증감률(%)")
            plt.ylabel("정류소")
            plt.legend([], [], frameon=False)
            _save_current(plt, "10_yoy_change_top10.png")

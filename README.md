# 대구 시내버스 정류소 이용 수요 및 노선 공급 불균형 분석

대구 시내버스 정류소별 월별 이용자 수, 시간대별 승하차 인원, 정류소 위치정보, 월별 날씨 자료를 결합해 정류소 이용 수요와 경유 노선 수의 상대적 불균형 후보를 탐색하는 Streamlit 대시보드입니다.

이 프로젝트는 원본 CSV를 Streamlit 앱에서 매번 처리하지 않습니다. 먼저 `prepare_data.py`로 전처리를 실행하고, `app.py`는 `outputs/processed`에 저장된 정제 CSV만 읽습니다.

## 프로젝트 목적

- 이용객이 많은 정류소와 구·군별 이용량을 확인합니다.
- 출근 시간과 퇴근 시간에 수요가 집중되는 정류소를 찾습니다.
- 경유 노선 수와 승차 인원의 관계를 탐색합니다.
- 이용 수요 대비 경유 노선 수가 상대적으로 적은 추가 검토 후보 정류소를 찾습니다.
- 월별 장기 자료로 전년 동월 대비 이용량 증가·감소 정류소를 확인합니다.
- 월별 날씨 지표와 버스 이용량 사이의 상관 경향을 살펴봅니다.

## 데이터 파일 준비

원본 CSV 파일은 `data/raw` 폴더 또는 그 하위 폴더에 넣습니다.

예상 데이터는 다음과 같습니다.

- 정류소별 월별 이용자 수 데이터
- 정류소별 시간대별 승하차 인원 데이터
- 정류소 위치정보 데이터
- 정류소별 경유 노선 수 또는 노선 정보 데이터
- 대구 날씨 데이터

CSV 인코딩은 `utf-8`, `utf-8-sig`, `cp949`, `euc-kr` 순서로 읽기를 시도합니다.

## 설치 방법

Windows PowerShell에서 아래 순서로 실행합니다.

```powershell
python -m venv venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 데이터 전처리

```powershell
python prepare_data.py
```

전처리 결과는 `outputs/processed`에 저장됩니다.

주요 산출물:

- `stop_summary.csv`
- `hourly_summary.csv`
- `monthly_summary.csv`
- `column_mapping.csv`
- `data_check_report.csv`
- `weather_monthly.csv`
- `weather_mapping.csv`
- `weather_audit.csv`
- `bus_weather_monthly_merged.csv`
- `weather_correlation_results.csv`
- `imbalance_candidates.csv`
- `analysis_summary.md`

PPT 발표용 정적 그래프는 `outputs/figures`에 PNG로 저장됩니다.

## Streamlit 실행

```powershell
streamlit run app.py
```

정제 CSV가 없으면 앱은 원본 CSV를 즉석 처리하지 않고 `python prepare_data.py`를 먼저 실행하라는 안내를 보여줍니다. 원본 CSV가 정제 CSV보다 최신이면 전처리를 다시 실행하라는 경고를 표시합니다.

## 프로젝트 폴더 구조

```text
project/
├─ app.py
├─ prepare_data.py
├─ requirements.txt
├─ README.md
├─ data/
│  └─ raw/
├─ src/
│  ├─ __init__.py
│  ├─ data_loader.py
│  ├─ preprocessing.py
│  ├─ column_mapper.py
│  ├─ analysis.py
│  ├─ visualization.py
│  ├─ weather_analysis.py
│  └─ utils.py
└─ outputs/
   ├─ figures/
   └─ processed/
```

## 주요 분석 내용

- 전체 현황: 전체 승차·하차 인원, 분석 정류소 수, 혼잡 시간대, TOP 정류소
- 시간대별 분석: 시간대별 승하차, 정류소·구군 히트맵, 출근·퇴근 시간 TOP 정류소
- 정류소별 분석: 선택 정류소의 요약 지표, 시간대 패턴, 월별 추세
- 수요·공급 불균형 분석: 경유 노선 수와 승차 인원 산점도, 노선당 승차 인원, 추가 검토 후보
- 지도 분석: 좌표가 있는 정류소의 공간 분포
- 장기 추세 분석: 선택 기준월의 전년 동월 대비 증가율·감소율
- 날씨와 버스 이용: 월별 날씨 지표와 버스 승차 인원의 상관 경향

## 데이터 한계

- 하차 태그를 하지 않은 승객이 있을 수 있어 실제 하차 인원과 차이가 날 수 있습니다.
- 현금 승차가 데이터에서 제외될 수 있습니다.
- 경유 노선 수가 많다고 실제 배차 횟수가 많은 것은 아닙니다.
- 버스 배차 간격과 차량 크기 데이터가 없습니다.
- 이용객 수만으로 실제 차량 내부 혼잡도를 확정할 수 없습니다.
- 특정 정류소의 이용량이 높은 이유는 주변 학교, 상권, 병원, 환승센터 등 추가 정보가 필요합니다.
- 분석 결과는 노선 부족을 확정하는 것이 아니라 추가 검토 후보를 찾는 것입니다.
- 데이터 수집 기간에 따라 계절성과 일시적 이벤트의 영향을 받을 수 있습니다.
- 공휴일과 방학, 노선 개편, 지역 행사, 유가 변화, 코로나19 등의 외부 요인이 이용량에 영향을 줄 수 있습니다.
- 월별 자료에서는 개별 강수일의 즉각적인 영향을 확인하기 어렵습니다.

## 발표 시 강조할 핵심 결과

전처리 후 `analysis_summary.md`에서 분석 기간, 정류소 수, 가장 이용객이 많은 정류소, 날씨 결합 가능 월 수를 먼저 확인하세요. 대시보드에서는 “노선 부족 확정”이 아니라 “추가 검토 후보 탐색”이라는 표현을 유지하는 것이 중요합니다.


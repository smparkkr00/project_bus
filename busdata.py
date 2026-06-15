import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

# ==========================================
# 0. 페이지 기본 설정 및 세션 상태 초기화
# ==========================================
st.set_page_config(page_title="대구 버스 데이터 분석", layout="wide")

# 분석 기준 데이터 정의
districts = ['군위군', '남구', '달서구', '달성군', '동구', '북구', '서구', '수성구', '중구']
months = [f"{i}월" for i in range(1, 13)]

# (1) 화면 전환용 page 상태 초기화
if 'page' not in st.session_state:
    st.session_state.page = 'home'

# (2) 구 선택 세션 상태 key 사전 등록
if 'all_districts' not in st.session_state:
    st.session_state['all_districts'] = False

for d in districts:
    if d not in st.session_state:
        st.session_state[d] = False

# (3) 월 선택 세션 상태 key 사전 등록
if 'all_months' not in st.session_state:
    st.session_state['all_months'] = False

for m in months:
    if m not in st.session_state:
        st.session_state[m] = False


# ==========================================
# 1. 콜백 함수 정의 (상태 변화 실시간 처리)
# ==========================================

# --- (A) 구 선택 관련 콜백 ---
def on_change_all_districts():
    # '구 전체선택' 상태를 모든 구에 일괄 적용
    for d in districts:
        st.session_state[d] = st.session_state['all_districts']

def on_change_district():
    # 모든 개별 구가 True인지 확인하여 전체선택에 반영
    all_checked = all(st.session_state[d] for d in districts)
    st.session_state['all_districts'] = all_checked


# --- (B) 월 선택 관련 콜백 ---
def on_change_all_months():
    # '월 전체선택' 상태를 1~12월에 일괄 적용
    for m in months:
        st.session_state[m] = st.session_state['all_months']

def on_change_month():
    # 모든 개별 월이 True인지 확인하여 전체선택에 반영
    all_checked = all(st.session_state[m] for m in months)
    st.session_state['all_months'] = all_checked


# ==========================================
# 2. 홈 화면 레이아웃
# ==========================================
if st.session_state.page == 'home':
    st.title("대구광역시 시내버스 이용 수요 분석 시스템")
    st.write("대구 시내버스 정류소별 데이터를 바탕으로 이용 수요 및 노선 공급 불균형을 분석하는 프로젝트입니다.")
    st.markdown("---")
    
    st.subheader("데이터 분석 메뉴")
    st.write("아래 버튼을 누르면 상세 이용현황 분석 페이지로 이동합니다.")
    
    if st.button("대구 월별 시내버스 이용현황 확인하기", use_container_width=True):
        st.session_state.page = 'status'
        st.rerun()

# ==========================================
# 3. 대구 월별 시내버스 이용현황 화면 레이아웃
# ==========================================
elif st.session_state.page == 'status':
    # 홈으로 돌아가는 버튼
    if st.button("메인 홈화면으로 돌아가기"):
        st.session_state.page = 'home'
        st.rerun()
        
    st.markdown("---")
    st.title("대구 월별 시내버스 이용현황")
    st.write("분석하고자 하는 지역과 월을 선택해 주세요. (데이터 연동은 다음 단계에서 진행됩니다.)")
    
    # ------------------------------------------
    # 카테고리 1: 구 선택
    # ------------------------------------------
    st.markdown("### 구 선택")
    
    # 구 전체선택 체크박스
    st.checkbox("구 전체선택", key="all_districts", on_change=on_change_all_districts)
    
    # 9개 구 개별 체크박스 (3열 배치)
    cols_d = st.columns(3)
    for i, d in enumerate(districts):
        with cols_d[i % 3]:
            st.checkbox(d, key=d, on_change=on_change_district)
            
    st.markdown("---")
    
    # ------------------------------------------
    # 카테고리 2: 월 선택
    # ------------------------------------------
    st.markdown("### 월 선택")
    
    # 월 전체선택 체크박스
    st.checkbox("월 전체선택", key="all_months", on_change=on_change_all_months)
    
    # 1~12월 개별 체크박스 (4열 배치)
    cols_m = st.columns(4)
    for i, m in enumerate(months):
        with cols_m[i % 4]:
            st.checkbox(m, key=m, on_change=on_change_month)
            
    st.markdown("---")
    
    # ------------------------------------------
    # 결과 확인용 실시간 리스트 추출
    # ------------------------------------------
    selected_districts = [d for d in districts if st.session_state[d]]
    selected_months = [m for m in months if st.session_state[m]]
    
    st.markdown("### 선택된 필터 조건")
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        if selected_districts:
            st.info(f"**선택된 구·군 ({len(selected_districts)}개):**\n{', '.join(selected_districts)}")
        else:
            st.warning("선택된 구·군이 없습니다.")
            
    with res_col2:
        if selected_months:
            st.success(f"**선택된 월 ({len(selected_months)}개):**\n{', '.join(selected_months)}")
        else:
            st.warning("선택된 월이 없습니다.")
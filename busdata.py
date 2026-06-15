import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st



st.title('대구 시내버스 이용현황')

bus_loc = pd.read_csv('./raw/대구광역시_시내버스 정류소 위치정보_20250903.csv')

st.write(bus_loc)
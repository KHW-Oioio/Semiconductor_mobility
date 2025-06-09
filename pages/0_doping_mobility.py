import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 페이지 제목
st.title("도핑 농도에 따른 반도체 전도도 변화")

# 사용자 입력
doping_values = st.slider("도핑 농도 (단위: x10¹⁶ /cm³)", 1, 100, 10)

# 계산 함수 (간단한 모델: σ = q * μ * N)
def calculate_conductivity(N, q=1.6e-19, mu=1350):
    return q * mu * N * 1e16  # N 입력을 x10¹⁶ 기준으로 환산

N_values = np.arange(1, doping_values + 1)
conductivities = calculate_conductivity(N_values)

# 데이터프레임 생성
data = pd.DataFrame({
    "도핑 농도 (x10¹⁶/cm³)": N_values,
    "전기 전도도 (S/cm)": conductivities
})

# 시각화
st.line_chart(data.set_index("도핑 농도 (x10¹⁶/cm³)"))

# 데이터표 출력
st.dataframe(data)


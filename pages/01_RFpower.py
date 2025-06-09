# streamlit_etching_app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import altair as alt

st.set_page_config(
    page_title="Semiconductor Etch Rate Simulator",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 사이드바: 사용자 입력 파라미터 설정 ---
st.sidebar.header("🔧 공정 파라미터 설정")

# 1.1 데이터 업로드
uploaded_file = st.sidebar.file_uploader("📥 CSV 파일 업로드 (RF_power, Temperature, Pressure, etch_rate 포함)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("샘플 데이터를 사용합니다.")
    np.random.seed(42)
    powers = np.linspace(50, 500, 20)
    temps = np.random.uniform(20, 80, size=20)       # 온도: 20~80도
    pressures = np.random.uniform(50, 200, size=20)  # 압력: 50~200 mTorr

    rates = (
        0.0008 * powers**2 
        - 0.3 * powers 
        + 0.2 * temps 
        - 0.1 * pressures 
        + 20 
        + np.random.normal(0, 5, len(powers))
    )
    df = pd.DataFrame({
        "RF_power": powers,
        "Temperature": temps,
        "Pressure": pressures,
        "etch_rate": rates
    })

# 1.2 회귀 차수 선택
poly_degree = st.sidebar.selectbox("🔢 다항 회귀 차수 선택", options=[1, 2, 3], index=1)

# 1.3 사용자 입력 슬라이더
user_power = st.sidebar.slider("⚡ 시뮬레이션 RF 전력 (W)", float(df.RF_power.min()), float(df.RF_power.max()), float(df.RF_power.mean()), step=1.0)
user_temp = st.sidebar.slider("🌡️ 온도 (°C)", float(df.Temperature.min()), float(df.Temperature.max()), float(df.Temperature.mean()), step=1.0)
user_pressure = st.sidebar.slider("🧪 압력 (mTorr)", float(df.Pressure.min()), float(df.Pressure.max()), float(df.Pressure.mean()), step=1.0)

# 1.4 총 식각 시간
etch_time = st.sidebar.number_input("⏱ 총 시뮬레이션 시간 (초)", min_value=10, max_value=600, value=60, step=10)

# --- 2. 데이터 & 모델 준비 ---
st.header("1️⃣ 데이터 확인 및 회귀 모델 학습")

with st.expander("▶️ Raw Data 보기"):
    st.dataframe(df.style.format({
        "RF_power": "{:.1f}",
        "Temperature": "{:.1f}",
        "Pressure": "{:.1f}",
        "etch_rate": "{:.2f}"
    }))

# 2.1 회귀 모델 학습
features = ["RF_power", "Temperature", "Pressure"]
X = df[features].values
y = df["etch_rate"].values

if poly_degree > 1:
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)

    def predict(rf, temp, pressure):
        xp = poly.transform([[rf, temp, pressure]])
        return model.predict(xp)[0]
else:
    model = LinearRegression().fit(X, y)

    def predict(rf, temp, pressure):
        return model.predict([[rf, temp, pressure]])[0]

# --- 모델 정보 출력 ---
coef = model.coef_
intercept = model.intercept_
st.markdown(f"- **모델 차수:** {poly_degree}차")
st.markdown(f"- **예측 결과:** {user_power:.1f} W / {user_temp:.1f}°C / {user_pressure:.1f} mTorr → **{predict(user_power, user_temp, user_pressure):.2f} Å/분**")

# --- 3. 시각화 (RF Power vs Etch Rate만 표시) ---
st.subheader("RF 전력 vs. 식각 속도 (기타 변수 고정)")

powers_line = np.linspace(df.RF_power.min(), df.RF_power.max(), 200)
pred_line = [predict(p, user_temp, user_pressure) for p in powers_line]

base = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X("RF_power", title="RF Power (W)"),
    y=alt.Y("etch_rate", title="Etch Rate (Å/min)")
)

line = alt.Chart(pd.DataFrame({
    "RF_power": powers_line,
    "pred_rate": pred_line
})).mark_line(color="red").encode(
    x="RF_power", y="pred_rate"
)

st.altair_chart(base + line, use_container_width=True)

# --- 4. 실시간 식각 시뮬레이션 ---
st.header("2️⃣ 실시간 식각 시뮬레이션")

etch_rate = predict(user_power, user_temp, user_pressure)
rate_per_sec = etch_rate / 60  # Å/sec
times = np.linspace(0, etch_time, etch_time + 1)
depths = rate_per_sec * times

st.write(f"> 총 **{etch_time}초** 동안, RF: **{user_power:.1f} W**, 온도: **{user_temp:.1f}°C**, 압력: **{user_pressure:.1f} mTorr** 조건에서 식각 속도 {etch_rate:.2f} Å/분으로 시뮬레이션합니다.")

progress_bar = st.progress(0)
etch_depth_text = st.empty()
chart_placeholder = st.empty()

sim_df = pd.DataFrame({
    "time": times,
    "depth": depths
})

for i, t in enumerate(times.astype(int)):
    percent = int((i / etch_time) * 100)
    progress_bar.progress(percent)
    etch_depth_text.markdown(f"**시간:** {t}s → **식각 깊이:** {depths[i]:.1f} Å")

    c = alt.Chart(sim_df.iloc[:i+1]).mark_line().encode(
        x="time",
        y="depth"
    )
    chart_placeholder.altair_chart(c, use_container_width=True)
    time.sleep(0.05)

st.success("✅ 시뮬레이션 완료!")

# --- 5. 모델 파라미터 & 다운로드 ---
st.header("3️⃣ 추가 기능")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔍 모델 파라미터")

    # intercept를 NaN으로 채워서 숫자 포맷 가능하게 함
    intercept_list = [intercept] + [np.nan] * (len(coef) - 1)
    params = pd.DataFrame({
        "계수(coef)": coef,
        "절편(intercept)": intercept_list
    }, index=[f"x{i+1}" for i in range(len(coef))])

    st.table(params.style.format("{:.4e}", subset=["계수(coef)", "절편(intercept)"]))

with col2:
    st.markdown("#### 📥 예측 결과 다운로드")
    out_df = pd.DataFrame({
        "RF_power": powers_line,
        "Temperature": [user_temp]*len(powers_line),
        "Pressure": [user_pressure]*len(powers_line),
        "predicted_etch_rate": pred_line
    })
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button("CSV로 다운로드", data=csv, file_name="predicted_etch_rates.csv", mime="text/csv")

st.markdown("---")
st.caption("© 2025 Semiconductor Etch Simulator")

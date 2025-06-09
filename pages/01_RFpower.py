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
uploaded_file = st.sidebar.file_uploader("📥 CSV 파일 업로드 (RF_power, etch_rate 컬럼 포함)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("샘플 데이터를 사용합니다.")
    np.random.seed(42)
    powers = np.linspace(50, 500, 20)
    rates = 0.0008 * powers**2 - 0.3 * powers + 20 + np.random.normal(0, 5, len(powers))
    df = pd.DataFrame({
        "RF_power": powers,
        "etch_rate": rates
    })

# 1.2 회귀모델 차수 선택
poly_degree = st.sidebar.selectbox(
    "🔢 다항 회귀 차수 선택",
    options=[1, 2, 3],
    index=1,
    help="RF 전력 vs. 식각 속도의 비선형성을 모델링할 차수를 선택하세요."
)

# 1.3 시뮬레이션용 RF 전력 슬라이더
user_power = st.sidebar.slider(
    "⚡ 시뮬레이션 RF 전력 (W)",
    min_value=float(df.RF_power.min()),
    max_value=float(df.RF_power.max()),
    value=float(df.RF_power.mean()),
    step=1.0
)

# 1.4 총 식각 시간 입력
etch_time = st.sidebar.number_input(
    "⏱ 총 시뮬레이션 시간 (초)",
    min_value=10, max_value=600, value=60, step=10
)

# 1.5 추가: 온도 입력
temperature = st.sidebar.number_input(
    "🌡 온도 (℃)",
    min_value=0.0, max_value=500.0, value=25.0, step=0.1,
    help="공정 온도를 입력하세요."
)

# 1.6 추가: 압력 입력
pressure = st.sidebar.number_input(
    "⚖ 압력 (Torr)",
    min_value=0.1, max_value=1000.0, value=10.0, step=0.1,
    help="공정 압력을 입력하세요."
)

# --- 2. 데이터 & 모델 준비 ---
st.header("1️⃣ 데이터 확인 및 회귀 모델 학습")

with st.expander("▶️ Raw Data 보기"):
    st.dataframe(df.style.format({"RF_power":"{:.1f}", "etch_rate":"{:.2f}"}))

# 2.1 회귀 모델 학습
X = df[["RF_power"]].values
y = df["etch_rate"].values

if poly_degree > 1:
    poly = PolynomialFeatures(degree=poly_degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    def predict(rate_power):
        xp = poly.transform([[rate_power]])
        return model.predict(xp)[0]
else:
    model = LinearRegression().fit(X, y)
    def predict(rate_power):
        return model.predict([[rate_power]])[0]

# 모델 파라미터 출력
coef = model.coef_
intercept = model.intercept_
st.markdown(f"- **모델 형태:** {' + '.join(f'{coef[i]:.4e}·RF^{i+1}' for i in range(len(coef)))} + {intercept:.2f}")
st.markdown(f"- **선택된 RF 전력:** {user_power:.1f} W → 예측 식각 속도: **{predict(user_power):.2f} Å/분**")

# 2.2 산점도 + 회귀 곡선 시각화
st.subheader("RF 전력 vs. 식각 속도")
base = alt.Chart(df).mark_circle(size=60, opacity=0.7).encode(
    x=alt.X("RF_power", title="RF Power (W)"),
    y=alt.Y("etch_rate", title="Etch Rate (Å/min)")
)

powers_line = np.linspace(df.RF_power.min(), df.RF_power.max(), 200)
pred_line = [predict(p) for p in powers_line]
line = alt.Chart(pd.DataFrame({
    "RF_power": powers_line,
    "pred_rate": pred_line
})).mark_line(color="red").encode(
    x="RF_power", y="pred_rate"
)

st.altair_chart(base + line, use_container_width=True)

# --- 3. 실시간 식각 프로세스 시뮬레이션 ---
st.header("2️⃣ 실시간 식각 프로세스 시뮬레이션")

st.write(f"> 총 **{etch_time}초** 동안, **{user_power:.1f} W**에서 예상 식각 속도({predict(user_power):.2f} Å/분)로 시뮬레이션합니다.")
st.write(f"> 온도: **{temperature:.1f} ℃**, 압력: **{pressure:.1f} Torr**")

progress_bar = st.progress(0)
etch_depth_text = st.empty()
chart_placeholder = st.empty()

# 시뮬레이션 계산
rate_per_sec = predict(user_power) / 60.0  # Å/초
times = np.linspace(0, etch_time, etch_time + 1)
depths = rate_per_sec * times

sim_df = pd.DataFrame({
    "time": times,
    "depth": depths
})

for i, t in enumerate(times.astype(int)):
    percent = int((i / etch_time) * 100)
    progress_bar.progress(percent)
    etch_depth_text.markdown(f"**시간:** {t}s → **식각 깊이:** {depths[i]:.1f} Å")
    
    c = alt.Chart(sim_df.iloc[:i+1]).mark_line().encode(
        x=alt.X("time", title="Time (s)"),
        y=alt.Y("depth", title="Etch Depth (Å)")
    )
    chart_placeholder.altair_chart(c, use_container_width=True)
    
    time.sleep(0.05)

st.success("✅ 시뮬레이션 완료!")

# 시뮬레이션 최종 결과 출력
st.markdown(f"### 📊 시뮬레이션 결과 요약")
st.write(f"- 총 시뮬레이션 시간: **{etch_time}초**")
st.write(f"- 최종 식각 깊이: **{depths[-1]:.1f} Å**")
st.write(f"- 사용 RF 전력: **{user_power:.1f} W**")
st.write(f"- 공정 온도: **{temperature:.1f} ℃**")
st.write(f"- 공정 압력: **{pressure:.1f} Torr**")

# --- 4. 추가 기능 & 다운로드 ---
st.header("3️⃣ 추가 기능")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 🔍 모델 파라미터")
    params = pd.DataFrame({
        "계수(coef)": coef,
        "절편(intercept)": [intercept] + [np.nan]*(len(coef)-1)
    }, index=[f"RF^{i+1}" for i in range(len(coef))])
    st.dataframe(params.style.format("{:.4e}", na_rep=" "))

with col2:
    st.markdown("#### 📥 예측 결과 다운로드")
    out_df = pd.DataFrame({
        "RF_power": powers_line,
        "predicted_etch_rate": pred_line
    })
    csv = out_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="CSV로 다운로드",
        data=csv,
        file_name="predicted_etch_rates.csv",
        mime="text/csv"
    )

st.markdown("---")
st.caption("© 2025 Semiconductor Etch Simulator")

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor, export_text
import altair as alt

st.set_page_config(
    page_title="KNN + 결정 트리 기반 식각 속도 예측 시뮬레이터",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 1. 사이드바: 사용자 입력 파라미터 설정 ---
st.sidebar.header("🔧 공정 파라미터 설정")

# 1.1 데이터 업로드
uploaded_file = st.sidebar.file_uploader("📥 CSV 파일 업로드 (RF_power, etch_rate 포함)", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if not {"Temperature", "Pressure"}.issubset(df.columns):
        np.random.seed(42)
        df["Temperature"] = np.random.uniform(20, 150, len(df))
        df["Pressure"] = np.random.uniform(0.5, 100, len(df))
else:
    st.sidebar.info("샘플 데이터를 사용합니다.")
    np.random.seed(42)
    powers = np.linspace(50, 500, 50)
    temps = np.random.uniform(20, 150, len(powers))
    pressures = np.random.uniform(0.5, 100, len(powers))
    rates = (0.0008 * powers**2 - 0.3 * powers + 20) + 0.05 * temps - 0.02 * pressures + np.random.normal(0, 3, len(powers))
    df = pd.DataFrame({
        "RF_power": powers,
        "Temperature": temps,
        "Pressure": pressures,
        "etch_rate": rates
    })

# 1.2 K 선택
k_value = st.sidebar.slider(
    "🔢 K 값 선택 (KNN 이웃 수)", min_value=1, max_value=10, value=3
)

# 1.3 시뮬레이션 파라미터
user_power = st.sidebar.slider(
    "⚡ 시뮬레이션 RF 전력 (W)", 
    min_value=float(df.RF_power.min()), 
    max_value=float(df.RF_power.max()), 
    value=float(df.RF_power.mean()), 
    step=1.0
)

col1, col2, col3 = st.sidebar.columns(3)
with col1:
    etch_time = st.slider("⏱ 시간 (초)", 30, 300, 60, step=10)
with col2:
    temperature = st.slider("🌡 온도 (℃, 건식 공정 기준)", 20.0, 150.0, 25.0, step=1.0)
with col3:
    pressure = st.slider("⚖ 압력 (mTorr, 고진공 공정 기준)", 0.5, 100.0, 10.0, step=0.5)

# --- 2. KNN 단일 변수 모델 학습 ---
st.header("1️⃣ KNN 기반 RF 전력 → 식각 속도 예측")

X_knn = df[["RF_power"]].values
y_knn = df["etch_rate"].values

knn_model = KNeighborsRegressor(n_neighbors=k_value)
knn_model.fit(X_knn, y_knn)

def predict_knn(power):
    return knn_model.predict([[power]])[0]

knn_pred_rate = predict_knn(user_power)

st.markdown(f"- 선택한 RF 전력: **{user_power:.1f} W** → 예측 식각 속도: **{knn_pred_rate:.2f} Å/분**")

# 시각화
powers_line = np.linspace(df.RF_power.min(), df.RF_power.max(), 200)
knn_pred_line = [predict_knn(p) for p in powers_line]

chart_knn = alt.Chart(df).mark_circle(size=60).encode(
    x="RF_power", y="etch_rate"
) + alt.Chart(pd.DataFrame({
    "RF_power": powers_line,
    "etch_rate": knn_pred_line
})).mark_line(color="red").encode(
    x="RF_power", y="etch_rate"
)

st.altair_chart(chart_knn, use_container_width=True)

# --- 3. 결정 트리 기반 다변량 최적 조건 탐색 ---
st.header("2️⃣ 결정 트리 기반 다변량 조건 최적화")

X_tree = df[["RF_power", "Temperature", "Pressure"]].values
y_tree = df["etch_rate"].values

tree_model = DecisionTreeRegressor(max_depth=4, random_state=42)
tree_model.fit(X_tree, y_tree)

power_range = np.linspace(df.RF_power.min(), df.RF_power.max(), 20)
temp_range = np.linspace(20, 150, 10)
pressure_range = np.linspace(0.5, 100, 10)

grid = np.array([[p, t, pr] for p in power_range for t in temp_range for pr in pressure_range])
preds = tree_model.predict(grid)

best_idx = np.argmax(preds)
best_condition = grid[best_idx]
best_rate = preds[best_idx]

st.markdown(f"**추천 최적 조건 (결정 트리 기준):**")
st.markdown(f"- RF 전력: **{best_condition[0]:.1f} W**")
st.markdown(f"- 온도: **{best_condition[1]:.1f} ℃ (건식 공정 기준)**")
st.markdown(f"- 압력: **{best_condition[2]:.1f} mTorr (고진공 공정 기준)**")
st.markdown(f"- 예측 식각 속도: **{best_rate:.2f} Å/분**")

st.subheader("결정 트리 규칙 미리보기")
rules = export_text(tree_model, feature_names=["RF_power", "Temperature", "Pressure"])
st.code(rules, language="plaintext")

# --- 4. 시뮬레이션 ---
st.header("3️⃣ 식각 공정 시뮬레이션 (KNN 기반)")

st.write(f"> 조건: RF 전력 **{user_power:.1f} W**, 온도 **{temperature:.1f} ℃**, 압력 **{pressure:.1f} mTorr**, 시간 **{etch_time}초**")

etch_rate_per_sec = knn_pred_rate / 60.0
times = np.linspace(0, etch_time, etch_time + 1)
depths = etch_rate_per_sec * times

sim_df = pd.DataFrame({
    "time": times,
    "depth": depths
})

progress_bar = st.progress(0)
etch_depth_text = st.empty()
chart_placeholder = st.empty()

for i, t in enumerate(times.astype(int)):
    percent = int((i / etch_time) * 100)
    progress_bar.progress(min(percent, 100))
    etch_depth_text.markdown(f"**시간:** {t}s → **식각 깊이:** {depths[i]:.1f} Å")
    
    c = alt.Chart(sim_df.iloc[:i+1]).mark_line().encode(
        x=alt.X("time", title="Time (s)"),
        y=alt.Y("depth", title="Etch Depth (Å)")
    )
    chart_placeholder.altair_chart(c, use_container_width=True)
    time.sleep(0.02)

st.success("✅ 시뮬레이션 완료!")

# --- 5. 결과 요약 ---
st.header("4️⃣ 결과 요약")

avg_rate = depths[-1] / etch_time
st.info(f"""
- 최종 식각 깊이: {depths[-1]:.2f} Å  
- 평균 식각 속도: {avg_rate:.2f} Å/초  
- RF 전력: {user_power:.1f} W  
- 온도: {temperature:.1f} ℃ (건식 공정 기준)  
- 압력: {pressure:.1f} mTorr (고진공 공정 기준)  
""")

# --- 6. 다운로드 ---
st.header("5️⃣ 예측 결과 다운로드")
out_df = pd.DataFrame({
    "RF_power": powers_line,
    "predicted_etch_rate": knn_pred_line
})
csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 CSV 다운로드", csv, "knn_predicted_etch_rates.csv", "text/csv")

st.caption("© 2025 RF Power-Based Etch Predictor using KNN + Decision Tree")

import streamlit as st
import pandas as pd
import numpy as np
import time
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import altair as alt

st.set_page_config(
    page_title="KNN + 결정트리 기반 식각 예측 시뮬레이터",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. 사이드바
st.sidebar.header("🔧 공정 파라미터 설정")

uploaded_file = st.sidebar.file_uploader("📥 CSV 업로드 (RF_power, etch_rate 포함)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    np.random.seed(42)
    powers = np.linspace(50, 500, 20)
    rates = 0.0008 * powers**2 - 0.3 * powers + 20 + np.random.normal(0, 5, len(powers))
    df = pd.DataFrame({"RF_power": powers, "etch_rate": rates})

k_value = st.sidebar.slider("🔢 KNN 이웃 수", 1, 10, 3)
user_power = st.sidebar.slider("⚡ RF 전력 (W)", float(df.RF_power.min()), float(df.RF_power.max()), float(df.RF_power.mean()), step=1.0)
col1, col2, col3 = st.sidebar.columns(3)
with col1:
    etch_time = st.slider("⏱ 시간 (초)", 30, 300, 60, step=10)
with col2:
    temperature = st.slider("🌡 온도 (℃, 건식 공정 기준)", 20.0, 150.0, 25.0, step=1.0)
with col3:
    pressure = st.slider("⚖ 압력 (mTorr, 고진공 공정 기준)", 0.5, 100.0, 10.0, step=0.5)

# 2. KNN 모델
st.header("1️⃣ KNN 예측 모델")
st.dataframe(df.style.format({"RF_power":"{:.1f}", "etch_rate":"{:.2f}"}))

X = df[["RF_power"]].values
y = df["etch_rate"].values
knn_model = KNeighborsRegressor(n_neighbors=k_value)
knn_model.fit(X, y)

def predict_knn(power):
    return knn_model.predict([[power]])[0]

st.markdown(f"- **K 값:** {k_value}, 선택 RF 전력: {user_power:.1f} W → 예측 식각 속도: **{predict_knn(user_power):.2f} Å/분**")

powers_line = np.linspace(df.RF_power.min(), df.RF_power.max(), 200)
pred_line = [predict_knn(p) for p in powers_line]
chart = alt.Chart(df).mark_circle(size=60).encode(
    x="RF_power", y="etch_rate"
) + alt.Chart(pd.DataFrame({
    "RF_power": powers_line,
    "pred_rate": pred_line
})).mark_line(color="red").encode(
    x="RF_power", y="pred_rate"
)
st.altair_chart(chart, use_container_width=True)

# 3. 기본 시뮬레이션
st.header("2️⃣ 사용자 조건 기반 시뮬레이션")

etch_rate_per_sec = predict_knn(user_power) / 60.0
times = np.linspace(0, etch_time, etch_time + 1)
depths = etch_rate_per_sec * times
sim_df = pd.DataFrame({"time": times, "depth": depths})

progress_bar = st.progress(0)
etch_depth_text = st.empty()
chart_placeholder = st.empty()

for i, t in enumerate(times.astype(int)):
    percent = int((i / etch_time) * 100)
    progress_bar.progress(min(percent, 100))
    etch_depth_text.markdown(f"**시간:** {t}s → **식각 깊이:** {depths[i]:.1f} Å")
    c = alt.Chart(sim_df.iloc[:i+1]).mark_line().encode(x="time", y="depth")
    chart_placeholder.altair_chart(c, use_container_width=True)
    time.sleep(0.01)

# 4. 결정 트리 최적 조건 탐색
st.header("3️⃣ 결정 트리 기반 최적 조건 탐색")

np.random.seed(1)
cond_data = pd.DataFrame({
    "RF_power": np.random.uniform(50, 500, 300),
    "temperature": np.random.uniform(20, 150, 300),
    "pressure": np.random.uniform(0.5, 100, 300),
})
cond_data["etch_rate"] = cond_data["RF_power"].apply(predict_knn)

tree_X = cond_data[["RF_power", "temperature", "pressure"]]
tree_y = cond_data["etch_rate"]
tree_model = DecisionTreeRegressor(max_depth=5)
tree_model.fit(tree_X, tree_y)

best_index = tree_y.idxmax()
best_condition = cond_data.loc[best_index, ["RF_power", "temperature", "pressure"]]
best_rate = tree_y[best_index]

st.markdown(f"""
- 📍 **최적 조건 탐색 결과**  
  - RF 전력: **{best_condition[0]:.1f} W**  
  - 온도: **{best_condition[1]:.1f} ℃**  
  - 압력: **{best_condition[2]:.1f} mTorr**  
  - 예측 식각 속도: **{best_rate:.2f} Å/분**
""")

# 5. 최적 조건으로 시뮬레이션
st.header("3️⃣-2 최적 조건 시뮬레이션")

if st.button("🧪 결정 트리 최적 조건으로 실행"):
    optimal_power = best_condition[0]
    optimal_temp = best_condition[1]
    optimal_pressure = best_condition[2]

    st.markdown(f"""
    > 최적 조건:  
    RF 전력: **{optimal_power:.1f} W**  
    온도: **{optimal_temp:.1f} ℃**  
    압력: **{optimal_pressure:.1f} mTorr**  
    시간: **{etch_time}초**
    """)

    optimal_rate = predict_knn(optimal_power)
    etch_rate_per_sec_opt = optimal_rate / 60.0
    times_opt = np.linspace(0, etch_time, etch_time + 1)
    depths_opt = etch_rate_per_sec_opt * times_opt
    sim_opt_df = pd.DataFrame({"time": times_opt, "depth": depths_opt})

    opt_chart = alt.Chart(sim_opt_df).mark_line(color="green").encode(
        x="time", y="depth"
    ).properties(title="📈 최적 조건 시뮬레이션 결과")

    st.altair_chart(opt_chart, use_container_width=True)

    st.info(f"""
    - 예측 식각 속도: {optimal_rate:.2f} Å/분  
    - 최종 식각 깊이: {depths_opt[-1]:.2f} Å  
    - 평균 속도: {depths_opt[-1]/etch_time:.2f} Å/초
    """)

# 6. 다운로드
st.header("4️⃣ 예측 결과 다운로드")
out_df = pd.DataFrame({"RF_power": powers_line, "predicted_etch_rate": pred_line})
csv = out_df.to_csv(index=False).encode("utf-8")
st.download_button("📥 CSV 다운로드", csv, "knn_predicted_etch_rates.csv", "text/csv")

st.caption("© 2025 KNN + Decision Tree Etch Rate Simulator")


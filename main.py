import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================================
# 기본 설정
# =========================================
st.set_page_config(
    page_title="서울 연평균 기온 선형회귀",
    page_icon="🌡️",
    layout="wide"
)

st.title("🌡️ 서울 연평균 기온 선형회귀 모델")
st.caption(
    "과거 기온 데이터를 이용해 선형회귀 모델을 학습하고 "
    "최근 20년의 기온을 얼마나 잘 예측하는지 비교합니다."
)


# =========================================
# 데이터 불러오기
# =========================================
URL = "https://raw.githubusercontent.com/greatsong/modudata/main/data/seoul.csv"


@st.cache_data
def load_data():
    df = pd.read_csv(URL)

    # 날짜 처리
    df["날짜"] = pd.to_datetime(df["날짜"], errors="coerce")

    # 평균기온 숫자로 변환
    df["평균기온"] = pd.to_numeric(df["평균기온"], errors="coerce")

    # 필요한 데이터만 남김
    df = df.dropna(subset=["날짜", "평균기온"])

    # 연도 만들기
    df["연도"] = df["날짜"].dt.year

    # 일별 자료라면 연평균 계산
    annual = (
        df.groupby("연도", as_index=False)["평균기온"]
        .mean()
        .rename(columns={"평균기온": "연평균기온"})
    )

    return annual


df = load_data()


# =========================================
# 연평균 데이터 확인
# =========================================
st.subheader("📊 연평균 기온 데이터")

st.write(
    f"사용 가능한 연도: **{int(df['연도'].min())}년 ~ {int(df['연도'].max())}년**"
)

st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
)


# =========================================
# 데이터 구간 설정
# =========================================
train_50 = df[
    (df["연도"] >= 1956) &
    (df["연도"] <= 2005)
].copy()

train_100 = df[
    (df["연도"] >= 1906) &
    (df["연도"] <= 2005)
].copy()

test = df[
    (df["연도"] >= 2006) &
    (df["연도"] <= 2025)
].copy()


# =========================================
# 데이터 개수 확인
# =========================================
st.subheader("📁 학습 데이터와 테스트 데이터")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "최근 50년 학습",
        f"{len(train_50)}년",
        "1956~2005"
    )

with col2:
    st.metric(
        "최근 100년 학습",
        f"{len(train_100)}년",
        "1906~2005"
    )

with col3:
    st.metric(
        "공통 테스트",
        f"{len(test)}년",
        "2006~2025"
    )


# =========================================
# 데이터 부족 확인
# =========================================
if len(train_50) == 0:
    st.error("1956~2005년 데이터를 찾을 수 없습니다.")
    st.stop()

if len(train_100) == 0:
    st.error("1906~2005년 데이터를 찾을 수 없습니다.")
    st.stop()

if len(test) == 0:
    st.error("2006~2025년 테스트 데이터를 찾을 수 없습니다.")
    st.stop()


# =========================================
# 선형회귀 함수
# =========================================
def make_model(train_data):
    X = train_data[["연도"]]
    y = train_data["연평균기온"]

    model = LinearRegression()
    model.fit(X, y)

    return model


# =========================================
# 모델 학습
# =========================================
model_50 = make_model(train_50)
model_100 = make_model(train_100)


# =========================================
# 테스트 데이터 예측
# =========================================
X_test = test[["연도"]]
y_test = test["연평균기온"]

pred_50 = model_50.predict(X_test)
pred_100 = model_100.predict(X_test)


# =========================================
# 평가 함수
# =========================================
def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return mae, mse, r2


mae_50, mse_50, r2_50 = evaluate_model(y_test, pred_50)
mae_100, mse_100, r2_100 = evaluate_model(y_test, pred_100)


# =========================================
# 회귀식
# =========================================
slope_50 = model_50.coef_[0]
intercept_50 = model_50.intercept_

slope_100 = model_100.coef_[0]
intercept_100 = model_100.intercept_


# =========================================
# 모델 기울기 비교
# =========================================
st.subheader("📈 50년 학습 vs 100년 학습 회귀선")

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🟦 최근 50년 모델")
    st.write(
        f"**회귀식:** "
        f"연평균기온 = {slope_50:.4f} × 연도 + {intercept_50:.2f}"
    )
    st.metric(
        "기울기",
        f"{slope_50:.4f} °C/년"
    )
    st.write(
        f"→ 10년당 약 **{slope_50 * 10:.2f}°C** 변화"
    )

with col2:
    st.markdown("### 🟩 최근 100년 모델")
    st.write(
        f"**회귀식:** "
        f"연평균기온 = {slope_100:.4f} × 연도 + {intercept_100:.2f}"
    )
    st.metric(
        "기울기",
        f"{slope_100:.4f} °C/년"
    )
    st.write(
        f"→ 10년당 약 **{slope_100 * 10:.2f}°C** 변화"
    )


# =========================================
# 회귀선 그래프
# =========================================
plot_df = df.copy()

plot_df["50년 회귀선"] = model_50.predict(
    plot_df[["연도"]]
)

plot_df["100년 회귀선"] = model_100.predict(
    plot_df[["연도"]]
)


fig1 = px.scatter(
    plot_df,
    x="연도",
    y="연평균기온",
    title="서울 연평균 기온과 50년·100년 회귀선",
    labels={
        "연도": "연도",
        "연평균기온": "연평균 기온 (°C)"
    }
)

fig1.add_scatter(
    x=plot_df["연도"],
    y=plot_df["50년 회귀선"],
    mode="lines",
    name="1956~2005 학습 회귀선"
)

fig1.add_scatter(
    x=plot_df["연도"],
    y=plot_df["100년 회귀선"],
    mode="lines",
    name="1906~2005 학습 회귀선"
)

fig1.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    fig1,
    use_container_width=True
)


# =========================================
# 테스트 데이터 예측 그래프
# =========================================
st.subheader("🎯 최근 20년 테스트 데이터 예측")

test_result = test.copy()

test_result["50년 모델 예측"] = pred_50
test_result["100년 모델 예측"] = pred_100

fig2 = px.line(
    test_result,
    x="연도",
    y=[
        "연평균기온",
        "50년 모델 예측",
        "100년 모델 예측"
    ],
    markers=True,
    title="2006~2025 실제 기온과 두 회귀모델의 예측 비교",
    labels={
        "연도": "연도",
        "value": "기온 (°C)",
        "variable": "구분"
    }
)

fig2.update_layout(
    hovermode="x unified"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)


# =========================================
# 성능 평가
# =========================================
st.subheader("📊 테스트 데이터 예측 성능")

comparison = pd.DataFrame({
    "모델": [
        "최근 50년 학습 (1956~2005)",
        "최근 100년 학습 (1906~2005)"
    ],
    "기울기 (°C/년)": [
        slope_50,
        slope_100
    ],
    "10년당 변화 (°C)": [
        slope_50 * 10,
        slope_100 * 10
    ],
    "MAE (°C)": [
        mae_50,
        mae_100
    ],
    "MSE (°C²)": [
        mse_50,
        mse_100
    ],
    "R²": [
        r2_50,
        r2_100
    ]
})

st.dataframe(
    comparison.style.format({
        "기울기 (°C/년)": "{:.4f}",
        "10년당 변화 (°C)": "{:.2f}",
        "MAE (°C)": "{:.3f}",
        "MSE (°C²)": "{:.3f}",
        "R²": "{:.3f}"
    }),
    use_container_width=True,
    hide_index=True
)


# =========================================
# 평가 지표 설명
# =========================================
st.subheader("🔎 평가 지표는 어떻게 읽을까?")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### MAE")
    st.write(
        "예측값과 실제값의 평균적인 차이입니다. "
        "값이 작을수록 예측이 좋습니다."
    )

with col2:
    st.markdown("### MSE")
    st.write(
        "예측 오차를 제곱해서 평균낸 값입니다. "
        "값이 작을수록 좋으며 큰 오차에 더 민감합니다."
    )

with col3:
    st.markdown("### R²")
    st.write(
        "모델이 실제 기온의 변동을 얼마나 설명하는지를 나타냅니다. "
        "일반적으로 1에 가까울수록 좋습니다."
    )


# =========================================
# 자동 비교 결과
# =========================================
st.subheader("📝 50년 모델과 100년 모델 비교")

if mae_50 < mae_100:
    better_mae = "50년 모델"
else:
    better_mae = "100년 모델"

if mse_50 < mse_100:
    better_mse = "50년 모델"
else:
    better_mse = "100년 모델"

if r2_50 > r2_100:
    better_r2 = "50년 모델"
else:
    better_r2 = "100년 모델"

st.write(
    f"""
**① 기울기 비교**

- 50년 모델의 기울기: **{slope_50:.4f} °C/년**
- 100년 모델의 기울기: **{slope_100:.4f} °C/년**
- 따라서 두 모델이 계산한 장기적인 기온 상승 추세가 얼마나 다른지 확인할 수 있습니다.

**② MAE 비교**

- 50년 모델: **{mae_50:.3f}°C**
- 100년 모델: **{mae_100:.3f}°C**
- MAE가 더 작은 모델: **{better_mae}**

**③ MSE 비교**

- 50년 모델: **{mse_50:.3f}**
- 100년 모델: **{mse_100:.3f}**
- MSE가 더 작은 모델: **{better_mse}**

**④ R² 비교**

- 50년 모델: **{r2_50:.3f}**
- 100년 모델: **{r2_100:.3f}**
- R²가 더 큰 모델: **{better_r2}**
"""
)


# =========================================
# 최종 해석
# =========================================
st.subheader("💡 결과 해석")

if slope_50 > slope_100:
    slope_comment = (
        "최근 50년을 이용한 회귀선의 기울기가 더 커서, "
        "최근 시기의 기온 상승 추세가 100년 전체를 이용했을 때보다 더 가파르게 나타납니다."
    )
elif slope_50 < slope_100:
    slope_comment = (
        "100년을 이용한 회귀선의 기울기가 더 커서, "
        "장기간의 데이터를 이용했을 때 계산되는 상승 추세가 더 가파르게 나타납니다."
    )
else:
    slope_comment = "두 모델의 기울기가 거의 같습니다."

if r2_50 > r2_100:
    performance_comment = (
        "최근 50년 모델이 공통 테스트 기간의 기온 변화를 더 잘 설명했습니다."
    )
elif r2_50 < r2_100:
    performance_comment = (
        "최근 100년 모델이 공통 테스트 기간의 기온 변화를 더 잘 설명했습니다."
    )
else:
    performance_comment = "두 모델의 테스트 성능이 거의 같습니다."

st.info(
    f"""
**결론**

{slope_comment}

공통 테스트 기간인 **2006~2025년**에 대해 평가한 결과,
MAE는 **{mae_50:.3f}°C vs {mae_100:.3f}°C**,
MSE는 **{mse_50:.3f} vs {mse_100:.3f}**,
R²는 **{r2_50:.3f} vs {r2_100:.3f}**입니다.

{performance_comment}

즉, 단순히 학습 데이터가 많다고 해서 반드시 최근 20년의 기온을 더 잘 예측하는 것은 아니며,
**어떤 기간의 데이터를 학습에 사용했는지에 따라 기울기와 예측 성능이 달라질 수 있습니다.**
"""
)

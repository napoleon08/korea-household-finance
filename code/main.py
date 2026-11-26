
import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path

# =====================================
# 1. PAGE SETTINGS
# =====================================

st.set_page_config(
    page_title="지역별 가계부채 분석",
    page_icon="📊",
    layout="wide"
)

st.title("📊 지역별 가계부채와 재정 부담 분석")
st.caption(
    "대한민국 17개 시도의 가계부채와 소득 비교 | 2024년 기준"
)

# =====================================
# 2. LOAD DATA
# =====================================

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "household_debt_income_2024_verified.csv"
)

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)

    required_columns = [
        "region",
        "household_debt_2024_03_만원",
        "household_income_2023_만원",
        "approx_DTI_percent"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"필요한 열이 없습니다: {missing_columns}"
        )

    for col in required_columns[1:]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[required_columns].copy()


try:
    raw_df = load_data()

except Exception as e:
    st.error("데이터를 불러오지 못했습니다.")
    st.code(str(e))
    st.stop()


# =====================================
# 3. DATA OVERVIEW
# =====================================

st.header("1. 데이터 개요")

st.write(
    "분석 대상은 대한민국 17개 시도이며, "
    "가구 부채와 소득 및 추정 DTI를 비교합니다."
)

col1, col2, col3 = st.columns(3)

col1.metric("원본 행 수", f"{len(raw_df)}개")
col2.metric("열 수", f"{len(raw_df.columns)}개")
col3.metric(
    "지역 수",
    f"{raw_df['region'].nunique()}개"
)

st.subheader("원본 데이터")

st.dataframe(raw_df, use_container_width=True)

st.subheader("자료형과 결측치")

overview = pd.DataFrame({
    "자료형": raw_df.dtypes.astype(str),
    "결측치 수": raw_df.isna().sum(),
    "결측치 비율(%)": (
        raw_df.isna().mean() * 100
    ).round(2)
})

st.dataframe(overview, use_container_width=True)


# =====================================
# 4. MISSING VALUE HANDLING
# =====================================

st.header("2. 결측치 확인 및 처리")

before_rows = len(raw_df)

analysis_columns = [
    "region",
    "household_debt_2024_03_만원",
    "household_income_2023_만원",
    "approx_DTI_percent"
]

missing_counts = raw_df[analysis_columns].isna().sum()

st.write("변수별 결측치 수")

st.dataframe(
    missing_counts.rename("결측치 수"),
    use_container_width=True
)

# Required values must exist for analysis.
df = raw_df.dropna(
    subset=analysis_columns
).copy()

after_missing_rows = len(df)

st.write(f"처리 전 행 수: {before_rows}")
st.write(f"처리 후 행 수: {after_missing_rows}")
st.write(
    f"제외된 행 수: {before_rows - after_missing_rows}"
)

# =====================================
# 5. OUTLIER DETECTION
# =====================================

st.header("3. 이상치 확인")

st.write(
    "IQR 방법을 사용하여 이상치 후보를 확인합니다. "
    "이상치 후보는 실제 지역적 특성일 수 있으므로 "
    "임의로 삭제하지 않습니다."
)

numeric_columns = [
    "household_debt_2024_03_만원",
    "household_income_2023_만원",
    "approx_DTI_percent"
]

outlier_summary = []

for col in numeric_columns:

    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)

    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    candidates = df[
        (df[col] < lower) |
        (df[col] > upper)
    ]

    outlier_summary.append({
        "변수": col,
        "하한": round(lower, 2),
        "상한": round(upper, 2),
        "이상치 후보 수": len(candidates),
        "후보 지역": (
            ", ".join(candidates["region"].tolist())
            if len(candidates) > 0
            else "없음"
        )
    })

outlier_df = pd.DataFrame(outlier_summary)

st.dataframe(
    outlier_df,
    use_container_width=True
)

st.info(
    "이상치 후보는 별도로 표시하고 삭제하지 않았습니다. "
    "17개 지역이라는 작은 표본에서 실제로 부채나 소득이 "
    "높은 지역을 삭제하면 분석 결과가 왜곡될 수 있기 때문입니다."
)

# =====================================
# 6. PREPROCESSING SUMMARY
# =====================================

st.header("4. 전처리 결과")

st.metric(
    "최종 분석 데이터 행 수",
    f"{len(df)}개"
)

st.write(f"원본 데이터: {before_rows}행")
st.write(f"결측치 처리 후: {after_missing_rows}행")
st.write(f"최종 데이터: {len(df)}행")

st.write(
    "이상치 후보는 확인했지만 삭제하지 않았습니다."
)

# =====================================
# 7. GRAPH 1
# =====================================

st.header("5. 지역별 가계부채 부담 비교")

fig1 = px.bar(
    df.sort_values(
        "approx_DTI_percent",
        ascending=False
    ),
    x="region",
    y="approx_DTI_percent",
    title="지역별 가계부채 대비 연간 소득 비율",
    labels={
        "region": "지역",
        "approx_DTI_percent": "추정 DTI (%)"
    },
    text="approx_DTI_percent"
)

fig1.update_traces(
    texttemplate="%{text:.1f}%",
    textposition="outside"
)

fig1.update_layout(
    xaxis_title="시도",
    yaxis_title="추정 DTI (%)",
    yaxis=dict(rangemode="tozero")
)

st.plotly_chart(
    fig1,
    use_container_width=True
)

# =====================================
# 8. GRAPH 2
# =====================================

st.header("6. 가계부채와 소득의 관계")

fig2 = px.scatter(
    df,
    x="household_income_2023_만원",
    y="household_debt_2024_03_만원",
    text="region",
    title="지역별 가구 소득과 가계부채의 관계",
    labels={
        "household_income_2023_만원":
            "연간 가구 소득 (만원)",
        "household_debt_2024_03_만원":
            "가구 부채 (만원)"
    }
)

fig2.update_traces(
    textposition="top center"
)

fig2.update_layout(
    xaxis_title="연간 가구 소득 (만원)",
    yaxis_title="가구 부채 (만원)"
)

st.plotly_chart(
    fig2,
    use_container_width=True
)

# =====================================
# 9. BASIC STATISTICS
# =====================================

st.header("7. 기초 통계")

st.dataframe(
    df[numeric_columns].describe().round(2),
    use_container_width=True
)

# =====================================
# 10. CORRELATION
# =====================================

st.header("8. 변수 간 상관관계")

corr = df[numeric_columns].corr()

fig3 = px.imshow(
    corr,
    text_auto=".2f",
    title="가계부채·소득·추정 DTI 상관관계",
    labels=dict(
        x="변수",
        y="변수",
        color="상관계수"
    ),
    aspect="auto",
    zmin=-1,
    zmax=1
)

st.plotly_chart(
    fig3,
    use_container_width=True
)

st.caption(
    "상관관계는 변수 간 관계를 나타내며 "
    "인과관계를 의미하지 않습니다."
)

# =====================================
# 11. CONCLUSION
# =====================================

st.header("9. 분석 시 주의할 점")

st.write("""
- 분석 대상은 17개 시도이며 표본 수가 적습니다.
- 부채는 2024년 3월 기준이고 소득은 2023년 기준입니다.
- 추정 DTI는 지역별 평균 부채와 평균 소득을 이용한
  집계 수준의 근사 지표입니다.
- 개인별 공식 DTI 또는 DSR과 동일하지 않습니다.
- 이 데이터만으로 생활비 전체를 평가할 수 없습니다.
- 이상치 후보를 삭제하지 않았으므로 실제 지역 특성을
  유지한 상태에서 분석합니다.
""")

st.success("기초 데이터 탐색 및 시각화가 완료되었습니다.")

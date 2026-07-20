import altair as alt
import pandas as pd
import streamlit as st

from analyze_sales import summarize_dataframe, validate_and_prepare_dataframe

PRIMARY_COLOR = "#0E7490"
ACCENT_COLOR = "#F59E0B"


def format_amount(series: pd.Series) -> pd.Series:
    return series.map(lambda value: f"{value:,.0f}")


def to_department_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("department", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"department": "부서", "amount": "총액"})
    )
    grouped["총액"] = format_amount(grouped["총액"])
    return grouped


def to_department_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("department", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"department": "부서", "amount": "총액"})
    )


def to_item_table(df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        df.groupby("item", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"item": "항목", "amount": "총액"})
    )
    grouped["총액"] = format_amount(grouped["총액"])
    return grouped


def to_item_chart_data(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("item", as_index=False)["amount"]
        .sum()
        .sort_values("amount", ascending=False)
        .rename(columns={"item": "항목", "amount": "총액"})
    )


def to_grand_total_table(total: float) -> pd.DataFrame:
    return pd.DataFrame({"구분": ["전체 총액"], "총액": [f"{total:,.0f}"]})


def apply_theme() -> None:
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #f7fbfc 0%, #eef6f9 100%);
            }}
            .block-container {{
                padding-top: 2rem;
            }}
            h1, h2, h3 {{
                color: {PRIMARY_COLOR};
                letter-spacing: 0.01em;
            }}
            .stCaption {{
                color: #0f172a;
                font-weight: 600;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sorted_bar_chart(data: pd.DataFrame, category_col: str, color: str, show_labels: bool) -> None:
    base = alt.Chart(data).encode(
        x=alt.X("총액:Q", title="총액"),
        y=alt.Y(f"{category_col}:N", sort="-x", title=None),
        tooltip=[
            alt.Tooltip(f"{category_col}:N", title=category_col),
            alt.Tooltip("총액:Q", title="총액", format=",.0f"),
        ],
    )

    bars = base.mark_bar(cornerRadiusTopRight=6, cornerRadiusBottomRight=6, color=color)

    if show_labels:
        labels = base.mark_text(
            align="left",
            baseline="middle",
            dx=4,
            color="#0f172a",
            fontWeight="bold",
        ).encode(text=alt.Text("총액:Q", format=",.0f"))
        chart = (bars + labels).properties(height=320)
    else:
        chart = bars.properties(height=320)

    st.altair_chart(chart, use_container_width=True)


def toggle_all_departments(departments: list[str]) -> None:
    should_select_all = st.session_state.get("select_all_departments", True)
    for department in departments:
        st.session_state[f"department_filter_{department}"] = should_select_all


def main() -> None:
    st.set_page_config(page_title="Sales Dashboard", layout="wide")
    apply_theme()
    st.title("CSV 매출 분석 대시보드")
    st.write("CSV 파일을 업로드하면 전체/부서별/항목별 총액을 확인할 수 있습니다.")

    uploaded_file = st.file_uploader("CSV 파일 업로드", type=["csv"])

    if uploaded_file is None:
        st.info("분석할 CSV 파일을 업로드해주세요.")
        return

    try:
        raw_df = pd.read_csv(uploaded_file)
        prepared_df = validate_and_prepare_dataframe(raw_df)
    except Exception as exc:
        st.error(f"업로드한 CSV를 처리할 수 없습니다: {exc}")
        st.stop()

    departments = sorted(prepared_df["department"].dropna().astype(str).unique().tolist())

    st.subheader("부서 필터")
    st.caption("체크한 부서만 집계에 포함됩니다.")
    selected_departments: list[str] = []

    if "select_all_departments" not in st.session_state:
        st.session_state["select_all_departments"] = True

    for department in departments:
        key = f"department_filter_{department}"
        if key not in st.session_state:
            st.session_state[key] = True

    st.checkbox(
        "전체 부서 선택",
        key="select_all_departments",
        on_change=toggle_all_departments,
        args=(departments,),
    )

    column_count = min(3, max(1, len(departments)))
    dept_columns = st.columns(column_count)
    for idx, department in enumerate(departments):
        is_checked = dept_columns[idx % column_count].checkbox(
            department,
            key=f"department_filter_{department}",
        )
        if is_checked:
            selected_departments.append(department)

    if selected_departments:
        filtered_df = prepared_df[prepared_df["department"].astype(str).isin(selected_departments)]
    else:
        filtered_df = prepared_df.iloc[0:0].copy()
        st.warning("선택된 부서가 없습니다. 부서를 1개 이상 체크해주세요.")

    with st.expander("차트 옵션", expanded=False):
        show_labels = st.checkbox("막대 값 라벨 표시", value=True)
        use_top_n = st.checkbox("상위 N개만 표시", value=False)
        top_n = st.slider("N 값", min_value=3, max_value=20, value=10, step=1, disabled=not use_top_n)

    department_totals, item_totals, grand_total = summarize_dataframe(filtered_df)

    st.subheader("전체 총액")
    st.table(to_grand_total_table(grand_total))

    st.subheader("부서별 총액")
    if department_totals:
        st.table(to_department_table(filtered_df))
        st.caption("부서별 총액 막대 차트")
        department_chart_data = to_department_chart_data(filtered_df)
        if use_top_n:
            department_chart_data = department_chart_data.head(top_n)
        render_sorted_bar_chart(department_chart_data, "부서", PRIMARY_COLOR, show_labels)
    else:
        st.info("표시할 부서별 데이터가 없습니다.")

    st.subheader("항목별 총액")
    if item_totals:
        st.table(to_item_table(filtered_df))
        st.caption("항목별 총액 막대 차트")
        item_chart_data = to_item_chart_data(filtered_df)
        if use_top_n:
            item_chart_data = item_chart_data.head(top_n)
        render_sorted_bar_chart(item_chart_data, "항목", ACCENT_COLOR, show_labels)
    else:
        st.info("표시할 항목별 데이터가 없습니다.")


if __name__ == "__main__":
    main()

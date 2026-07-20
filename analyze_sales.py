import argparse
import csv
import sys
from collections import defaultdict

REQUIRED_COLUMNS = {"date", "department", "item", "amount"}


def format_money(value: float) -> str:
    return f"{value:,.0f}"


def print_report(department_totals: dict[str, float], item_totals: dict[str, float], grand_total: float) -> None:
    print("=== 부서별 총액 ===")
    for dept, total in sorted(department_totals.items()):
        print(f"- {dept}: {format_money(total)}")

    print("\n=== 항목별 총액 ===")
    for item, total in sorted(item_totals.items()):
        print(f"- {item}: {format_money(total)}")

    print("\n=== 전체 총액 ===")
    print(f"- {format_money(grand_total)}")


def validate_and_prepare_dataframe(df):
    import pandas as pd

    missing_columns = REQUIRED_COLUMNS - set(df.columns)
    if missing_columns:
        missing_text = ", ".join(sorted(missing_columns))
        raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_text}")

    numeric_amount = pd.to_numeric(df["amount"], errors="coerce")
    invalid_rows = df[numeric_amount.isna()]
    if not invalid_rows.empty:
        invalid_values = invalid_rows["amount"].astype(str).head(3).tolist()
        sample_text = ", ".join(invalid_values)
        raise ValueError(
            "amount 컬럼에 숫자가 아닌 값이 있습니다. "
            f"예시 값: {sample_text}"
        )

    prepared_df = df.copy()
    prepared_df["amount"] = numeric_amount
    return prepared_df


def summarize_dataframe(df):
    department_totals = df.groupby("department")["amount"].sum().to_dict()
    item_totals = df.groupby("item")["amount"].sum().to_dict()
    grand_total = float(df["amount"].sum())
    return department_totals, item_totals, grand_total


def analyze_with_pandas(input_path: str):
    import pandas as pd

    try:
        df = pd.read_csv(input_path)
    except FileNotFoundError:
        raise ValueError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    except Exception as exc:
        raise ValueError(f"CSV 파일을 읽는 중 오류가 발생했습니다: {exc}") from exc

    prepared_df = validate_and_prepare_dataframe(df)
    return summarize_dataframe(prepared_df)


def analyze_with_csv(input_path: str):
    department_totals: dict[str, float] = defaultdict(float)
    item_totals: dict[str, float] = defaultdict(float)
    grand_total = 0.0

    try:
        with open(input_path, newline="", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            if reader.fieldnames is None:
                raise ValueError("CSV 헤더를 찾을 수 없습니다.")

            missing_columns = REQUIRED_COLUMNS - set(reader.fieldnames)
            if missing_columns:
                missing_text = ", ".join(sorted(missing_columns))
                raise ValueError(f"필수 컬럼이 누락되었습니다: {missing_text}")

            for row_index, row in enumerate(reader, start=2):
                raw_amount = (row.get("amount") or "").strip()
                try:
                    amount = float(raw_amount)
                except ValueError as exc:
                    raise ValueError(
                        "amount 컬럼에 숫자가 아닌 값이 있습니다. "
                        f"문제 행: {row_index}, 값: '{raw_amount}'"
                    ) from exc

                department = (row.get("department") or "").strip()
                item = (row.get("item") or "").strip()

                department_totals[department] += amount
                item_totals[item] += amount
                grand_total += amount

    except FileNotFoundError:
        raise ValueError(f"입력 파일을 찾을 수 없습니다: {input_path}")
    except csv.Error as exc:
        raise ValueError(f"CSV 파싱 중 오류가 발생했습니다: {exc}") from exc

    return dict(department_totals), dict(item_totals), grand_total


def main() -> int:
    parser = argparse.ArgumentParser(description="sales.csv 매출 데이터를 집계합니다.")
    parser.add_argument("--input", required=True, help="입력 CSV 파일 경로")
    args = parser.parse_args()

    try:
        try:
            department_totals, item_totals, grand_total = analyze_with_pandas(args.input)
            backend = "pandas"
        except ModuleNotFoundError:
            department_totals, item_totals, grand_total = analyze_with_csv(args.input)
            backend = "csv"

        print(f"[INFO] 처리 엔진: {backend}")
        print_report(department_totals, item_totals, grand_total)
        return 0

    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())

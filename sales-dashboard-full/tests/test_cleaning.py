import pandas as pd
import pytest

from helpers import raw_sales
from salesdash.cleaning import MissingColumnsError, clean_sales_data


def test_removes_exact_duplicates():
    df = raw_sales()
    df = pd.concat([df, df.head(5)], ignore_index=True)
    res = clean_sales_data(df)
    assert res.report.duplicates_removed == 5
    assert len(res.sales) == len(df) - 5


def test_negative_quantities_become_returns():
    df = raw_sales()
    df.loc[0, ["quantity", "sales_amount"]] = [-2, -50.0]
    res = clean_sales_data(df)
    assert len(res.returns) == 1
    assert (res.sales["quantity"] > 0).all()
    assert res.report.returns_separated == 1


def test_missing_required_column_names_the_column():  # TC-02
    df = raw_sales().drop(columns=["category", "quantity"])
    with pytest.raises(MissingColumnsError) as exc:
        clean_sales_data(df)
    assert set(exc.value.missing) == {"category", "quantity"}
    assert "category" in str(exc.value)


def test_rows_with_missing_date_or_amount_are_dropped():
    df = raw_sales()
    df.loc[0, "order_date"] = None
    df.loc[1, "sales_amount"] = None
    res = clean_sales_data(df)
    assert res.report.invalid_rows_dropped == 2


def test_real_superstore_headers_are_mapped_and_unit_price_derived():
    df = pd.DataFrame({
        "Order ID": ["A1", "A2"], "Order Date": ["11/8/2016", "11/9/2016"],
        "Product ID": ["P1", "P2"], "Category": ["furniture ", "TECHNOLOGY"],
        "Sub-Category": ["Chairs", "Phones"], "Segment": ["Consumer", "Corporate"],
        "Ship Mode": ["Second Class", "Standard Class"],
        "Sales": [262.0, 731.9], "Quantity": [2, 3], "Discount": [0.0, 0.2], "Profit": [41.9, 219.6],
    })
    res = clean_sales_data(df)
    s = res.sales
    assert {"sales_amount", "sub_category", "customer_segment", "shipping_mode"} <= set(s.columns)
    assert set(s["category"]) == {"Furniture", "Technology"}          # trimmed and title-cased
    assert s["order_date"].iloc[0] == pd.Timestamp("2016-11-08")      # month-first by default
    assert res.report.unit_price_derived == 2
    assert s.loc[s["order_id"] == "A1", "unit_price"].iloc[0] == pytest.approx(131.0)


def test_missing_optional_values_are_filled_and_reported():
    df = raw_sales()
    df.loc[:3, "region"] = None
    res = clean_sales_data(df)
    assert (res.sales["region"] == "Unknown").sum() == 4
    assert res.report.missing_values_filled["region"] == 4


def test_outlier_flag_and_cap():
    df = raw_sales()
    df["category"] = "Furniture"
    df.loc[0, ["sales_amount", "quantity"]] = [1_000_000.0, 1]
    flagged = clean_sales_data(df, outlier_action="flag")
    assert flagged.sales["is_outlier"].sum() >= 1
    assert flagged.sales["sales_amount"].max() == 1_000_000.0          # flag does not change values
    capped = clean_sales_data(df, outlier_action="cap")
    assert capped.sales["sales_amount"].max() < 1_000_000.0


def test_invalid_outlier_action_rejected():
    with pytest.raises(ValueError):
        clean_sales_data(raw_sales(), outlier_action="delete")

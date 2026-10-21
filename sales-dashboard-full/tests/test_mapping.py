import os
import tempfile

import pandas as pd
import pytest

from helpers import raw_sales
from salesdash.cleaning import MissingColumnsError, clean_sales_data, load_and_clean_csv, read_csv_flexible
from salesdash.mapping import apply_mapping, suggest_mapping

SUPERSTORE = pd.DataFrame({
    "Row ID": [1, 2], "Order ID": ["A", "B"], "Order Date": ["11/8/2016", "11/9/2016"],
    "Product ID": ["P1", "P2"], "Product Name": ["Chair", "Phone"], "Category": ["Furniture", "Technology"],
    "Sub-Category": ["Chairs", "Phones"], "Segment": ["Consumer", "Corporate"], "Region": ["West", "East"],
    "Sales": [100.0, 500.0], "Quantity": [1, 2], "Discount": [0.0, 0.1], "Profit": [10.0, 50.0],
})


def test_suggest_mapping_recognises_superstore_headers():
    m = suggest_mapping(list(SUPERSTORE.columns))
    assert m["sales_amount"] == "Sales"
    assert m["order_date"] == "Order Date"
    assert m["sub_category"] == "Sub-Category"
    assert m["customer_segment"] == "Segment"
    assert m["shipping_mode"] is None            # not in this file


def test_apply_mapping_then_clean_end_to_end():
    m = suggest_mapping(list(SUPERSTORE.columns))
    res = clean_sales_data(apply_mapping(SUPERSTORE, m))
    assert len(res.sales) == 2
    assert "Row ID" not in res.sales.columns     # unmapped columns are dropped
    assert res.sales["sales_amount"].sum() == pytest.approx(600.0)


def test_user_override_for_unrecognised_header():
    df = SUPERSTORE.rename(columns={"Sales": "Net Turnover (USD)"})
    m = suggest_mapping(list(df.columns))
    assert m["sales_amount"] is None
    with pytest.raises(MissingColumnsError):
        apply_mapping(df, m)
    m["sales_amount"] = "Net Turnover (USD)"
    assert "sales_amount" in apply_mapping(df, m).columns


def test_missing_required_names_every_field():
    m = suggest_mapping(["Order ID", "Order Date"])
    with pytest.raises(MissingColumnsError) as exc:
        apply_mapping(pd.DataFrame(columns=["Order ID", "Order Date"]), m)
    assert {"product_id", "category", "quantity", "sales_amount"} <= set(exc.value.missing)


def test_same_source_column_cannot_feed_two_fields():
    m = suggest_mapping(list(SUPERSTORE.columns))
    m["profit"] = "Sales"
    with pytest.raises(ValueError, match="more than one field"):
        apply_mapping(SUPERSTORE, m)


def test_mapped_column_absent_from_file_is_rejected():
    m = suggest_mapping(list(SUPERSTORE.columns))
    m["region"] = "Ghost"
    with pytest.raises(ValueError, match="not found"):
        apply_mapping(SUPERSTORE, m)


def test_csv_reader_falls_back_to_latin1():
    df = raw_sales(n_days=5)
    df["product_name"] = "Caf\u00e9 Chair"
    data = df.to_csv(index=False).encode("latin-1")        # not valid utf-8
    out = read_csv_flexible(data)
    assert out["product_name"].iloc[0] == "Caf\u00e9 Chair"
    fd, path = tempfile.mkstemp(suffix=".csv")
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        assert len(load_and_clean_csv(path).sales) == len(df)
    finally:
        os.remove(path)

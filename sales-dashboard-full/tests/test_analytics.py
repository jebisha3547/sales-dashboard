import pytest

from helpers import raw_sales
from salesdash import analytics as A
from salesdash import db
from salesdash.cleaning import clean_sales_data


def make_conn(n_days=120):
    res = clean_sales_data(raw_sales(n_days=n_days))
    conn = db.connect()
    db.load_transactions(conn, res.sales, res.returns)
    return conn, res.sales


def test_kpis_match_pandas():
    conn, sales = make_conn()
    k = A.kpis(conn)["current"]
    assert k["revenue"] == pytest.approx(sales["sales_amount"].sum())
    assert k["profit"] == pytest.approx(sales["profit"].sum())
    assert k["units"] == sales["quantity"].sum()
    assert k["orders"] == sales["order_id"].nunique()
    assert k["avg_order_value"] == pytest.approx(k["revenue"] / k["orders"])


def test_filters_apply_consistently():  # TC-03
    conn, sales = make_conn()
    f = A.Filters(start="2023-02-01", end="2023-03-15", categories=("Furniture",), regions=("West",))
    want = sales[(sales["order_date"] >= "2023-02-01") & (sales["order_date"] <= "2023-03-15")
                 & (sales["category"] == "Furniture") & (sales["region"] == "West")]
    assert A.kpis(conn, f)["current"]["revenue"] == pytest.approx(want["sales_amount"].sum())
    assert A.trends(conn, "M", f)["revenue"].sum() == pytest.approx(want["sales_amount"].sum())
    assert A.breakdown(conn, "category", f)["revenue"].sum() == pytest.approx(want["sales_amount"].sum())


def test_growth_compares_with_previous_equal_window():
    conn, sales = make_conn()
    f = A.Filters(start="2023-03-01", end="2023-03-31")
    k = A.kpis(conn, f)
    assert k["previous_period"] == ("2023-01-29", "2023-02-28")
    prev = sales[(sales["order_date"] >= "2023-01-29") & (sales["order_date"] <= "2023-02-28")]["sales_amount"].sum()
    cur = k["current"]["revenue"]
    assert k["growth_pct"]["revenue"] == pytest.approx((cur - prev) / prev * 100)


def test_no_growth_when_no_prior_data():
    conn, _ = make_conn()
    assert A.kpis(conn)["growth_pct"] is None


def test_aggregate_tables_sum_to_total():
    conn, sales = make_conn()
    total = sales["sales_amount"].sum()
    for table in ("sales_daily", "sales_weekly", "sales_monthly"):
        assert conn.execute(f"SELECT SUM(revenue) FROM {table}").fetchone()[0] == pytest.approx(total)


def test_weekly_periods_start_on_monday():
    conn, _ = make_conn()
    weeks = A.trends(conn, "W")["period"]
    assert (weeks.dt.dayofweek == 0).all()


def test_dimension_and_freq_whitelists_block_injection():
    conn, _ = make_conn()
    with pytest.raises(ValueError):
        A.breakdown(conn, "category; DROP TABLE sales_transactions")
    with pytest.raises(ValueError):
        A.trends(conn, "X")
    assert conn.execute("SELECT COUNT(*) FROM sales_transactions").fetchone()[0] > 0


def test_filter_values_are_parameterised():
    conn, _ = make_conn()
    f = A.Filters(categories=("Furniture' OR '1'='1",))
    assert A.kpis(conn, f)["current"]["revenue"] == 0


def test_abc_classes_cover_all_revenue_and_start_with_a():
    conn, _ = make_conn()
    abc = A.abc_analysis(conn, dimension="product_id")
    assert abc["share_pct"].sum() == pytest.approx(100.0)
    assert abc.iloc[0]["abc_class"] == "A"
    assert abc["cum_share_pct"].is_monotonic_increasing


def test_new_vs_returning_counts_customers_once_as_new():
    conn, sales = make_conn()
    nv = A.new_vs_returning(conn)
    assert nv["new_customers"].sum() == sales["customer_id"].nunique()

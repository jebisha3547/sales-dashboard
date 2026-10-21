"""Plotly figure builders (colour-blind-friendly Okabe-Ito palette, shared clean styling)."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

BLUE, ORANGE, GREEN, RED, SKY, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7"
PALETTE = [BLUE, ORANGE, GREEN, RED, SKY, PURPLE, "#F0E442", "#999999"]
GRID = "rgba(128, 128, 128, 0.18)"      # neutral grey, readable on light and dark backgrounds
MONEY = {"revenue", "profit"}           # metrics shown with a $ prefix


def _style(fig: go.Figure, title: str = "", height: int = 360) -> go.Figure:
    """Shared look: left-aligned title, transparent background, light grid, tidy margins."""
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=15)) if title else None,
        height=height,
        margin=dict(l=10, r=10, t=50 if title else 20, b=10),
        legend=dict(orientation="h", y=-0.18, x=0),
        hovermode="x unified",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        hoverlabel=dict(font_size=13),
    )
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def _value_axis(fig: go.Figure, axis: str, is_money: bool) -> None:
    """$ prefix on ticks and thousands separators in hover labels."""
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    update(tickprefix="$" if is_money else "", hoverformat="$,.0f" if is_money else ",.0f")


def line_chart(df: pd.DataFrame, x: str, y: str, title: str = "", height: int = 360) -> go.Figure:
    fig = go.Figure(go.Scatter(x=df[x], y=df[y], mode="lines", name=y.title(), line=dict(color=BLUE, width=2)))
    _style(fig, title, height)
    _value_axis(fig, "y", y in MONEY)
    return fig


def bar_chart(df: pd.DataFrame, label: str, value: str, title: str = "", height: int | None = None) -> go.Figure:
    fig = go.Figure(go.Bar(x=df[value], y=df[label].astype(str), orientation="h", marker_color=BLUE))
    _style(fig, title, height or max(260, 28 * len(df) + 90))
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=True, gridcolor=GRID)         # horizontal bars: grid on the value axis
    fig.update_layout(hovermode="closest")
    _value_axis(fig, "x", value in MONEY)
    return fig


def donut(df: pd.DataFrame, names: str, values: str, title: str = "", height: int = 360) -> go.Figure:
    fig = go.Figure(go.Pie(labels=df[names], values=df[values], hole=0.55, sort=False,
                           marker=dict(colors=PALETTE), textinfo="percent",
                           hovertemplate="%{label}<br>$%{value:,.0f} (%{percent})<extra></extra>"))
    _style(fig, title, height)
    fig.update_layout(hovermode="closest")
    return fig


def stacked_bar(df: pd.DataFrame, x: str, ycols: list[str], title: str = "") -> go.Figure:
    fig = go.Figure([go.Bar(x=df[x], y=df[c], name=c.replace("_", " ").title(), marker_color=PALETTE[i])
                     for i, c in enumerate(ycols)])
    _style(fig, title)
    fig.update_layout(barmode="stack")
    _value_axis(fig, "y", False)
    return fig


def anomaly_chart(df: pd.DataFrame, title: str = "") -> go.Figure:
    fig = go.Figure(go.Scatter(x=df["period"], y=df["revenue"], mode="lines", name="Revenue", line=dict(color=BLUE, width=1.5)))
    if "is_anomaly" in df and df["is_anomaly"].any():
        a = df[df["is_anomaly"]]
        fig.add_trace(go.Scatter(x=a["period"], y=a["revenue"], mode="markers", name="Anomaly",
                                 marker=dict(color=RED, size=9, symbol="diamond")))
    _style(fig, title)
    _value_axis(fig, "y", True)
    return fig


def forecast_chart(history: pd.Series, fc: pd.DataFrame, level: float, title: str = "") -> go.Figure:
    """Actuals (solid), forecast (dashed) and a shaded prediction interval."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history.index, y=history.values, mode="lines", name="Actual",
                             line=dict(color=BLUE, width=2)))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["upper"], mode="lines", line=dict(width=0),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=fc.index, y=fc["lower"], mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(230,159,0,0.25)", name=f"{level:.0%} prediction interval",
                             hoverinfo="skip"))
    x = [history.index[-1]] + list(fc.index)          # join the dashed line to the last actual
    y = [history.iloc[-1]] + list(fc["yhat"])
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines+markers", name="Forecast",
                             line=dict(color=ORANGE, width=2, dash="dash")))
    _style(fig, title, 430)
    _value_axis(fig, "y", True)
    return fig

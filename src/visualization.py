"""Plotly figure builders (no Streamlit code, so they are testable)."""
from __future__ import annotations

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .knowledge_graph import LAYER, NODE_TYPES

BG, SURFACE, GRID = "#0b1220", "#111a2e", "#22304d"
CYAN, AMBER, GREEN, ORANGE, RED, MUTED = "#38bdf8", "#fbbf24", "#34d399", "#fb923c", "#ef4444", "#94a3b8"
STATUS_COLORS = {
    "normal": GREEN, "anomalous_behaviour": ORANGE, "possible_documented_fault": RED,
    "insufficient_evidence": AMBER, "conflicting_evidence": "#c084fc",
}
NODE_COLORS = {"solar_panel": CYAN, "measurement": "#60a5fa", "condition": AMBER, "anomaly": ORANGE,
               "rule": "#a78bfa", "fault": RED, "action": GREEN}


def _layout(fig: go.Figure, height: int) -> go.Figure:
    fig.update_layout(paper_bgcolor=BG, plot_bgcolor=SURFACE, font=dict(color="#e2e8f0", size=13),
                      height=height, margin=dict(l=60, r=20, t=40, b=40),
                      legend=dict(orientation="h", y=1.08, x=0))
    fig.update_xaxes(gridcolor=GRID, zeroline=False)
    fig.update_yaxes(gridcolor=GRID, zeroline=False)
    return fig


def telemetry_figure(df: pd.DataFrame, title: str = "") -> go.Figure:
    """Voltage / current / temperature vs time of ONE panel window with markers."""
    df = df.sort_values("Elapsed_s")
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.06,
                        subplot_titles=("Voltage (mV)", "Current (mA)", "Temperature (°C)"))
    for row, (col, name, color) in enumerate(
            [("Voltage_mV", "Voltage", CYAN), ("Current_mA", "Current", AMBER),
             ("Temperature_C", "Temperature", MUTED)], start=1):
        fig.add_trace(go.Scatter(x=df["Elapsed_s"], y=df[col], mode="lines", name=name,
                                 line=dict(color=color, width=1.4),
                                 hovertemplate=f"t=%{{x}} s<br>{name}=%{{y:.2f}}<extra></extra>"), row=row, col=1)
        if "ML_Status" in df.columns and row < 3:
            a = df[df["ML_Status"] == "Anomaly"]
            fig.add_trace(go.Scatter(x=a["Elapsed_s"], y=a[col], mode="markers", name="ML anomaly",
                                     marker=dict(color=ORANGE, size=4), legendgroup="ml",
                                     showlegend=(row == 1),
                                     hovertemplate="ML anomaly<br>t=%{x} s<br>%{y:.2f}<extra></extra>"), row=row, col=1)
        if "Documented_Type" in df.columns and row < 3:
            d = df[df["Documented_Type"] != "None"]
            fig.add_trace(go.Scatter(x=d["Elapsed_s"], y=d[col], mode="markers", name="Documented condition",
                                     marker=dict(color=RED, size=6, symbol="x"), legendgroup="doc",
                                     showlegend=(row == 1), text=d["Documented_Type"],
                                     hovertemplate="%{text}<br>t=%{x} s<br>%{y:.2f}<extra></extra>"), row=row, col=1)
    fig.update_xaxes(title_text="Elapsed time in session (s)", row=3, col=1)
    fig.update_layout(title=title)
    return _layout(fig, 720)


def confusion_figure(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    fig = go.Figure(go.Heatmap(z=matrix, x=[f"Predicted {l}" for l in labels], y=[f"Documented {l}" for l in labels],
                               text=matrix, texttemplate="%{text}", colorscale=[[0, SURFACE], [1, CYAN]],
                               showscale=False))
    fig.update_yaxes(autorange="reversed")
    return _layout(fig, 360)


def distribution_figure(df: pd.DataFrame, x: str, color: str, y: str = "count") -> go.Figure:
    cmap = {"Normal": GREEN, "Anomaly": ORANGE, "Not scored": MUTED}
    fig = go.Figure()
    for k, g in df.groupby(color):
        fig.add_trace(go.Bar(x=g[x], y=g[y], name=k, marker_color=cmap.get(k, CYAN)))
    fig.update_layout(barmode="stack", xaxis_title=x, yaxis_title="Observations")
    return _layout(fig, 360)


def graph_figure(G: nx.DiGraph) -> go.Figure:
    """Layered layout (panel -> ... -> action); zoom/pan via Plotly."""
    fig = go.Figure()
    if G.number_of_nodes() == 0:
        return _layout(fig, 300)
    by_layer: dict[int, list[str]] = {}
    for n, a in G.nodes(data=True):
        by_layer.setdefault(a["layer"], []).append(n)
    pos = {}
    for layer, nodes in by_layer.items():
        nodes.sort(key=lambda n: (G.nodes[n]["ntype"], n))
        for i, n in enumerate(nodes):
            pos[n] = (layer, -(i - (len(nodes) - 1) / 2))
    for on_path, color, width in ((False, "#475569", 1.2), (True, AMBER, 3.5)):
        xs, ys = [], []
        for a, b, d in G.edges(data=True):
            if d["on_path"] == on_path:
                xs += [pos[a][0], pos[b][0], None]; ys += [pos[a][1], pos[b][1], None]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=width),
                                 hoverinfo="skip", showlegend=on_path, name="Current diagnosis path"))
    fig.update_layout(annotations=[
        dict(x=(pos[a][0] + pos[b][0]) / 2, y=(pos[a][1] + pos[b][1]) / 2, text=d["label"], showarrow=False,
             font=dict(size=10, color=AMBER if d["on_path"] else MUTED), bgcolor=SURFACE)
        for a, b, d in G.edges(data=True)])
    for ntype, color in NODE_COLORS.items():
        nodes = [n for n, a in G.nodes(data=True) if a["ntype"] == ntype]
        if not nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes], mode="markers+text",
            name=NODE_TYPES[ntype], text=[G.nodes[n]["label"].split("\n")[0][:34] for n in nodes],
            textposition="top center", textfont=dict(size=10),
            hovertext=[G.nodes[n]["label"].replace("\n", "<br>") for n in nodes], hoverinfo="text",
            marker=dict(size=[26 if G.nodes[n]["on_path"] else 18 for n in nodes], color=color,
                        line=dict(width=[3 if G.nodes[n]["on_path"] else 0 for n in nodes], color=AMBER))))
    fig.update_xaxes(visible=False, range=[-0.6, max(LAYER.values()) + 0.6])
    fig.update_yaxes(visible=False)
    return _layout(fig, 640)

"""Plotly-Abbildungen der GAT-Demo. Achsen sind gesperrt (fixedrange)."""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import gat_algorithm as A
import gat_constants as C

PALETTE = ["#4c78a8", "#e45756", "#54a24b", "#b279a2"]
REF_COLOR = "#7f7f7f"
GOOD = "#54a24b"
BAD = "#e45756"
WARN = "#f58518"
PURPLE = "#b279a2"
LINE_COLOR = "#4c78a8"
TEAL = "#17becf"


def lock_axes(fig):
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    return fig


def _base(fig, height):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), legend=dict(orientation="h", y=-0.25), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_map(a, model, epoch):
    """Kunden im Gebiet: Farbe = Vorhersage (oder wahrer Typ), Quadrat = unzuverlässiger Kunde, x = falsch vorhergesagt, schwarzer Ring = bekanntes Etikett. model: 'gat', 'gcn' oder 'truth'."""
    g = a.graph
    if model == "truth":
        pred = g.y
    else:
        hist = a.gat if model == "gat" else a.gcn
        pred = hist.history["pred"][epoch - 1]
    fig = go.Figure()
    iu = np.array(np.nonzero(np.triu(g.A, 1)))
    xs, ys = [], []
    for i, j in zip(*iu):
        xs += [g.xy[i, 0], g.xy[j, 0], None]
        ys += [g.xy[i, 1], g.xy[j, 1], None]
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="rgba(150,150,150,0.30)", width=0.8), hoverinfo="skip", showlegend=False))
    for c in range(g.n_classes):
        for unrel, sym in ((False, "circle"), (True, "square")):
            m = (pred == c) & (g.U == unrel)
            if m.any():
                fig.add_trace(go.Scatter(x=g.xy[m, 0], y=g.xy[m, 1], mode="markers", marker=dict(size=8, color=PALETTE[c], symbol=sym, line=dict(color="white", width=0.6)), name=C.CLASS_NAMES[c], legendgroup=C.CLASS_NAMES[c],
                                         showlegend=not unrel, hovertemplate=f"{C.CLASS_NAMES[c]}{' (unzuverlässig)' if unrel else ''}<extra></extra>"))
    if model != "truth":
        bad = pred != g.y
        if bad.any():
            fig.add_trace(go.Scatter(x=g.xy[bad, 0], y=g.xy[bad, 1], mode="markers", marker=dict(size=12, symbol="x", color="black", line=dict(width=1.5)), name="falsch vorhergesagt"))
    tr = a.train_mask
    fig.add_trace(go.Scatter(x=g.xy[tr, 0], y=g.xy[tr, 1], mode="markers", marker=dict(size=15, symbol="circle-open", color="black", line=dict(width=2)), name="bekanntes Etikett"))
    fig.update_xaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False, scaleanchor="y")
    fig.update_yaxes(range=[-2, C.AREA + 2], showgrid=False, zeroline=False, showticklabels=False)
    return _base(fig, 470).update_layout(legend=dict(orientation="h", y=-0.05), margin=dict(l=10, r=10, t=10, b=10))


def build_attention_view(a, node):
    """Die Nachbarschaft eines Kunden zweimal: Strichbreite = Gewicht der Nachbarn beim GCN (links, nur aus den Graden) und die mittlere Aufmerksamkeit der ersten GAT-Schicht (rechts). Orange = unzuverlässiger Nachbar."""
    g = a.graph
    nbrs = np.flatnonzero(g.A[node])
    w_gcn = a.gcn_weights()[node, nbrs]
    w_gat = a.mean_alpha()[node, nbrs]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("GCN: Gewicht aus den Graden", "GAT: mittlere Aufmerksamkeit (Schicht 1)"), horizontal_spacing=0.06)
    for col, w in ((1, w_gcn), (2, w_gat)):
        scale = 14.0 / max(w.max(), 1e-9)
        for j, wj in zip(nbrs, w):
            fig.add_trace(go.Scatter(x=[g.xy[node, 0], g.xy[j, 0]], y=[g.xy[node, 1], g.xy[j, 1]], mode="lines", line=dict(width=max(0.8, wj * scale), color="rgba(120,120,120,0.55)"), showlegend=False, hoverinfo="skip"), row=1, col=col)
        rel = ~g.U[nbrs]
        fig.add_trace(go.Scatter(x=g.xy[nbrs[rel], 0], y=g.xy[nbrs[rel], 1], mode="markers", marker=dict(size=11, color=LINE_COLOR), name="zuverlässiger Nachbar", showlegend=col == 1,
                                 hovertemplate="zuverlässig, Gewicht %{customdata:.2f}<extra></extra>", customdata=w[rel]), row=1, col=col)
        fig.add_trace(go.Scatter(x=g.xy[nbrs[~rel], 0], y=g.xy[nbrs[~rel], 1], mode="markers", marker=dict(size=11, color=WARN, symbol="square"), name="unzuverlässiger Nachbar", showlegend=col == 1,
                                 hovertemplate="unzuverlässig, Gewicht %{customdata:.2f}<extra></extra>", customdata=w[~rel]), row=1, col=col)
        fig.add_trace(go.Scatter(x=[g.xy[node, 0]], y=[g.xy[node, 1]], mode="markers", marker=dict(size=17, symbol="star", color="black"), name="gewählter Kunde", showlegend=col == 1), row=1, col=col)
    fig.update_xaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_yaxes(showgrid=False, zeroline=False, showticklabels=False)
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=50, b=10), legend=dict(orientation="h", y=-0.08), plot_bgcolor="rgba(0,0,0,0)")
    return lock_axes(fig)


def build_heads(a):
    """Anteil des Gewichts, den jeder Kopf der ersten Schicht auf unzuverlässige Nachbarn legt (Knoten mit Nachbarn), gegen ihren Anteil unter den Nachbarn und gegen das GCN."""
    g = a.graph
    masses = A.head_masses(a.alphas, g.A, g.U)
    share = A.neighbor_mass(np.ones_like(g.A), g.A, g.U)
    gcn_m = A.neighbor_mass(A.normalized_adjacency(g.A), g.A, g.U)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=[f"Kopf {h + 1}" for h in range(len(masses))], y=[100 * m for m in masses], marker_color=PURPLE, name="GAT-Kopf", text=[f"{100 * m:.0f} %" for m in masses], textposition="outside"))
    fig.add_hline(y=100 * share, line=dict(color=REF_COLOR, dash="dot"), annotation_text=f"Anteil unzuverlässiger Nachbarn: {100 * share:.0f} %", annotation_position="bottom left")
    fig.add_hline(y=100 * gcn_m, line=dict(color=LINE_COLOR, dash="dash"), annotation_text=f"GCN: {100 * gcn_m:.0f} %", annotation_position="top left")
    fig.update_yaxes(title_text="Gewicht auf unzuverlässige Nachbarn (%)", range=[0, 105])
    return _base(fig, 320).update_layout(showlegend=False)


def build_curves(a, epoch):
    E = len(a.gat.history["loss"])
    xs = list(range(1, E + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.gat.history["test_acc"]], mode="lines", name="GAT", line=dict(color=PURPLE, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.gcn.history["test_acc"]], mode="lines", name="GCN", line=dict(color=LINE_COLOR, width=2.5)))
    fig.add_trace(go.Scatter(x=xs, y=[100 * v for v in a.gat.history["train_acc"]], mode="lines", name="GAT, bekannte Kunden", line=dict(color=REF_COLOR, width=1.5, dash="dot")))
    fig.add_vline(x=epoch, line=dict(color=WARN, dash="dash"))
    fig.update_xaxes(title_text="Epoche")
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[0, 102])
    return _base(fig, 320)


def _lines(rows, xkey, xtitle, models, height=340, reverse=False):
    fig = go.Figure()
    xs = [r[xkey] if xkey != "homophily" else 100 * r[xkey] for r in rows]
    spec = {"gat": ("GAT", PURPLE), "gcn": ("GCN", LINE_COLOR), "mlp": ("MLP (ohne Nachbarn)", REF_COLOR), "oracle": ("GCN mit bekannter Zuverlässigkeit", TEAL)}
    for m in models:
        name, color = spec[m]
        fig.add_trace(go.Scatter(x=xs, y=[100 * r[m] for r in rows], error_y=dict(type="data", array=[100 * r[m + "_se"] for r in rows]), mode="lines+markers", name=name, line=dict(color=color, width=2.5, dash="dot" if m == "oracle" else "solid")))
    fig.update_xaxes(title_text=xtitle, **({"autorange": "reversed"} if reverse else {}))
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[30, 100])
    return _base(fig, height).update_layout(legend=dict(orientation="h", y=-0.3))


def build_labels(rows):
    fig = _lines(rows, "labels", "Bekannte Etiketten je Gebietstyp", ("gat", "gcn", "mlp", "oracle"))
    fig.update_xaxes(type="log", tickvals=[r["labels"] for r in rows], ticktext=[str(r["labels"]) for r in rows])
    return fig


def build_unreliable(rows):
    fig = _lines(rows, "unreliable", "Anteil unzuverlässiger Kunden", ("gat", "gcn", "mlp", "oracle"))
    fig.update_xaxes(tickformat=".0%")
    return fig


def build_wrong(rows):
    return _lines(rows, "homophily", "Anteil der Kanten zwischen Kunden desselben Gebietstyps (%)", ("gat", "gcn", "mlp"), reverse=True)


def build_ablation(res):
    names = ["MLP", "GCN", "GAT ohne<br>Aufmerksamkeit", "GCN mit bekannter<br>Zuverlässigkeit", "GAT"]
    keys = ["mlp", "gcn", "uniform", "oracle", "gat"]
    colors = [REF_COLOR, LINE_COLOR, "#c7c7c7", TEAL, PURPLE]
    fig = go.Figure(go.Bar(x=names, y=[100 * res[k] for k in keys], error_y=dict(type="data", array=[100 * res[k + "_se"] for k in keys]), marker_color=colors, text=[f"{100 * res[k]:.0f} %" for k in keys],
                           textposition="outside", showlegend=False))
    fig.update_yaxes(title_text="Genauigkeit auf unbekannten Kunden (%)", range=[0, 112])
    return _base(fig, 340)

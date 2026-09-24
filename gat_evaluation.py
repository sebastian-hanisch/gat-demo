"""Auswertung: GAT gegen GCN und MLP auf einem Liefergebiets-Graphen mit unzuverlässigen Kunden und drei Experimente (Zahl der Etiketten, Anteil unzuverlässiger Kunden, falsche Kanten)."""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np

import gat_algorithm as A
import gat_constants as C
import gat_scenario as S


@dataclass(frozen=True)
class Settings:
    n: int = C.DEFAULT_N
    classes: int = C.DEFAULT_CLASSES
    labels: int = C.DEFAULT_LABELS
    neighbors: int = C.DEFAULT_NEIGHBORS
    noise: float = C.DEFAULT_NOISE
    unreliable: float = C.DEFAULT_UNREL
    extra: float = C.DEFAULT_EXTRA
    wrong: float = C.DEFAULT_WRONG
    seed: int = 7


def make(settings):
    g = S.generate(settings.n, settings.classes, settings.neighbors, settings.noise, settings.unreliable, settings.extra, settings.wrong, settings.seed)
    return g, S.split(g.y, settings.labels, settings.seed)


@dataclass
class Analysis:
    settings: Settings
    graph: S.Graph
    train_mask: np.ndarray
    gat: A.Model
    gcn: A.Model
    mlp: A.Model
    uniform: A.Model        # GAT ohne gelernte Aufmerksamkeit (Ablation)
    oracle: A.Model         # GCN mit bekannter Zuverlässigkeit (Referenz)
    pred_gat: np.ndarray
    pred_gcn: np.ndarray
    pred_mlp: np.ndarray
    pred_uniform: np.ndarray
    pred_oracle: np.ndarray
    alphas: list            # Aufmerksamkeit der ersten GAT-Schicht je Kopf (n x n)
    t_scores: np.ndarray    # Ziel-Bewertungen der ersten Schicht (n, Köpfe)

    def acc(self, pred):
        m = ~self.train_mask
        return float((pred[m] == self.graph.y[m]).mean())

    @property
    def acc_gat(self):
        return self.acc(self.pred_gat)

    @property
    def acc_gcn(self):
        return self.acc(self.pred_gcn)

    @property
    def acc_mlp(self):
        return self.acc(self.pred_mlp)

    @property
    def acc_uniform(self):
        return self.acc(self.pred_uniform)

    @property
    def acc_oracle(self):
        return self.acc(self.pred_oracle)

    def majority_rate(self):
        m = ~self.train_mask
        return float(np.bincount(self.graph.y[m]).max() / m.sum())

    def mean_alpha(self):
        return np.mean(self.alphas, axis=0)

    def gcn_weights(self):
        """Gewichte des GCN (A_hat), zeilenweise auf die Nachbarn normiert vergleichbar mit alpha."""
        return A.normalized_adjacency(self.graph.A)


@lru_cache(maxsize=32)
def analyse(settings):
    g, tr = make(settings)
    gat = A.train_gat(g.A, g.X, g.y, tr, seed=settings.seed, record_pred=True)
    gcn = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed, record_pred=True)
    mlp = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed, use_graph=False)
    uni = A.train_gat(g.A, g.X, g.y, tr, seed=settings.seed, attention=False)
    ora = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed, ahat=A.oracle_ahat(g.A, g.U))
    alphas, t = A.attention_layer1(gat, g.A, g.X)
    return Analysis(settings, g, tr, gat, gcn, mlp, uni, ora, A.predict_gat(gat, g.A, g.X), A.predict_gcn(gcn, g.A, g.X), A.predict_gcn(mlp, g.A, g.X), A.predict_gat(uni, g.A, g.X), A.predict_gcn(ora, g.A, g.X), alphas, t)


def _mean_se(v):
    v = np.asarray(v, dtype=float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0)


def _run(settings, with_attention=False):
    g, tr = make(settings)
    gat = A.train_gat(g.A, g.X, g.y, tr, seed=settings.seed)
    gcn = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed)
    mlp = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed, use_graph=False)
    ora = A.train_gcn(g.A, g.X, g.y, tr, seed=settings.seed, ahat=A.oracle_ahat(g.A, g.U))
    out = {"gat": gat.history["test_acc"][-1], "gcn": gcn.history["test_acc"][-1], "mlp": mlp.history["test_acc"][-1], "oracle": ora.history["test_acc"][-1], "g": g}
    if with_attention:
        alphas, t = A.attention_layer1(gat, g.A, g.X)
        out["mass_gat"] = A.neighbor_mass(np.mean(alphas, axis=0), g.A, g.U)
        out["mass_gcn"] = A.neighbor_mass(A.normalized_adjacency(g.A), g.A, g.U)
        out["share_unreliable_neighbors"] = A.neighbor_mass(np.ones_like(g.A), g.A, g.U)
        out["ranking"] = A.ranking_consistency(alphas, t, g.A)
    return out


def _row(settings_list, key, value, with_attention=False):
    res = [_run(s, with_attention) for s in settings_list]
    row = {key: value, "n_seeds": len(res)}
    for m in ("gat", "gcn", "mlp", "oracle"):
        mean, se = _mean_se([r[m] for r in res])
        row[m], row[m + "_se"] = mean, se
    d = np.array([r["gat"] - r["gcn"] for r in res])
    row["diff"], row["diff_se"] = _mean_se(d)
    row["gat_wins"] = int(np.sum(d > 0))
    if with_attention:
        for k in ("mass_gat", "mass_gcn", "share_unreliable_neighbors", "ranking"):
            row[k] = float(np.mean([r[k] for r in res]))
    return row


def _with(base, **kw):
    d = dict(base.__dict__)
    d.update(kw)
    return Settings(**d)


# --- Experiment 1: Zahl der bekannten Etiketten -------------------------------------------------------------------------------------------------


def labels_experiment(levels=None, seeds=None, base=None):
    levels = C.LABEL_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    return [_row([_with(base, labels=m, seed=s) for s in seeds], "labels", m, with_attention=True) for m in levels]


# --- Experiment 2: Anteil unzuverlässiger Kunden ---------------------------------------------------------------------------------------------


def unreliable_experiment(levels=None, seeds=None, base=None):
    levels = C.UNREL_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    return [_row([_with(base, unreliable=u, seed=s) for s in seeds], "unreliable", u, with_attention=True) for u in levels]


# --- Experiment 3: falsche Kanten ---------------------------------------------------------------------------------------------------------------


def wrong_experiment(levels=None, seeds=None, base=None):
    levels = C.WRONG_LEVELS if levels is None else levels
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings(unreliable=0.0, extra=0.0, labels=20) if base is None else base
    rows = []
    for w in levels:
        row = _row([_with(base, wrong=w, seed=s) for s in seeds], "wrong", w, with_attention=True)
        row["homophily"] = float(np.mean([S.generate(base.n, base.classes, base.neighbors, base.noise, base.unreliable, base.extra, w, s).edge_homophily() for s in seeds]))
        rows.append(row)
    return rows


# --- Experiment 4: Ablation und Blick in die Aufmerksamkeit ---------------------------------------------------------------------------------------


def ablation_experiment(seeds=None, base=None):
    """Dieselbe Architektur mit und ohne gelernte Aufmerksamkeit (Bewertungsvektoren auf 0 gehalten), GCN, MLP und das Orakel; dazu, wie viel Gewicht die Aufmerksamkeit auf unzuverlässige Nachbarn legt (vor und nach dem Training,
    je Kopf) und ob die Nachbar-Rangfolge vom fragenden Knoten unabhängig ist."""
    seeds = C.EXP_SEEDS if seeds is None else seeds
    base = Settings() if base is None else base
    rows = {"gat": [], "uniform": [], "gcn": [], "mlp": [], "oracle": []}
    mass_init, mass_trained, share, ranking, spread = [], [], [], [], []
    for s in seeds:
        st = _with(base, seed=s)
        g, tr = make(st)
        gat = A.train_gat(g.A, g.X, g.y, tr, seed=s)
        uni = A.train_gat(g.A, g.X, g.y, tr, seed=s, attention=False)
        gcn = A.train_gcn(g.A, g.X, g.y, tr, seed=s)
        mlp = A.train_gcn(g.A, g.X, g.y, tr, seed=s, use_graph=False)
        ora = A.train_gcn(g.A, g.X, g.y, tr, seed=s, ahat=A.oracle_ahat(g.A, g.U))
        for k, m in (("gat", gat), ("uniform", uni), ("gcn", gcn), ("mlp", mlp), ("oracle", ora)):
            rows[k].append(m.history["test_acc"][-1])
        alphas, t = A.attention_layer1(gat, g.A, g.X)
        init = A.attention_layer1_init(s, g.X, g.A, n_classes=st.classes)
        mass_init.append(A.neighbor_mass(np.mean(init, axis=0), g.A, g.U))
        mass_trained.append(A.neighbor_mass(np.mean(alphas, axis=0), g.A, g.U))
        share.append(A.neighbor_mass(np.ones_like(g.A), g.A, g.U))
        hm = A.head_masses(alphas, g.A, g.U)
        spread.append(max(hm) - min(hm))
        ranking.append(A.ranking_consistency(alphas, t, g.A))
    out = {"n_seeds": len(seeds)}
    for k, v in rows.items():
        out[k], out[k + "_se"] = _mean_se(v)
    d = np.array(rows["gat"]) - np.array(rows["uniform"])
    out["gat_minus_uniform"], out["gat_minus_uniform_se"] = _mean_se(d)
    d2 = np.array(rows["uniform"]) - np.array(rows["gcn"])
    out["uniform_minus_gcn"], out["uniform_minus_gcn_se"] = _mean_se(d2)
    out["mass_init"], out["mass_trained"], out["share_unreliable"] = float(np.mean(mass_init)), float(np.mean(mass_trained)), float(np.mean(share))
    out["head_spread"], out["ranking"] = float(np.mean(spread)), float(np.mean(ranking))
    return out

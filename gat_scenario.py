"""Vehikel D "Liefergebiete" (wie gcn-demo) mit unzuverlässigen Kunden: n Kunden in einem 100 x 100 km großen Gebiet, räumlich zusammenhängende Gebietstypen (nächstes von K verdeckten Zentren), vier Merkmale je Kunde mit
Mittelwerten je Gebietstyp; der Graph verbindet räumliche Nachbarn. Neu: ein Anteil der Kunden ist **unzuverlässig** (lückenhafte Telematik): ihre Merkmale sind zusätzlich stark verrauscht, und ein fünftes Merkmal
("Meldeausfall") zeigt das an. Ein GCN mittelt alle Nachbarn gleich, auch die unzuverlässigen; Aufmerksamkeit kann sie herunterwichten. Falsche Kanten wie in gcn-demo."""

from dataclasses import dataclass

import numpy as np

import gat_constants as C


@dataclass(frozen=True)
class Graph:
    xy: np.ndarray          # (n, 2)
    X: np.ndarray           # (n, 5) Merkmale (vier gemessene plus Meldeausfall-Kennzeichen)
    y: np.ndarray           # (n,) Gebietstyp
    A: np.ndarray           # (n, n) symmetrische 0/1-Nachbarschaft ohne Schleifen
    U: np.ndarray           # (n,) bool: unzuverlässiger Kunde
    n_classes: int
    seed: int

    @property
    def n(self):
        return len(self.y)

    def edge_homophily(self):
        i, j = np.nonzero(np.triu(self.A, 1))
        return float(np.mean(self.y[i] == self.y[j])) if len(i) else 1.0

    def n_edges(self):
        return int(np.triu(self.A, 1).sum())


def _centers(rng, k):
    for _ in range(1000):
        c = rng.random((k, 2)) * C.AREA
        d = np.linalg.norm(c[:, None] - c[None], axis=2) + np.eye(k) * 1e9
        if d.min() >= 0.28 * C.AREA:
            return c
    return c


def knn_adjacency(xy, k):
    n = len(xy)
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(n) * 1e9
    idx = np.argsort(d, axis=1)[:, :k]
    A = np.zeros((n, n))
    A[np.repeat(np.arange(n), k), idx.ravel()] = 1.0
    return np.maximum(A, A.T)


def rewire(A, fraction, rng):
    """Ersetzt einen Anteil der Kanten durch gleich viele zufällige (keine Schleifen, keine Doppelten): die Kantenzahl bleibt, die Homophilie sinkt."""
    if fraction <= 0:
        return A.copy()
    n = len(A)
    iu = np.array(np.nonzero(np.triu(A, 1))).T
    m = len(iu)
    n_swap = int(round(fraction * m))
    keep = np.ones(m, dtype=bool)
    keep[rng.permutation(m)[:n_swap]] = False
    B = np.zeros_like(A)
    B[iu[keep, 0], iu[keep, 1]] = 1.0
    B = np.maximum(B, B.T)
    added = 0
    while added < n_swap:
        i, j = rng.integers(0, n, 2)
        if i != j and B[i, j] == 0:
            B[i, j] = B[j, i] = 1.0
            added += 1
    return B


def generate(n=C.DEFAULT_N, n_classes=C.DEFAULT_CLASSES, k=C.DEFAULT_NEIGHBORS, noise=C.DEFAULT_NOISE, unreliable=C.DEFAULT_UNREL, extra=C.DEFAULT_EXTRA, wrong=C.DEFAULT_WRONG, seed=0):
    rng = np.random.default_rng(seed)
    xy = rng.random((n, 2)) * C.AREA
    centers = _centers(rng, n_classes)
    y = np.argmin(np.linalg.norm(xy[:, None] - centers[None], axis=2), axis=1)
    means = rng.normal(size=(n_classes, C.N_FEATURES - 1))
    means /= np.linalg.norm(means, axis=1, keepdims=True)
    X4 = means[y] + noise * rng.normal(size=(n, C.N_FEATURES - 1)) * 0.5
    rng2 = np.random.default_rng([seed, 55])                                   # eigener Strom: unabhängig von Lage und Grundrauschen
    U = rng2.random(n) < unreliable
    X4[U] += extra * rng2.normal(size=(int(U.sum()), C.N_FEATURES - 1))
    flag = U.astype(float)[:, None] + 0.05 * rng2.normal(size=(n, 1))
    X = np.hstack([X4, flag])
    A = rewire(knn_adjacency(xy, k), wrong, np.random.default_rng([seed, 99]))
    return Graph(xy, X, y, A, U, n_classes, int(seed))


def split(y, per_class, seed):
    """Je Gebietstyp `per_class` bekannte Etiketten (Training); alle übrigen Kunden sind unbekannt und dienen der Prüfung."""
    rng = np.random.default_rng([seed, 7])
    train = np.zeros(len(y), dtype=bool)
    for c in np.unique(y):
        idx = np.flatnonzero(y == c)
        train[rng.permutation(idx)[:min(per_class, max(len(idx) - 2, 1))]] = True          # mindestens zwei unbekannte Kunden je Typ bleiben zur Prüfung übrig
    return train

"""GAT nach Veličković et al. (2018) und die Vergleichsnetze GCN und MLP, von Grund auf in numpy.

GAT-Schicht mit K Köpfen: Wh = H W; Bewertung e_ij = LeakyReLU(a_quelle . Wh_i + a_ziel . Wh_j) für jede Kante (und die Schleife i-i); Aufmerksamkeit alpha_ij = softmax_j e_ij über die Nachbarn von i;
Ausgabe h_i = Summe_j alpha_ij Wh_j je Kopf, Köpfe nebeneinander gelegt. Zwei Schichten: 4 Köpfe x 8 Merkmale mit ReLU, dann ein Kopf mit den Klassen-Logits. Kein Dropout (feste Epochenzahl, nichts am Test abgestimmt).
Die Bewertung ist eine SUMME aus einem Anteil des empfangenden und einem des sendenden Knotens: die Rangfolge der Nachbarn hängt nur vom Nachbarn ab, nie vom Knoten, der fragt ("statische Aufmerksamkeit")."""

from dataclasses import dataclass, field

import numpy as np

import gat_constants as C


def softmax(Z):
    Z = Z - Z.max(axis=1, keepdims=True)
    E = np.exp(Z)
    return E / E.sum(axis=1, keepdims=True)


def _adam(params, grads, mo, ve, t, lr):
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in params:
        mo[k] = b1 * mo[k] + (1 - b1) * grads[k]
        ve[k] = b2 * ve[k] + (1 - b2) * grads[k] ** 2
        params[k] = params[k] - lr * (mo[k] / (1 - b1 ** t)) / (np.sqrt(ve[k] / (1 - b2 ** t)) + eps)


# --- GCN / MLP (wie gcn-demo, zwei Schichten) -------------------------------------------------------------------------------------------------


def normalized_adjacency(A):
    """A_hat = D^-1/2 (A + I) D^-1/2."""
    B = A + np.eye(len(A))
    d = B.sum(axis=1)
    inv = 1.0 / np.sqrt(d)
    return B * inv[:, None] * inv[None, :]


def _glorot(rng, a, b):
    lim = np.sqrt(6.0 / (a + b))
    return rng.uniform(-lim, lim, size=(a, b))


def gcn_forward(Ahat, X, p):
    Z1 = Ahat @ X @ p["W1"]
    H1 = np.maximum(Z1, 0.0)
    return Ahat @ H1 @ p["W2"], (Z1, H1)


def gcn_loss_and_grads(Ahat, X, y, train, p, wd):
    logits, (Z1, H1) = gcn_forward(Ahat, X, p)
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    reg = 0.5 * wd * sum(float((p[k] ** 2).sum()) for k in ("W1", "W2"))
    dZ = np.zeros_like(logits)
    dZ[idx] = P[idx]
    dZ[idx, y[idx]] -= 1.0
    dZ /= len(idx)
    dW2 = (Ahat @ H1).T @ dZ
    dZ1 = (Ahat.T @ (dZ @ p["W2"].T)) * (Z1 > 0)
    dW1 = (Ahat @ X).T @ dZ1
    return ce + reg, {"W1": dW1 + wd * p["W1"], "W2": dW2 + wd * p["W2"]}


@dataclass
class Model:
    params: dict
    history: dict = field(default_factory=dict)
    kind: str = "gcn"
    ahat: object = None


def train_gcn(A, X, y, train_mask, seed=0, use_graph=True, epochs=C.EPOCHS, lr=C.LEARNING_RATE, wd=C.WEIGHT_DECAY, hidden=C.HIDDEN_GCN, test_mask=None, record_pred=False, ahat=None):
    """GCN (use_graph=True) oder MLP (A_hat = I) mit zwei Schichten; `ahat` überschreibt die Propagationsmatrix (auch nicht symmetrisch)."""
    K = int(y.max()) + 1
    Ahat = ahat if ahat is not None else (normalized_adjacency(A) if use_graph else np.eye(len(A)))
    rng = np.random.default_rng([seed, 31])
    p = {"W1": _glorot(rng, X.shape[1], hidden), "W2": _glorot(rng, hidden, K)}
    mo = {k: np.zeros_like(v) for k, v in p.items()}
    ve = {k: np.zeros_like(v) for k, v in p.items()}
    test_mask = ~train_mask if test_mask is None else test_mask
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    preds = []
    for t in range(1, epochs + 1):
        loss, g = gcn_loss_and_grads(Ahat, X, y, train_mask, p, wd)
        _adam(p, g, mo, ve, t, lr)
        pred = gcn_forward(Ahat, X, p)[0].argmax(axis=1)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)
    return Model(p, hist, "gcn" if use_graph else "mlp", ahat)


def predict_gcn(model, A, X):
    Ahat = model.ahat if model.ahat is not None else (normalized_adjacency(A) if model.kind == "gcn" else np.eye(len(A)))
    return gcn_forward(Ahat, X, model.params)[0].argmax(axis=1)


# --- GAT ------------------------------------------------------------------------------------------------------------------------------------------


def edge_mask(A):
    """Nachbarn plus Schleife."""
    return ((A + np.eye(len(A))) > 0).astype(float)


def leaky(z):
    return np.where(z > 0, z, C.LEAKY_SLOPE * z)


def dleaky(z):
    return np.where(z > 0, 1.0, C.LEAKY_SLOPE)


def masked_softmax(E, mask):
    E = np.where(mask > 0, E, -1e30)
    E = E - E.max(axis=1, keepdims=True)
    P = np.exp(E) * mask
    return P / P.sum(axis=1, keepdims=True)


def layer_forward(H, W, a_src, a_dst, mask):
    """W: (d, Köpfe * f); a_src, a_dst: (Köpfe, f). Rückgabe (Ausgabe (n, Köpfe * f), Cache)."""
    n = H.shape[0]
    heads, f = a_src.shape
    Whh = (H @ W).reshape(n, heads, f)
    s = np.einsum("nhf,hf->nh", Whh, a_src)
    t = np.einsum("nhf,hf->nh", Whh, a_dst)
    out = np.zeros((n, heads, f))
    alphas, zs = [], []
    for h in range(heads):
        z = s[:, h][:, None] + t[:, h][None, :]
        al = masked_softmax(leaky(z), mask)
        out[:, h] = al @ Whh[:, h]
        alphas.append(al)
        zs.append(z)
    return out.reshape(n, heads * f), (H, Whh, s, t, alphas, zs)


def layer_backward(dout, W, a_src, a_dst, mask, cache):
    """Rückgabe (dH, dW, da_src, da_dst)."""
    H, Whh, s, t, alphas, zs = cache
    n = H.shape[0]
    heads, f = a_src.shape
    dO = dout.reshape(n, heads, f)
    dWhh = np.zeros_like(Whh)
    da_src = np.zeros_like(a_src)
    da_dst = np.zeros_like(a_dst)
    for h in range(heads):
        al = alphas[h]
        dWhh[:, h] += al.T @ dO[:, h]
        dal = dO[:, h] @ Whh[:, h].T
        de = al * (dal - (al * dal).sum(axis=1, keepdims=True))
        dz = de * dleaky(zs[h]) * mask
        ds, dt = dz.sum(axis=1), dz.sum(axis=0)
        dWhh[:, h] += ds[:, None] * a_src[h][None, :] + dt[:, None] * a_dst[h][None, :]
        da_src[h] = Whh[:, h].T @ ds
        da_dst[h] = Whh[:, h].T @ dt
    dWh = dWhh.reshape(n, heads * f)
    return dWh @ W.T, H.T @ dWh, da_src, da_dst


def init_gat(d, heads, f, K, seed):
    rng = np.random.default_rng([seed, 5])
    return {"W1": _glorot(rng, d, heads * f), "as1": rng.uniform(-0.3, 0.3, (heads, f)), "ad1": rng.uniform(-0.3, 0.3, (heads, f)),
            "W2": _glorot(rng, heads * f, K), "as2": rng.uniform(-0.3, 0.3, (1, K)), "ad2": rng.uniform(-0.3, 0.3, (1, K))}


def gat_forward(p, X, mask):
    o1, c1 = layer_forward(X, p["W1"], p["as1"], p["ad1"], mask)
    h1 = np.maximum(o1, 0.0)
    o2, c2 = layer_forward(h1, p["W2"], p["as2"], p["ad2"], mask)
    return o2, (o1, c1, h1, c2)


def gat_loss_and_grads(p, X, y, train, mask, wd):
    logits, (o1, c1, h1, c2) = gat_forward(p, X, mask)
    P = softmax(logits)
    idx = np.flatnonzero(train)
    ce = -np.log(P[idx, y[idx]] + C.EPS).mean()
    reg = 0.5 * wd * sum(float((p[k] ** 2).sum()) for k in ("W1", "W2"))
    dl = np.zeros_like(logits)
    dl[idx] = P[idx]
    dl[idx, y[idx]] -= 1.0
    dl /= len(idx)
    dh1, dW2, das2, dad2 = layer_backward(dl, p["W2"], p["as2"], p["ad2"], mask, c2)
    _, dW1, das1, dad1 = layer_backward(dh1 * (o1 > 0), p["W1"], p["as1"], p["ad1"], mask, c1)
    return ce + reg, {"W1": dW1 + wd * p["W1"], "as1": das1, "ad1": dad1, "W2": dW2 + wd * p["W2"], "as2": das2, "ad2": dad2}


def train_gat(A, X, y, train_mask, seed=0, heads=C.HEADS, f=C.HEAD_DIM, epochs=C.EPOCHS, lr=C.LEARNING_RATE, wd=C.WEIGHT_DECAY, test_mask=None, record_pred=False, attention=True):
    """attention=False: Bewertungsvektoren auf 0 gehalten (nicht gelernt) - alle Nachbarn zählen gleich, aber dieselben Köpfe und Merkmale (Ablation)."""
    K = int(y.max()) + 1
    mask = edge_mask(A)
    p = init_gat(X.shape[1], heads, f, K, seed)
    if not attention:
        for k in ("as1", "ad1", "as2", "ad2"):
            p[k] = np.zeros_like(p[k])
    mo = {k: np.zeros_like(v) for k, v in p.items()}
    ve = {k: np.zeros_like(v) for k, v in p.items()}
    test_mask = ~train_mask if test_mask is None else test_mask
    hist = {"loss": [], "train_acc": [], "test_acc": []}
    preds = []
    for t in range(1, epochs + 1):
        loss, g = gat_loss_and_grads(p, X, y, train_mask, mask, wd)
        if not attention:
            for k in ("as1", "ad1", "as2", "ad2"):
                g[k] = np.zeros_like(g[k])
        _adam(p, g, mo, ve, t, lr)
        pred = gat_forward(p, X, mask)[0].argmax(axis=1)
        hist["loss"].append(float(loss))
        hist["train_acc"].append(float((pred[train_mask] == y[train_mask]).mean()))
        hist["test_acc"].append(float((pred[test_mask] == y[test_mask]).mean()))
        if record_pred:
            preds.append(pred)
    if record_pred:
        hist["pred"] = np.array(preds)
    return Model(p, hist, "gat")


def predict_gat(model, A, X):
    return gat_forward(model.params, X, edge_mask(A))[0].argmax(axis=1)


def attention_layer1(model, A, X):
    """Aufmerksamkeit der ersten Schicht: Liste (je Kopf) von Matrizen alpha (n x n) und die Ziel-Bewertungen t (n, Köpfe)."""
    _, (o1, c1, h1, c2) = gat_forward(model.params, X, edge_mask(A))
    return c1[4], c1[3]


def neighbor_mass(weights, A, U):
    """Mittlerer Anteil des Gewichts, den ein Knoten auf seine UNZUVERLÄSSIGEN Nachbarn legt (ohne sich selbst, Gewichte je Knoten auf die Nachbarn normiert); nur Knoten mit Nachbarn."""
    W = weights * A
    tot = W.sum(axis=1)
    ok = tot > 0
    share = (W * U[None, :]).sum(axis=1)[ok] / tot[ok]
    return float(share.mean())


def ranking_consistency(alphas, t, A):
    """Anteil der Nachbarpaare (j, k) eines Knotens i, deren Aufmerksamkeitsrangfolge (alpha_ij gegen alpha_ik) mit der Rangfolge der Ziel-Bewertung (t_j gegen t_k) übereinstimmt - je Kopf gemittelt.
    Bei der GAT-Bewertung ist das immer 1: die Rangfolge der Nachbarn hängt nicht vom fragenden Knoten ab."""
    n = A.shape[0]
    agree = total = 0
    for h, al in enumerate(alphas):
        for i in range(n):
            nb = np.flatnonzero(A[i])
            if len(nb) < 2:
                continue
            a = al[i, nb]
            tt = t[nb, h]
            da = np.sign(a[:, None] - a[None, :])
            dt = np.sign(tt[:, None] - tt[None, :])
            m = (dt != 0) & (np.abs(a[:, None] - a[None, :]) > 1e-12)
            agree += int((da[m] == dt[m]).sum())
            total += int(m.sum())
    return agree / total if total else 1.0


def oracle_ahat(A, U, w=0.2):
    """GCN-Vergleich mit BEKANNTER Zuverlässigkeit (nicht lernbar, nur Referenz): Zeile i mittelt über sich selbst (Gewicht 1) und die Nachbarn j mit Gewicht 1 (zuverlässig) bzw. w (unzuverlässig), auf Summe 1 normiert."""
    n = len(A)
    W = (A + np.eye(n)) * np.where(U, w, 1.0)[None, :]
    W[np.arange(n), np.arange(n)] = 1.0
    return W / W.sum(axis=1, keepdims=True)


def attention_layer1_init(seed, X, A, heads=C.HEADS, f=C.HEAD_DIM, n_classes=3):
    """Aufmerksamkeit der ersten Schicht bei der Anfangsbelegung (vor jedem Training) - Vergleichsmaß dafür, was das Training verändert."""
    p = init_gat(X.shape[1], heads, f, n_classes, seed)
    return layer_forward(X, p["W1"], p["as1"], p["ad1"], edge_mask(A))[1][4]


def head_masses(alphas, A, U):
    """neighbor_mass je Kopf."""
    return [neighbor_mass(al, A, U) for al in alphas]

"""GAT-Kern: Handrechnung, Vorwärtsrechnung gegen explizite Schleifen, Gradienten gegen zentrale Differenzen, statische Aufmerksamkeit, Ablation; GCN/MLP-Grundlagen und Orakel-Matrix."""

import numpy as np
import pytest

import gat_algorithm as A
import gat_scenario as S


def path3():
    return np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)


# --- GAT-Schicht ------------------------------------------------------------------------------------------------------------------------------


def test_gat_layer_by_hand_for_two_nodes_scores_are_static():
    """Zwei verbundene Knoten, ein Kopf, f = 1: H = (1, 2), W = 1, a_E = a_S = 1 -> Wh = (1, 2), s = t = (1, 2); e_00 = 2, e_01 = 3, e_10 = 3, e_11 = 4;
    Zeile 0: softmax(2, 3) = (1/(1+e), e/(1+e)) = (0,2689; 0,7311), Zeile 1: softmax(3, 4) = dasselbe. Ausgabe beider Knoten: 0,2689 * 1 + 0,7311 * 2 = 1,7311."""
    H = np.array([[1.0], [2.0]])
    mask = A.edge_mask(np.array([[0, 1], [1, 0]], dtype=float))
    out, cache = A.layer_forward(H, np.array([[1.0]]), np.array([[1.0]]), np.array([[1.0]]), mask)
    e = np.e
    al = cache[4][0]
    assert al[0] == pytest.approx([1 / (1 + e), e / (1 + e)]) and al[1] == pytest.approx([1 / (1 + e), e / (1 + e)])
    assert out[:, 0] == pytest.approx([1 / (1 + e) + 2 * e / (1 + e)] * 2)


def test_leaky_relu_branches_by_hand():
    """Negative Bewertungen: z = (-1, 2) -> (-0,2, 2); Softmax der Zeile (0, 1) mit e = (-0,2, 2)."""
    assert A.leaky(np.array([-1.0, 2.0])) == pytest.approx([-0.2, 2.0]) and A.dleaky(np.array([-1.0, 2.0])) == pytest.approx([0.2, 1.0])
    P = A.masked_softmax(np.array([[-0.2, 2.0, 9.0]]), np.array([[1.0, 1.0, 0.0]]))
    assert P[0] == pytest.approx([1 / (1 + np.exp(2.2)), np.exp(2.2) / (1 + np.exp(2.2)), 0.0])


def test_layer_forward_agrees_with_an_explicit_loop_for_several_heads():
    g = S.generate(14, 3, 3, 1.0, 0.3, 4.0, 0.0, 3)
    heads, f = 3, 2
    rng = np.random.default_rng(1)
    W = rng.normal(size=(5, heads * f))
    a_s, a_d = rng.normal(size=(heads, f)), rng.normal(size=(heads, f))
    mask = A.edge_mask(g.A)
    out, _ = A.layer_forward(g.X, W, a_s, a_d, mask)
    Wh = (g.X @ W).reshape(14, heads, f)
    ref = np.zeros((14, heads, f))
    for k in range(heads):
        for i in range(14):
            nb = [j for j in range(14) if mask[i, j]]
            e = np.array([np.where(v > 0, v, 0.2 * v) for v in [Wh[i, k] @ a_s[k] + Wh[j, k] @ a_d[k] for j in nb]])
            al = np.exp(e - e.max())
            al /= al.sum()
            ref[i, k] = sum(al[q] * Wh[j, k] for q, j in enumerate(nb))
    assert out == pytest.approx(ref.reshape(14, heads * f))


def test_attention_rows_sum_to_one_and_vanish_outside_the_neighborhood():
    g = S.generate(30, 3, 4, 1.0, 0.3, 4.0, 0.0, 2)
    p = A.init_gat(5, 4, 8, 3, 1)
    mask = A.edge_mask(g.A)
    _, c = A.layer_forward(g.X, p["W1"], p["as1"], p["ad1"], mask)
    for al in c[4]:
        assert al.sum(axis=1) == pytest.approx(np.ones(30)) and np.all(al[mask == 0] == 0) and np.all(al[mask > 0] > 0)


@pytest.mark.parametrize("heads,f", [(1, 3), (2, 3), (4, 2)])
def test_gat_gradients_agree_with_central_differences(heads, f):
    g = S.generate(25, 3, 4, 1.0, 0.3, 4.0, 0.0, 1)
    tr = S.split(g.y, 4, 1)
    mask = A.edge_mask(g.A)
    p = A.init_gat(5, heads, f, 3, 2)
    _, gr = A.gat_loss_and_grads(p, g.X, g.y, tr, mask, 5e-4)
    rng = np.random.default_rng(0)
    for k in p:
        for _ in range(5):
            idx = tuple(rng.integers(0, s) for s in p[k].shape)
            h = 1e-6
            pp = {a: b.copy() for a, b in p.items()}
            pm = {a: b.copy() for a, b in p.items()}
            pp[k][idx] += h
            pm[k][idx] -= h
            num = (A.gat_loss_and_grads(pp, g.X, g.y, tr, mask, 5e-4)[0] - A.gat_loss_and_grads(pm, g.X, g.y, tr, mask, 5e-4)[0]) / (2 * h)
            assert gr[k][idx] == pytest.approx(num, rel=1e-4, abs=1e-8)


def test_neighbor_ranking_is_static_for_gat_and_a_dynamic_score_would_break_it():
    g = S.generate(60, 3, 5, 1.0, 0.4, 4.0, 0.0, 4)
    tr = S.split(g.y, 6, 4)
    m = A.train_gat(g.A, g.X, g.y, tr, seed=1, epochs=40)
    alphas, t = A.attention_layer1(m, g.A, g.X)
    assert A.ranking_consistency(alphas, t, g.A) == pytest.approx(1.0)
    # dynamische Bewertung nach GATv2-Art (Nichtlinearität VOR dem Skalarprodukt): die Rangfolge hängt vom fragenden Knoten ab
    rng = np.random.default_rng(3)
    Wh = g.X @ rng.normal(size=(5, 8))
    a = rng.normal(size=8)
    alphas2 = []
    z = np.array([[a @ np.where(v > 0, v, 0.2 * v) for v in (Wh[i][None, :] + Wh)] for i in range(60)])
    E2 = np.where(g.A + np.eye(60) > 0, z, -1e30)
    P = np.exp(E2 - E2.max(axis=1, keepdims=True))
    P /= P.sum(axis=1, keepdims=True)
    alphas2.append(P)
    assert A.ranking_consistency(alphas2, t[:, :1], g.A) < 0.99


def test_attention_false_keeps_scores_at_zero_and_weights_uniform():
    g = S.generate(40, 3, 4, 1.0, 0.3, 4.0, 0.0, 5)
    tr = S.split(g.y, 4, 5)
    m = A.train_gat(g.A, g.X, g.y, tr, seed=2, epochs=15, attention=False)
    assert all(np.all(m.params[k] == 0.0) for k in ("as1", "ad1", "as2", "ad2"))
    alphas, _ = A.attention_layer1(m, g.A, g.X)
    deg = (g.A + np.eye(40)).sum(axis=1)
    for al in alphas:
        assert al == pytest.approx((g.A + np.eye(40)) / deg[:, None])


def test_gat_training_is_reproducible_reduces_the_loss_and_fits_the_known_nodes():
    g = S.generate(120, 3, 5, 1.0, 0.0, 0.0, 0.0, 8)
    tr = S.split(g.y, 6, 8)
    a = A.train_gat(g.A, g.X, g.y, tr, seed=3, epochs=120)
    b = A.train_gat(g.A, g.X, g.y, tr, seed=3, epochs=120)
    assert all(np.array_equal(a.params[k], b.params[k]) for k in a.params)
    assert a.history["loss"][-1] < 0.5 * a.history["loss"][0] and a.history["train_acc"][-1] == 1.0
    pred = A.predict_gat(a, g.A, g.X)
    assert float((pred[~tr] == g.y[~tr]).mean()) == pytest.approx(a.history["test_acc"][-1])


def test_recorded_predictions_match_the_final_prediction():
    g = S.generate(60, 3, 5, 1.0, 0.3, 4.0, 0.0, 9)
    tr = S.split(g.y, 4, 9)
    m = A.train_gat(g.A, g.X, g.y, tr, seed=1, epochs=12, record_pred=True)
    assert m.history["pred"].shape == (12, 60) and np.array_equal(m.history["pred"][-1], A.predict_gat(m, g.A, g.X))


# --- GCN / MLP / Orakel -----------------------------------------------------------------------------------------------------------------------


def test_normalized_adjacency_by_hand_on_a_path_of_three_nodes():
    Ah = A.normalized_adjacency(path3())
    assert Ah[0, 0] == pytest.approx(1 / 2) and Ah[0, 1] == pytest.approx(1 / np.sqrt(6)) and Ah[1, 1] == pytest.approx(1 / 3) and Ah[0, 2] == 0.0


def test_gcn_gradients_agree_with_central_differences_also_for_a_nonsymmetric_propagation_matrix():
    g = S.generate(30, 3, 4, 1.0, 0.3, 4.0, 0.0, 2)
    tr = S.split(g.y, 4, 2)
    Ah = A.oracle_ahat(g.A, g.U)                                             # nicht symmetrisch
    assert not np.allclose(Ah, Ah.T)
    p = {"W1": np.random.default_rng(1).normal(size=(5, 6)) * 0.3, "W2": np.random.default_rng(2).normal(size=(6, 3)) * 0.3}
    _, gr = A.gcn_loss_and_grads(Ah, g.X, g.y, tr, p, 5e-4)
    for k in p:
        for i in range(3):
            idx = (i, i % p[k].shape[1])
            h = 1e-6
            pp = {a: b.copy() for a, b in p.items()}
            pm = {a: b.copy() for a, b in p.items()}
            pp[k][idx] += h
            pm[k][idx] -= h
            num = (A.gcn_loss_and_grads(Ah, g.X, g.y, tr, pp, 5e-4)[0] - A.gcn_loss_and_grads(Ah, g.X, g.y, tr, pm, 5e-4)[0]) / (2 * h)
            assert gr[k][idx] == pytest.approx(num, rel=1e-4, abs=1e-8)


def test_mlp_is_the_gcn_without_edges():
    g = S.generate(50, 3, 5, 1.0, 0.3, 4.0, 0.0, 4)
    tr = S.split(g.y, 4, 4)
    m1 = A.train_gcn(g.A, g.X, g.y, tr, seed=2, use_graph=False, epochs=15)
    m2 = A.train_gcn(np.zeros((50, 50)), g.X, g.y, tr, seed=2, use_graph=True, epochs=15)
    assert all(np.allclose(m1.params[k], m2.params[k]) for k in m1.params)


def test_oracle_matrix_by_hand():
    """Pfad 0-1-2, Knoten 1 unzuverlässig (w = 0,2): Zeile 0: Selbst 1, Nachbar 1 mit 0,2 -> (1, 0,2, 0)/1,2; Zeile 1: Selbst 1, Nachbarn 0 und 2 mit 1 -> (1, 1, 1)/3; Zeile 2 wie Zeile 0 gespiegelt."""
    Ah = A.oracle_ahat(path3(), np.array([False, True, False]), 0.2)
    assert Ah[0] == pytest.approx([1 / 1.2, 0.2 / 1.2, 0.0]) and Ah[1] == pytest.approx([1 / 3, 1 / 3, 1 / 3]) and Ah[2] == pytest.approx([0.0, 0.2 / 1.2, 1 / 1.2])
    assert Ah.sum(axis=1) == pytest.approx(np.ones(3))


def test_neighbor_mass_by_hand():
    """Pfad 0-1-2 mit gleichem Gewicht: Knoten 0 hat den Nachbarn 1 (unzuverlässig): Anteil 1; Knoten 1 hat 0 und 2 (zuverlässig): 0; Knoten 2 wie 0 -> Mittel (1 + 0 + 1)/3 = 2/3."""
    U = np.array([False, True, False])
    assert A.neighbor_mass(np.ones((3, 3)), path3(), U) == pytest.approx(2 / 3)
    W = np.array([[1, 3, 0], [1, 1, 3], [0, 1, 1]], dtype=float)                       # Knoten 1: 1 auf Nachbar 0, 3 auf Nachbar 2 (beide zuverlässig)
    assert A.neighbor_mass(W, path3(), U) == pytest.approx(2 / 3)
    assert len(A.head_masses([np.ones((3, 3))] * 4, path3(), U)) == 4

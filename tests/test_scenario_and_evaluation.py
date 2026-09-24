"""Vehikel (Reproduzierbarkeit, unzuverlässige Kunden, Kennzeichen, falsche Kanten, Aufteilung) und Auswertung (Analyse, vier Experimente) - schnelle Parameter über Funktionsargumente."""

import numpy as np
import pytest

import gat_constants as C
import gat_evaluation as E
import gat_scenario as S


def test_generate_is_reproducible_and_shaped():
    a, b = S.generate(100, 3, 5, 1.0, 0.5, 6.0, 0.0, 4), S.generate(100, 3, 5, 1.0, 0.5, 6.0, 0.0, 4)
    assert np.array_equal(a.X, b.X) and np.array_equal(a.A, b.A) and np.array_equal(a.U, b.U) and a.X.shape == (100, C.N_FEATURES) and set(np.unique(a.y)) == {0, 1, 2}
    assert not np.array_equal(a.X, S.generate(100, 3, 5, 1.0, 0.5, 6.0, 0.0, 5).X)


def test_unreliable_share_flag_and_extra_noise():
    g = S.generate(400, 3, 5, 1.0, 0.5, 6.0, 0.0, 2)
    assert abs(g.U.mean() - 0.5) < 0.08
    assert g.X[g.U, -1].mean() > 0.9 and abs(g.X[~g.U, -1].mean()) < 0.1 and g.X[g.U, -1].std() < 0.2                  # Kennzeichen 1 gegen 0
    assert g.X[g.U, :-1].std() > 3 * g.X[~g.U, :-1].std()                                                              # unzuverlässig = viel mehr Rauschen
    assert S.generate(100, 3, 5, 1.0, 0.0, 6.0, 0.0, 2).U.sum() == 0


def test_layout_and_features_do_not_depend_on_the_unreliable_share():
    a, b = S.generate(120, 3, 5, 1.0, 0.0, 0.0, 0.0, 6), S.generate(120, 3, 5, 1.0, 0.7, 6.0, 0.0, 6)
    assert np.array_equal(a.xy, b.xy) and np.array_equal(a.y, b.y) and np.array_equal(a.A, b.A)
    assert np.allclose(a.X[~b.U, :-1], b.X[~b.U, :-1])                                                                 # zuverlässige Kunden unverändert


def test_adjacency_is_symmetric_without_loops_and_rewiring_keeps_the_edge_count():
    g = S.generate(150, 3, 6, 1.0, 0.3, 4.0, 0.0, 2)
    assert np.array_equal(g.A, g.A.T) and np.all(np.diag(g.A) == 0) and g.A.sum(axis=1).min() >= 6 and g.edge_homophily() > 0.85
    homs = [S.generate(150, 3, 6, 1.0, 0.3, 4.0, w, 2) for w in (0.0, 0.5, 1.0)]
    assert [h.n_edges() for h in homs] == [g.n_edges()] * 3
    hs = [h.edge_homophily() for h in homs]
    assert hs == sorted(hs, reverse=True) and hs[-1] < 0.45


def test_split_gives_exactly_per_class_known_labels():
    y = np.array([0] * 30 + [1] * 30 + [2] * 30)
    tr = S.split(y, 5, 1)
    assert tr.sum() == 15 and all(tr[y == c].sum() == 5 for c in range(3)) and np.array_equal(tr, S.split(y, 5, 1))


def test_analyse_is_consistent():
    a = E.analyse(E.Settings(n=100, seed=3))
    assert a.gat.history["pred"].shape == (C.EPOCHS, 100) and a.gcn.history["pred"].shape == (C.EPOCHS, 100) and np.array_equal(a.gat.history["pred"][-1], a.pred_gat)
    assert a.acc_gat == pytest.approx(a.gat.history["test_acc"][-1]) and a.acc_gcn == pytest.approx(a.gcn.history["test_acc"][-1]) and a.acc_mlp == pytest.approx(a.mlp.history["test_acc"][-1])
    assert a.acc_uniform == pytest.approx(a.uniform.history["test_acc"][-1]) and 0 <= a.acc_oracle <= 1 and 0 < a.majority_rate() < 1
    assert len(a.alphas) == C.HEADS and a.mean_alpha().shape == (100, 100) and a.gcn_weights().shape == (100, 100)


def test_settings_extremes_run():
    for st in (E.Settings(100, 2, 2, 2, 0.5, 0.0, 0.0, 0.0, 0), E.Settings(100, 4, 40, 10, 3.0, 0.8, 8.0, 1.0, 1)):
        a = E.analyse(st)
        assert 0 <= a.acc_gat <= 1 and 0 <= a.acc_gcn <= 1


def test_experiments_shape_and_paired_differences():
    base = E.Settings(n=100, labels=10)
    rows = E.labels_experiment(levels=(5, 10), seeds=range(3), base=base)
    assert [r["labels"] for r in rows] == [5, 10] and all(r["n_seeds"] == 3 and 0 <= r["gat_wins"] <= 3 and r["diff"] == pytest.approx(r["gat"] - r["gcn"]) for r in rows)
    assert all(0.9 <= r["ranking"] <= 1.0 + 1e-12 and 0 <= r["mass_gat"] <= 1 for r in rows)
    rows = E.unreliable_experiment(levels=(0.0, 0.6), seeds=range(3), base=base)
    assert rows[0]["mass_gat"] == 0.0 and rows[0]["share_unreliable_neighbors"] == 0.0 and rows[1]["share_unreliable_neighbors"] > 0.3
    rows = E.wrong_experiment(levels=(0.0, 0.6), seeds=range(3), base=E.Settings(n=100, labels=10, unreliable=0.0, extra=0.0))
    assert rows[0]["homophily"] > rows[1]["homophily"] and rows[0]["mlp"] == pytest.approx(rows[1]["mlp"])


def test_ablation_experiment_shape():
    r = E.ablation_experiment(seeds=range(3), base=E.Settings(n=100, labels=10))
    assert r["n_seeds"] == 3 and r["ranking"] == pytest.approx(1.0) and 0 < r["mass_trained"] < 1 and 0 < r["mass_init"] < 1
    assert r["gat_minus_uniform"] == pytest.approx(r["gat"] - r["uniform"]) and r["uniform_minus_gcn"] == pytest.approx(r["uniform"] - r["gcn"])

"""Jede Zahl aus PRESET_HELP und README: Einzelgebiete (Presets) mit Bändern von etwa 2 Punkten, Mehr-Seed-Befunde mit Bändern um den Standardfehler (CI installiert die neueste numpy)."""

import numpy as np
import pytest

import gat_constants as C
import gat_evaluation as E
import gat_presets as P


def analyse_preset(name):
    p = P.PRESETS[name]
    return E.analyse(E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["unreliable"], p["extra"], p["wrong"], p["seed"]))


# --- Presets (Einzelgebiete: ein Knoten = 0,5 bis 1 Punkt) --------------------------------------------------------------------------------------


def test_preset_standard():
    a = analyse_preset("Standardfall")
    assert a.acc_gat == pytest.approx(0.886, abs=0.02) and a.acc_gcn == pytest.approx(0.829, abs=0.02) and a.acc_uniform == pytest.approx(0.807, abs=0.02) and a.acc_mlp == pytest.approx(0.493, abs=0.02)
    assert a.acc_oracle == pytest.approx(0.786, abs=0.02) and int(a.graph.U.sum()) == 104 and a.graph.n_edges() == 608 and int((~a.train_mask).sum()) == 140


def test_preset_five_labels():
    a = analyse_preset("Nur 5 Etiketten je Gebietstyp")
    assert int(a.train_mask.sum()) == 15 and a.acc_gat == pytest.approx(0.557, abs=0.02) and a.acc_gcn == pytest.approx(0.503, abs=0.02) and a.acc_mlp == pytest.approx(0.497, abs=0.02) and a.majority_rate() == pytest.approx(0.357, abs=0.002)


def test_preset_all_reliable():
    a = analyse_preset("Alle Kunden zuverlässig")
    assert int(a.graph.U.sum()) == 0 and a.acc_gat == pytest.approx(0.971, abs=0.02) and a.acc_gcn == pytest.approx(0.971, abs=0.02) and a.acc_uniform == pytest.approx(0.964, abs=0.02) and a.acc_mlp == pytest.approx(0.857, abs=0.02)
    assert abs(a.acc_gat - a.acc_gcn) < 0.03


def test_preset_many_unreliable():
    a = analyse_preset("Viele unzuverlässige (80 %)")
    assert int(a.graph.U.sum()) == 157 and a.acc_gat == pytest.approx(0.879, abs=0.02) and a.acc_gcn == pytest.approx(0.807, abs=0.02) and a.acc_uniform == pytest.approx(0.793, abs=0.02)
    assert a.acc_mlp == pytest.approx(0.386, abs=0.02) and a.majority_rate() == pytest.approx(0.364, abs=0.002)


def test_preset_wrong_edges():
    a = analyse_preset("Falsche Kanten")
    assert a.graph.edge_homophily() == pytest.approx(0.579, abs=0.002) and a.acc_gat == pytest.approx(0.764, abs=0.02) and a.acc_gcn == pytest.approx(0.829, abs=0.02) and a.acc_mlp == pytest.approx(0.857, abs=0.02)
    assert a.acc_gat < a.acc_gcn < a.acc_mlp


def test_preset_forty_labels():
    a = analyse_preset("40 Etiketten, 300 Kunden")
    assert a.acc_gat == pytest.approx(0.878, abs=0.02) and a.acc_gcn == pytest.approx(0.778, abs=0.02) and a.acc_uniform == pytest.approx(0.828, abs=0.02) and a.acc_oracle == pytest.approx(0.878, abs=0.02)


# --- Experimente (8 feste Seeds) ---------------------------------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def labels_rows():
    return {r["labels"]: r for r in E.labels_experiment()}


def test_labels_experiment(labels_rows):
    r = labels_rows
    assert r[5]["gat"] == pytest.approx(0.635, abs=0.04) and r[5]["gcn"] == pytest.approx(0.567, abs=0.04) and r[20]["gat"] == pytest.approx(0.866, abs=0.04) and r[20]["gcn"] == pytest.approx(0.769, abs=0.04)
    assert r[40]["gat"] == pytest.approx(0.866, abs=0.04) and r[40]["gcn"] == pytest.approx(0.829, abs=0.04)
    assert all(r[m]["diff"] > -0.02 for m in r) and all(r[m]["gat"] > r[m]["mlp"] + 0.05 for m in r)          # in allen vier Stufen nicht schlechter als GCN
    assert r[20]["diff"] == pytest.approx(0.097, abs=0.04) and r[20]["diff"] > 2 * r[20]["diff_se"] and r[20]["gat_wins"] >= 6                    # gemessen +9,7 +- 2,3 (8 von 8)
    assert sum(r[m]["diff"] > 2 * r[m]["diff_se"] for m in r) <= 2                                                                          # deutlich nur bei einer, höchstens zwei Stufen
    assert all(r[m]["ranking"] == pytest.approx(1.0) and 0.65 < r[m]["mass_gat"] < 0.85 and r[m]["share_unreliable_neighbors"] == pytest.approx(0.509, abs=0.01) for m in r)


@pytest.fixture(scope="module")
def unrel_rows():
    return {r["unreliable"]: r for r in E.unreliable_experiment()}


def test_unreliable_experiment(unrel_rows):
    r = unrel_rows
    assert r[0.0]["gat"] == pytest.approx(0.930, abs=0.04) and r[0.0]["gcn"] == pytest.approx(0.943, abs=0.04) and abs(r[0.0]["diff"]) < 0.04 and r[0.0]["mass_gat"] == 0.0
    assert r[0.6]["diff"] == pytest.approx(0.080, abs=0.04) and r[0.6]["diff"] > 2 * r[0.6]["diff_se"] and r[0.6]["gat_wins"] >= 5
    assert np.mean([r[u]["diff"] for u in (0.2, 0.4, 0.6, 0.8)]) > 0.01 and max(r[u]["diff_se"] for u in r) < 0.05                       # im Mittel positiv, aber mit großem Standardfehler
    assert all(r[u]["gat"] > r[u]["mlp"] for u in r) and r[0.8]["mass_gat"] > r[0.8]["share_unreliable_neighbors"] - 0.02


@pytest.fixture(scope="module")
def wrong_rows():
    return {r["wrong"]: r for r in E.wrong_experiment()}


def test_wrong_edges_experiment(wrong_rows):
    w = wrong_rows
    assert w[0.6]["homophily"] == pytest.approx(0.592, abs=0.02) and w[0.0]["homophily"] == pytest.approx(0.916, abs=0.02)
    assert w[0.6]["diff"] == pytest.approx(-0.086, abs=0.04) and w[0.6]["diff"] < -2 * w[0.6]["diff_se"] and w[0.6]["gat_wins"] <= 2 and w[0.4]["diff"] < 0.01              # GAT hinter GCN
    assert all(w[x]["diff"] < 0.03 for x in w) and all(w[x]["ranking"] == pytest.approx(1.0) for x in w)
    diffs = [w[x]["diff"] for x in sorted(w)]
    assert diffs[-1] == min(diffs)                                                                                                          # am schlechtesten bei den meisten falschen Kanten


@pytest.fixture(scope="module")
def ablation():
    return E.ablation_experiment()


def test_ablation_experiment(ablation):
    r = ablation
    assert r["gat"] == pytest.approx(0.866, abs=0.04) and r["uniform"] == pytest.approx(0.772, abs=0.04) and r["gcn"] == pytest.approx(0.769, abs=0.04) and r["mlp"] == pytest.approx(0.502, abs=0.04) and r["oracle"] == pytest.approx(0.789, abs=0.04)
    assert r["gat_minus_uniform"] == pytest.approx(0.094, abs=0.04) and r["gat_minus_uniform"] > 3 * r["gat_minus_uniform_se"]                                # die Aufmerksamkeit, nicht die Köpfe
    assert abs(r["uniform_minus_gcn"]) < 0.03                                                                                                        # gemessen +0,4 +- 0,8
    assert r["gat"] > r["oracle"] + 0.03                                                                                                            # von Hand herunterwichten reicht nicht
    assert r["mass_trained"] == pytest.approx(0.738, abs=0.05) and r["mass_init"] == pytest.approx(0.604, abs=0.05) and r["share_unreliable"] == pytest.approx(0.509, abs=0.01)
    assert r["mass_trained"] > r["share_unreliable"] + 0.10 and r["mass_trained"] > r["mass_init"]                                                   # MEHR Gewicht auf unzuverlässige Nachbarn, nicht weniger
    assert 0.05 < r["head_spread"] < 0.35 and r["ranking"] == pytest.approx(1.0)


def test_constants_used_in_the_texts():
    assert C.HEADS == 4 and C.HEAD_DIM == 8 and C.HIDDEN_GCN == 16 and C.EPOCHS == 200 and C.LEAKY_SLOPE == pytest.approx(0.2) and C.EXP_SEEDS == tuple(range(8))

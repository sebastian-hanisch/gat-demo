"""AppTest-Rauchtests: Voreinstellung, jedes Preset, Kartenansichten, Kundenwahl, Würfel-Knopf, Permalink-Grenzen, Extremwerte, vier Experimente auf Abruf, Footer."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import gat_constants as C
import gat_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")


def _run(**state):
    at = AppTest.from_file(APP, default_timeout=300)
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    return at


def _ok(at):
    assert not at.exception, [e.value for e in at.exception]


def test_default_run_shows_gat_ahead_of_gcn():
    at = _run()
    _ok(at)
    assert at.metric and any("GAT liegt vorn" in s.value for s in at.success)


@pytest.mark.parametrize("name", list(P.PRESETS))
def test_every_preset_button_runs(name):
    at = _run()
    next(b for b in at.button if b.key == f"preset_{name}").click().run()
    _ok(at)
    p = P.PRESETS[name]
    for key, state_key in P.PRESET_KEYS.items():
        assert at.session_state[state_key] == p[key]
    assert at.get("plotly_chart")


def test_wrong_edges_preset_shows_gat_behind():
    at = _run()
    next(b for b in at.button if b.key == "preset_Falsche Kanten").click().run()
    _ok(at)
    assert any("GAT liegt hinten" in w.value for w in at.warning)


def test_all_map_views_and_epoch_slider_run():
    at = _run()
    for view in ("Vorhersage des GCN", "Wahrer Gebietstyp", "Vorhersage des GAT"):
        at.radio(key="map_mode").set_value(view).run()
        _ok(at)
    at.slider(key="epoch_slider").set_value(1).run()
    _ok(at)


def test_node_selection_survives_shrinking_n_and_shows_the_neighborhood():
    at = _run(n_slider=400, node_select=399)
    _ok(at)
    at.slider(key="n_slider").set_value(120).run()
    _ok(at)
    assert at.session_state["node_select"] < 120 and any("Nachbarn, davon" in c.value for c in at.caption)


def test_dice_button_changes_the_seed():
    at = _run()
    old = at.session_state["seed_input"]
    next(b for b in at.button if b.label == "🎲 Neues Gebiet generieren").click().run()
    _ok(at)
    assert at.session_state["seed_input"] != old


def test_permalink_values_are_snapped_and_clamped():
    at = AppTest.from_file(APP, default_timeout=300)
    at.query_params["n"] = "9999"
    at.query_params["unrel"] = "0.33"
    at.query_params["extra"] = "7.3"
    at.query_params["labels"] = "abc"
    at.query_params["wrong"] = "5"
    at.run()
    _ok(at)
    assert at.session_state["n_slider"] == C.N_MAX and at.session_state["unrel_slider"] == 0.35 and at.session_state["extra_slider"] == 7.5
    assert at.session_state["wrong_slider"] == C.WRONG_MAX and at.session_state["labels_slider"] == C.DEFAULT_LABELS


@pytest.mark.parametrize("kw", [dict(n_slider=C.N_MIN, classes_slider=2, labels_slider=40), dict(n_slider=C.N_MAX, classes_slider=4, neighbors_slider=10), dict(unrel_slider=0.0), dict(unrel_slider=C.UNREL_MAX, extra_slider=0.0),
                                dict(wrong_slider=1.0), dict(labels_slider=C.LABELS_MIN, neighbors_slider=C.NEIGHBORS_MIN)])
def test_extreme_settings_run(kw):
    _ok(_run(**kw))


def test_labels_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "LABEL_LEVELS", (5, 20))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "labels_start").click().run()
    _ok(at)
    assert at.session_state["labels_on"] and any("GAT minus GCN - 5 Etiketten" in w.value for w in at.warning)


def test_unreliable_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "UNREL_LEVELS", (0.0, 0.6))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "unrel_start").click().run()
    _ok(at)
    assert at.session_state["unrel_on"] and any("Ohne unzuverlässige Kunden" in w.value for w in at.warning)


def test_wrong_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "WRONG_LEVELS", (0.0, 0.6))
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "wrong_start").click().run()
    _ok(at)
    assert at.session_state["wrong_on"] and any("Die Aufmerksamkeit schützt nicht vor falschen Kanten" in w.value for w in at.warning)


def test_ablation_experiment_runs_on_demand(monkeypatch):
    monkeypatch.setattr(C, "EXP_SEEDS", (0, 1))
    at = _run()
    next(b for b in at.button if b.key == "ablation_start").click().run()
    _ok(at)
    assert at.session_state["ablation_on"] and any("dieselbe Architektur ohne gelernte Aufmerksamkeit" in w.value for w in at.warning)


def test_footer_and_grenzen_are_present_and_no_unresolved_f_strings():
    at = _run()
    assert any("Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net)" in c.value for c in at.caption)
    assert any("Wo die Annahmen enden" in s.value for s in at.subheader)
    for el in list(at.caption) + list(at.markdown) + list(at.warning) + list(at.success) + list(at.info):
        assert "{de(" not in el.value and "{pct(" not in el.value and "{pts(" not in el.value

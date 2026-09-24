"""SETTING_SPECS-Permalink-Muster, Presets und Zufalls-Seed-Button (Standardmuster des Portfolios, vgl. gcn_presets.py)."""

import math
import random
from dataclasses import dataclass
from typing import Callable, Optional

import streamlit as st

import gat_constants as C


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: Callable
    default: object
    lo: Optional[float] = None
    hi: Optional[float] = None


SETTING_SPECS = {
    "n_slider": SettingSpec("n", int, C.DEFAULT_N, C.N_MIN, C.N_MAX),
    "classes_slider": SettingSpec("classes", int, C.DEFAULT_CLASSES, C.CLASSES_MIN, C.CLASSES_MAX),
    "labels_slider": SettingSpec("labels", int, C.DEFAULT_LABELS, C.LABELS_MIN, C.LABELS_MAX),
    "neighbors_slider": SettingSpec("neighbors", int, C.DEFAULT_NEIGHBORS, C.NEIGHBORS_MIN, C.NEIGHBORS_MAX),
    "noise_slider": SettingSpec("noise", float, C.DEFAULT_NOISE, C.NOISE_MIN, C.NOISE_MAX),
    "unrel_slider": SettingSpec("unrel", float, C.DEFAULT_UNREL, C.UNREL_MIN, C.UNREL_MAX),
    "extra_slider": SettingSpec("extra", float, C.DEFAULT_EXTRA, C.EXTRA_MIN, C.EXTRA_MAX),
    "wrong_slider": SettingSpec("wrong", float, C.DEFAULT_WRONG, C.WRONG_MIN, C.WRONG_MAX),
    "seed_input": SettingSpec("seed", int, 7, 0, C.SEED_MAX),
}
PRESET_KEYS = {"n": "n_slider", "classes": "classes_slider", "labels": "labels_slider", "neighbors": "neighbors_slider", "noise": "noise_slider", "unreliable": "unrel_slider", "extra": "extra_slider",
               "wrong": "wrong_slider", "seed": "seed_input"}
STEPS = {"n_slider": C.N_STEP, "noise_slider": C.NOISE_STEP, "unrel_slider": C.UNREL_STEP, "extra_slider": C.EXTRA_STEP, "wrong_slider": C.WRONG_STEP}


def _p(**kw):
    base = {"n": 200, "classes": 3, "labels": 20, "neighbors": 5, "noise": 1.0, "unreliable": 0.5, "extra": 6.0, "wrong": 0.0, "seed": 7}
    base.update(kw)
    return base


PRESETS = {
    "Standardfall": _p(),
    "Nur 5 Etiketten je Gebietstyp": _p(labels=5, seed=8),
    "Alle Kunden zuverlässig": _p(unreliable=0.0),
    "Viele unzuverlässige (80 %)": _p(unreliable=0.8, seed=8),
    "Falsche Kanten": _p(unreliable=0.0, extra=0.0, wrong=0.6),
    "40 Etiketten, 300 Kunden": _p(labels=40, n=300, seed=1),
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def load_permalink_settings():
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
                if isinstance(value, float) and not math.isfinite(value):
                    continue
                st.session_state[state_key] = max(spec.lo, min(spec.hi, value))
            except (ValueError, TypeError):
                pass
    for key, step in STEPS.items():
        if key in st.session_state:
            spec = SETTING_SPECS[key]
            snapped = spec.lo + round((st.session_state[key] - spec.lo) / step) * step
            snapped = min(spec.hi, max(spec.lo, snapped))
            st.session_state[key] = int(snapped) if isinstance(spec.default, int) else round(float(snapped), 2)
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        st.session_state[state_key] = PRESETS[name][key]


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)


PRESET_HELP = {
    "Standardfall": "Ein Einzelgebiet (200 Kunden, 104 davon unzuverlässig, 20 Etiketten je Typ): GAT 88,6 %, GCN 82,9 %, dieselbe Architektur ohne gelernte Aufmerksamkeit 80,7 %, MLP 49,3 %; ein GCN mit bekannter Zuverlässigkeit 78,6 %. Die Experimente unten mitteln über acht Gebiete.",
    "Nur 5 Etiketten je Gebietstyp": "Ein Einzelgebiet mit 15 bekannten Kunden: GAT 55,7 %, GCN 50,3 %, MLP 49,7 % (Raten: 35,7 %). Mit so wenigen Etiketten ist der Vorsprung klein und unsicher (Experiment 1).",
    "Alle Kunden zuverlässig": "Kein unzuverlässiger Kunde: GAT und GCN liegen mit 97,1 % gleichauf (ohne Aufmerksamkeit 96,4 %, MLP 85,7 %) - die Aufmerksamkeit bringt hier nichts.",
    "Viele unzuverlässige (80 %)": "157 von 200 Kunden unzuverlässig: GAT 87,9 %, GCN 80,7 %, ohne Aufmerksamkeit 79,3 %, MLP 38,6 % (Raten: 36,4 %). Ein Einzelgebiet.",
    "Falsche Kanten": "60 % der Kanten zufällig ersetzt (Homophilie 57,9 %), keine unzuverlässigen Kunden: GAT 76,4 % liegt hinter dem GCN (82,9 %), und beide hinter dem MLP ohne Nachbarn (85,7 %). Aufmerksamkeit schützt nicht vor falschen Kanten.",
    "40 Etiketten, 300 Kunden": "Ein Einzelgebiet mit 300 Kunden und 40 Etiketten je Typ: GAT 87,8 %, GCN 77,8 %, ohne Aufmerksamkeit 82,8 %; das GCN mit bekannter Zuverlässigkeit erreicht ebenfalls 87,8 %.",
}

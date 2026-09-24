"""Presets und Permalink-Werte: Vollständigkeit, gültige Werte, Grenzen/Schrittweiten - reine Datenprüfungen ohne Streamlit-Session."""

import gat_constants as C
import gat_evaluation as E
import gat_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(P.PRESETS) == set(P.PRESET_HELP)
    for name, p in P.PRESETS.items():
        assert set(p) == set(P.PRESET_KEYS) and P.PRESET_HELP[name]


def test_preset_values_are_valid_and_on_the_slider_grid():
    for p in P.PRESETS.values():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            assert spec.lo <= p[key] <= spec.hi
            spec.caster(p[key])
        assert (p["n"] - C.N_MIN) % C.N_STEP == 0
        for key, state_key in (("noise", "noise_slider"), ("unreliable", "unrel_slider"), ("extra", "extra_slider"), ("wrong", "wrong_slider")):
            spec, step = P.SETTING_SPECS[state_key], P.STEPS[state_key]
            k = (p[key] - spec.lo) / step
            assert abs(k - round(k)) < 1e-9


def test_default_preset_equals_the_default_settings():
    p = P.PRESETS["Standardfall"]
    assert E.Settings(p["n"], p["classes"], p["labels"], p["neighbors"], p["noise"], p["unreliable"], p["extra"], p["wrong"], p["seed"]) == E.Settings()


def test_bounds_and_steps_constants():
    assert P.bounds("n_slider") == (C.N_MIN, C.N_MAX) and P.bounds("unrel_slider") == (C.UNREL_MIN, C.UNREL_MAX) and set(P.STEPS) == {"n_slider", "noise_slider", "unrel_slider", "extra_slider", "wrong_slider"}


def test_url_params_are_unique():
    assert len({spec.url_param for spec in P.SETTING_SPECS.values()}) == len(P.SETTING_SPECS)

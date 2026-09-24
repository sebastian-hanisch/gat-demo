"""GAT - Graph Attention Network - Aufmerksamkeit auf Nachbarn - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Zweites Stück der Graph-Neural-Network-Linie der "Konzepte"-Reihe (Nachfolger von gcn-demo): dasselbe Liefergebiet, aber ein Teil der Kunden ist unzuverlässig. Ein GCN mittelt alle Nachbarn gleich; GAT lernt Gewichte.

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import streamlit as st

import gat_algorithm as A
import gat_constants as C
from gat_evaluation import Settings, ablation_experiment, analyse, labels_experiment, unreliable_experiment, wrong_experiment
from gat_presets import PRESET_HELP, PRESETS, apply_preset, bounds, init_session_state_defaults, load_permalink_settings, randomize_seed, sync_query_params
from gat_visualization import build_ablation, build_attention_view, build_curves, build_heads, build_labels, build_map, build_unreliable, build_wrong

st.set_page_config(page_title="GAT – Sebastian Hanisch", layout="wide")


def de(x, digits=1):
    """Deutsche Zahlenschreibweise: Punkt als Tausendertrenner, Komma als Dezimalzeichen."""
    x = round(float(x), digits)
    if x == 0:
        x = 0.0
    return f"{x:,.{digits}f}".replace(",", "#").replace(".", ",").replace("#", ".")


def pct(x, digits=0):
    return f"{de(100 * x, digits)} %"


def pts(x, se=None, digits=1):
    """Differenz in Prozentpunkten mit Vorzeichen, optional mit Standardfehler."""
    s = f"{'+' if x >= 0 else '−'}{de(abs(100 * x), digits)}"
    return s + (f" ± {de(100 * se, digits)}" if se is not None else "")


@st.cache_data(show_spinner=False)
def _labels(levels, seeds, base):
    return labels_experiment(levels=levels, seeds=seeds, base=base)


@st.cache_data(show_spinner=False)
def _unreliable(levels, seeds, base):
    return unreliable_experiment(levels=levels, seeds=seeds, base=base)


@st.cache_data(show_spinner=False)
def _wrong(levels, seeds, base):
    return wrong_experiment(levels=levels, seeds=seeds, base=base)


@st.cache_data(show_spinner=False)
def _ablation(seeds, base):
    return ablation_experiment(seeds=seeds, base=base)


st.title("🎯 GAT – lernbare Aufmerksamkeit auf Nachbarn")
st.markdown(
    """
Im Vorgänger-Stück mittelte ein **GCN** die Merkmale eines Kunden mit denen aller Nachbarn - jeder zählt nur nach der Zahl seiner Nachbarn, nie nach seinem **Inhalt**. Hier ist ein Teil der Kunden **unzuverlässig**
(lückenhafte Telematik: stark verrauschte Merkmale, aber ein Kennzeichen zeigt es an). Ein **Graph Attention Network** (Veličković et al. 2018) lernt stattdessen für jeden Nachbarn ein Gewicht aus den Merkmalen beider
Kunden. Die Demo misst, wann das etwas bringt, was es dafür braucht - und was die gelernte Aufmerksamkeit **nicht** tut.
"""
)
st.caption(
    "Zweites Stück der **Graph-Neural-Network-Linie** der \"Konzepte\"-Reihe, Nachfolger von **gcn-demo** (dasselbe Liefergebiet-Vehikel, erweitert um unzuverlässige Kunden; das Netz ist von Grund auf in numpy geschrieben). "
    "**Bezug zu OR:** Datenqualität ist ein Alltagsproblem der Planung - eine unzuverlässige Telematikmeldung verfälscht, was in Prognose und Tourenplanung als Eingabe ankommt."
)

with st.expander("So funktioniert ein GAT", expanded=True):
    st.markdown(
        """
1. **Bewertung.** Für Kunde $i$ und jeden Nachbarn $j$ (und $i$ selbst): $e_{ij} = \\mathrm{LeakyReLU}(a_{\\text{Empfänger}}\\cdot Wh_i + a_{\\text{Sender}}\\cdot Wh_j)$.
2. **Aufmerksamkeit.** $\\alpha_{ij} = \\mathrm{softmax}_j(e_{ij})$ - die Gewichte eines Kunden summieren sich zu 1. Ein GCN würde stattdessen $1/\\sqrt{(d_i+1)(d_j+1)}$ nehmen.
3. **Schicht.** $h_i = \\sum_j \\alpha_{ij}\\,Wh_j$, mit vier **Köpfen** (je ein eigenes $W$ und eigene Bewertung), nebeneinandergelegt; darauf eine zweite Schicht mit einem Kopf.
4. **Statisch.** Weil die Bewertung eine **Summe** aus einem Empfänger- und einem Senderanteil ist, ordnet jeder Kunde seine Nachbarn in derselben Reihenfolge wie jeder andere - die Reihenfolge hängt nur vom Nachbarn ab
   (unten gemessen). GAT kann also "diesen Nachbarn mag jeder mehr", aber nicht "diesen Nachbarn passe ich".
5. **Vergleiche.** GCN und MLP wie im Vorgänger; dazu dieselbe Architektur **ohne gelernte Aufmerksamkeit** (alle Nachbarn gleich) und ein **GCN mit bekannter Zuverlässigkeit**, das unzuverlässige Nachbarn von Hand herunterwichtet.
        """
    )

st.caption("🎯 Schnellstart – ein Beispielszenario laden:")
preset_names = list(PRESETS.keys())
for row in (preset_names[:3], preset_names[3:]):
    cols = st.columns(len(row))
    for col, name in zip(cols, row):
        with col:
            st.button(name, width="stretch", on_click=apply_preset, args=(name,), help=PRESET_HELP.get(name), key=f"preset_{name}")

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    st.header("⚙️ Einstellungen")
    n_nodes = st.slider("Kunden", *bounds("n_slider"), key="n_slider", step=C.N_STEP, help="Zahl der Kunden im Gebiet.")
    classes = st.slider("Gebietstypen", *bounds("classes_slider"), key="classes_slider", help="Zahl der räumlich zusammenhängenden Gebietstypen.")
    labels = st.slider("Bekannte Etiketten je Gebietstyp", *bounds("labels_slider"), key="labels_slider", help="Nur diese Kunden verraten dem Netz ihren Gebietstyp; alle anderen werden vorhergesagt und zur Prüfung benutzt.")
    neighbors = st.slider("Nachbarn je Kunde", *bounds("neighbors_slider"), key="neighbors_slider", help="Jeder Kunde wird mit seinen k räumlich nächsten Kunden verbunden.")
    noise = st.slider("Grundrauschen der Merkmale", *bounds("noise_slider"), key="noise_slider", step=C.NOISE_STEP, help="Streuung der Merkmale aller Kunden um den Mittelwert ihres Gebietstyps.")
    unrel = st.slider("Anteil unzuverlässiger Kunden", *bounds("unrel_slider"), key="unrel_slider", step=C.UNREL_STEP, help="Anteil der Kunden mit lückenhafter Telematik: zusätzliches Rauschen und Kennzeichen \"Meldeausfall\".")
    extra = st.slider("Zusatzrauschen der unzuverlässigen", *bounds("extra_slider"), key="extra_slider", step=C.EXTRA_STEP, help="Stärke des zusätzlichen Rauschens bei unzuverlässigen Kunden (0 = kein Unterschied außer dem Kennzeichen).")
    wrong = st.slider("Anteil falscher Kanten", *bounds("wrong_slider"), key="wrong_slider", step=C.WRONG_STEP, help="Anteil der Kanten, die durch zufällige ersetzt werden (die Zahl der Kanten bleibt).")
    seed = st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Legt Gebiet, Merkmale, bekannte Etiketten und Anfangsgewichte fest.")
    st.button("🎲 Neues Gebiet generieren", width="stretch", on_click=randomize_seed)

sync_query_params({"n_slider": int(n_nodes), "classes_slider": int(classes), "labels_slider": int(labels), "neighbors_slider": int(neighbors), "noise_slider": round(float(noise), 2), "unrel_slider": round(float(unrel), 2),
                   "extra_slider": round(float(extra), 2), "wrong_slider": round(float(wrong), 2), "seed_input": int(seed)})

settings = Settings(int(n_nodes), int(classes), int(labels), int(neighbors), round(float(noise), 2), round(float(unrel), 2), round(float(extra), 2), round(float(wrong), 2), int(seed))
with st.spinner("Trainiere GAT, GCN, MLP und die Vergleichsnetze..."):
    a = analyse(settings)
g = a.graph
n = g.n
unknown = ~a.train_mask

# --- Das Gebiet --------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Das Liefergebiet und was die Netze lernen")
c1, c2 = st.columns([1, 1])
with c1:
    epoch = st.slider("Trainingsepoche", 1, C.EPOCHS, C.EPOCHS, key="epoch_slider", help="Wie weit das Netz trainiert ist.")
with c2:
    view = st.radio("Karte zeigt", ["Vorhersage des GAT", "Vorhersage des GCN", "Wahrer Gebietstyp"], horizontal=True, key="map_mode")
model_key = {"Vorhersage des GAT": "gat", "Vorhersage des GCN": "gcn", "Wahrer Gebietstyp": "truth"}[view]
st.plotly_chart(build_map(a, model_key, epoch), width="stretch", key="map_chart")
st.caption(
    f"{n} Kunden, {g.n_edges()} Kanten; {int(g.U.sum())} unzuverlässige Kunden (Quadrate), {int(a.train_mask.sum())} bekannte Etiketten (schwarze Ringe), {int(unknown.sum())} unbekannte Kunden. Ein Kreuz markiert eine falsche Vorhersage."
)

st.markdown("---")

# --- Vergleich ---------------------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 GAT gegen GCN: was bringt die Aufmerksamkeit?")
m1, m2, m3, m4 = st.columns(4)
m1.metric("GAT", pct(a.acc_gat, 1), help="Lernbare Aufmerksamkeit, vier Köpfe.")
m2.metric("GCN", pct(a.acc_gcn, 1), delta=f"{pts(a.acc_gat - a.acc_gcn)} Punkte für GAT", delta_color="off", help="Alle Nachbarn zählen nach ihren Graden.")
m3.metric("GAT ohne Aufmerksamkeit", pct(a.acc_uniform, 1), delta=f"{pts(a.acc_gat - a.acc_uniform)} Punkte für die Aufmerksamkeit", delta_color="off",
          help="Dieselbe Architektur (Köpfe, Merkmale), aber alle Nachbarn zählen gleich: die Bewertungsvektoren werden nicht gelernt.")
m4.metric("MLP (ohne Nachbarn)", pct(a.acc_mlp, 1), help="Dasselbe Netz ohne Nachbarn.")
st.plotly_chart(build_curves(a, epoch), width="stretch", key="curves_chart")
diff = a.acc_gat - a.acc_gcn
if diff > 0.03:
    st.success(f"✅ GAT liegt vorn: {pct(a.acc_gat, 1)} gegen {pct(a.acc_gcn, 1)} beim GCN. Dieselbe Architektur ohne gelernte Aufmerksamkeit erreicht {pct(a.acc_uniform, 1)} - der Vorsprung kommt also von der Aufmerksamkeit, nicht von den "
               "zusätzlichen Köpfen. (Ein Einzelgebiet: die Experimente unten mitteln über acht Gebiete.)")
elif diff < -0.03:
    st.warning(f"⚠️ GAT liegt hinten: {pct(a.acc_gat, 1)} gegen {pct(a.acc_gcn, 1)} beim GCN. Aufmerksamkeit kostet zusätzliche Parameter, die aus {int(a.train_mask.sum())} bekannten Etiketten gelernt werden müssen.")
else:
    st.info(f"Kein klarer Unterschied: GAT {pct(a.acc_gat, 1)}, GCN {pct(a.acc_gcn, 1)} (GAT ohne Aufmerksamkeit {pct(a.acc_uniform, 1)}).")
st.caption(f"Zum Vergleich: GCN mit **bekannter** Zuverlässigkeit (unzuverlässige Nachbarn von Hand auf ein Fünftel herunterwichtet): {pct(a.acc_oracle, 1)}. Raten (häufigster Typ): {pct(a.majority_rate(), 1)}.")

st.markdown("---")

# --- Aufmerksamkeit ansehen -----------------------------------------------------------------------------------------------------------------------

st.markdown("## 🎯 Wohin schaut die Aufmerksamkeit?")
default_node = int(np.argmin(np.linalg.norm(g.xy - C.AREA / 2, axis=1)))
if "node_select" not in st.session_state or st.session_state["node_select"] >= n:
    st.session_state["node_select"] = default_node
node = int(st.number_input("Kunde Nr.", 0, n - 1, key="node_select", step=1, help="Nummer eines Kunden (0 bis n-1)."))
st.plotly_chart(build_attention_view(a, node), width="stretch", key="attention_view")
nb = np.flatnonzero(g.A[node])
w_gat, w_gcn = a.mean_alpha()[node, nb], a.gcn_weights()[node, nb]
order = np.argsort(-w_gat)
rows = [{"Nachbar": int(nb[k]), "Zuverlässig": "nein" if g.U[nb[k]] else "ja", "Wahrer Typ": C.CLASS_NAMES[g.y[nb[k]]], "GCN-Gewicht": de(w_gcn[k], 3), "GAT-Aufmerksamkeit (Mittel der 4 Köpfe)": de(w_gat[k], 3)} for k in order]
st.dataframe(rows, hide_index=True)
st.caption(
    f"Kunde {node} ({'unzuverlässig' if g.U[node] else 'zuverlässig'}) hat {len(nb)} Nachbarn, davon {int(g.U[nb].sum())} unzuverlässig. Die Aufmerksamkeit der Schicht 1 (Mittel der Köpfe) legt {pct(float(w_gat[g.U[nb]].sum() / w_gat.sum()))} des "
    f"Nachbargewichts auf die unzuverlässigen, das GCN {pct(float(w_gcn[g.U[nb]].sum() / w_gcn.sum()))}. Eigengewicht des Kunden: GAT {de(a.mean_alpha()[node, node], 3)}, GCN {de(a.gcn_weights()[node, node], 3)}."
)
st.markdown("##### Was jeder Kopf über alle Kunden tut")
st.plotly_chart(build_heads(a), width="stretch", key="heads_chart")
masses = A.head_masses(a.alphas, g.A, g.U)
share = A.neighbor_mass(np.ones_like(g.A), g.A, g.U)
st.caption(
    f"Die vier Köpfe der ersten Schicht legen {pct(min(masses))} bis {pct(max(masses))} ihres Nachbargewichts auf unzuverlässige Nachbarn (Anteil unter den Nachbarn: {pct(share)}; GCN: "
    f"{pct(A.neighbor_mass(a.gcn_weights(), g.A, g.U))}). Wer erwartet, dass Aufmerksamkeit unzuverlässige Nachbarn ausblendet, sieht hier oft das Gegenteil - siehe Experiment 4."
)

st.markdown("---")

# --- Experiment 1 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viele Etiketten braucht die Aufmerksamkeit?")
st.caption(f"Standardgebiet mit unzuverlässigen Kunden (Anteil {pct(settings.unreliable)}), Etiketten je Typ von {C.LABEL_LEVELS[0]} bis {C.LABEL_LEVELS[-1]}; Mittel über {len(C.EXP_SEEDS)} feste Seeds, Fehlerbalken und ± sind Standardfehler "
           "der Differenz GAT minus GCN (gepaart je Seed). Dauer etwa eine Minute.")
if st.button("Etiketten durchrechnen", key="labels_start"):
    st.session_state["labels_on"] = True
if st.session_state.get("labels_on"):
    base_l = Settings(200, 3, 20, 5, settings.noise, settings.unreliable, settings.extra, 0.0, 0)
    rows_l = _labels(C.LABEL_LEVELS, C.EXP_SEEDS, base_l)
    st.plotly_chart(build_labels(rows_l), width="stretch", key="labels_chart")
    parts = "; ".join(f"{r['labels']} Etiketten: {pts(r['diff'], r['diff_se'])} Punkte (GAT gewinnt in {r['gat_wins']} von {r['n_seeds']})" for r in rows_l)
    clear = [r["labels"] for r in rows_l if r["diff"] > 2 * r["diff_se"]]
    st.warning(
        f"**Befund:** GAT minus GCN - {parts}. "
        + (f"Der Vorsprung liegt nur bei {', '.join(str(x) for x in clear)} Etiketten deutlich über dem zweifachen Standardfehler; " if clear else "Bei keiner Etikettenzahl liegt der Vorsprung über dem zweifachen Standardfehler; ")
        + "er ist nicht gleichmäßig größer mit mehr Etiketten. Aufmerksamkeit muss aus den bekannten Etiketten gelernt werden - bei wenigen ist das Bild unscharf."
    )

st.markdown("---")

# --- Experiment 2 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Wie viele unzuverlässige Kunden braucht der Vorsprung?")
st.caption(f"20 Etiketten je Typ, Anteil unzuverlässiger Kunden {', '.join(pct(x) for x in C.UNREL_LEVELS)}; {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Anteil durchrechnen", key="unrel_start"):
    st.session_state["unrel_on"] = True
if st.session_state.get("unrel_on"):
    base_u = Settings(200, 3, 20, 5, settings.noise, 0.5, settings.extra, 0.0, 0)
    rows_u = _unreliable(C.UNREL_LEVELS, C.EXP_SEEDS, base_u)
    st.plotly_chart(build_unreliable(rows_u), width="stretch", key="unrel_chart")
    parts = "; ".join(f"{pct(r['unreliable'])}: {pts(r['diff'], r['diff_se'])}" for r in rows_u)
    r0 = rows_u[0]
    st.warning(
        f"**Befund:** GAT minus GCN in Punkten - {parts}. Ohne unzuverlässige Kunden ({pct(r0['unreliable'])}) bringt die Aufmerksamkeit {'nichts' if r0['diff'] < 2 * r0['diff_se'] else 'etwas'}: GAT {pct(r0['gat'], 1)} gegen GCN {pct(r0['gcn'], 1)}. "
        f"Die Werte schwanken von Stufe zu Stufe stärker, als eine glatte Kurve erwarten ließe (Standardfehler bis {de(100 * max(r['diff_se'] for r in rows_u), 1)} Punkte bei nur {len(C.EXP_SEEDS)} Gebieten) - der Verlauf ist nur grob belastbar."
    )

st.markdown("---")

# --- Experiment 3 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Und wenn die Nachbarn nicht stimmen?")
st.caption(f"Keine unzuverlässigen Kunden, 20 Etiketten je Typ, ein Anteil der Kanten wird durch zufällige ersetzt ({', '.join(pct(x) for x in C.WRONG_LEVELS)}); {len(C.EXP_SEEDS)} feste Seeds. Dauer etwa eine Minute.")
if st.button("Falsche Kanten durchrechnen", key="wrong_start"):
    st.session_state["wrong_on"] = True
if st.session_state.get("wrong_on"):
    base_w = Settings(200, 3, 20, 5, settings.noise, 0.0, 0.0, 0.0, 0)
    rows_w = _wrong(C.WRONG_LEVELS, C.EXP_SEEDS, base_w)
    st.plotly_chart(build_wrong(rows_w), width="stretch", key="wrong_chart")
    parts = "; ".join(f"Homophilie {pct(r['homophily'])}: {pts(r['diff'], r['diff_se'])}" for r in rows_w)
    rl = rows_w[-1]
    st.warning(
        f"**Befund:** GAT minus GCN - {parts}. Die Aufmerksamkeit schützt nicht vor falschen Kanten: bei {pct(rl['homophily'])} Homophilie liegt GAT mit {pct(rl['gat'], 1)} um {de(abs(100 * rl['diff']), 1)} Punkte unter dem GCN ({pct(rl['gcn'], 1)}). "
        "Grund: die Bewertung ist eine Summe (statische Aufmerksamkeit) - sie kann einen Nachbarn nicht danach beurteilen, ob er zum fragenden Kunden passt. Das behebt GATv2 im nächsten Stück."
    )

st.markdown("---")

# --- Experiment 4 -----------------------------------------------------------------------------------------------------------------------------------

st.subheader("🔬 Ablation: woher kommt der Vorsprung - und was tut die Aufmerksamkeit?")
st.caption(f"Einstellungen der Seitenleiste (Seed ersetzt durch {len(C.EXP_SEEDS)} feste Seeds): dieselbe Architektur mit und ohne gelernte Aufmerksamkeit, GCN, MLP, GCN mit bekannter Zuverlässigkeit; dazu das Gewicht auf unzuverlässige Nachbarn "
           "und ob die Nachbar-Rangfolge vom fragenden Kunden abhängt. Dauer etwa eine halbe Minute.")
if st.button("Ablation durchrechnen", key="ablation_start"):
    st.session_state["ablation_on"] = True
if st.session_state.get("ablation_on"):
    res_a = _ablation(C.EXP_SEEDS, Settings(settings.n, settings.classes, settings.labels, settings.neighbors, settings.noise, settings.unreliable, settings.extra, settings.wrong, 0))
    st.plotly_chart(build_ablation(res_a), width="stretch", key="ablation_chart")
    st.warning(
        f"**Befund:** GAT {pct(res_a['gat'], 1)}, GCN {pct(res_a['gcn'], 1)}, dieselbe Architektur ohne gelernte Aufmerksamkeit {pct(res_a['uniform'], 1)}; die Aufmerksamkeit bringt {pts(res_a['gat_minus_uniform'], res_a['gat_minus_uniform_se'])} Punkte, "
        f"die zusätzlichen Köpfe allein {pts(res_a['uniform_minus_gcn'], res_a['uniform_minus_gcn_se'])}. Das von Hand gesetzte Herunterwichten unzuverlässiger Nachbarn erreicht {pct(res_a['oracle'], 1)}. "
        f"Die gelernte Aufmerksamkeit legt im Mittel **{pct(res_a['mass_trained'])}** ihres Nachbargewichts auf unzuverlässige Nachbarn (Anteil unter den Nachbarn: {pct(res_a['share_unreliable'])}; bei der Anfangsbelegung {pct(res_a['mass_init'])}) - "
        f"sie blendet sie also nicht aus; die Köpfe unterscheiden sich um im Mittel {de(100 * res_a['head_spread'], 0)} Punkte. Warum sie trotzdem hilft, wurde nicht bis zur Ursache verfolgt: Aufmerksamkeit ist hier keine einfache Erklärung. "
        f"Die Rangfolge der Nachbarn stimmt in {pct(res_a['ranking'])} der Nachbarpaare mit der Rangfolge der Senderbewertung überein - sie hängt nie vom fragenden Kunden ab."
    )

st.markdown("---")

# --- Grenzen -------------------------------------------------------------------------------------------------------------------------------------

st.subheader("🚧 Wo die Annahmen enden")
st.markdown(
    """
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Etiketten, um die Aufmerksamkeit zu lernen** | Mit wenigen Etiketten ist der Vorsprung unscharf (Experiment 1); GAT hat mehr Parameter als GCN. | – |
| **Nachbarn unterscheiden sich in ihrem Wert unabhängig vom Fragenden** | Statische Aufmerksamkeit kann nicht beurteilen, ob ein Nachbar zum fragenden Kunden passt; bei falschen Kanten fällt GAT hinter das GCN. | GATv2 (nächstes Stück) |
| **Aufmerksamkeit zeigt, was das Netz wichtig findet** | Hier legt sie im Mittel mehr Gewicht auf die unzuverlässigen Nachbarn und hilft trotzdem; Gewichte sind keine gesicherte Erklärung. | – |
| **Der ganze Graph ist beim Training bekannt** | GAT ist wie GCN transduktiv trainiert; neue Kunden oder sehr große Graphen brauchen Stichproben der Nachbarschaft. Hier nicht gemessen. | GraphSAGE |
| **Erzeugte Daten, acht Gebiete je Messpunkt** | Vier Merkmale, ein Kennzeichen, Voronoi-Zonen: die Zahlen gelten für dieses Vehikel; die Standardfehler sind groß. | – |
"""
)
st.caption("Die Linie: GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer (GraphSAGE, GATv2, GIN und Graph Transformer noch nicht gebaut).")

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    st.markdown(
        r"""
**Schicht mit $K$ Köpfen.** Eingabe $H \in \mathbb R^{n \times d}$, je Kopf $k$: $W_k \in \mathbb R^{d \times f}$, $a^{\text{E}}_k, a^{\text{S}}_k \in \mathbb R^f$. Mit $Wh^{(k)}_i = h_i W_k$:
$e^{(k)}_{ij} = \mathrm{LeakyReLU}_{0{,}2}\big(a^{\text{E}}_k \cdot Wh^{(k)}_i + a^{\text{S}}_k \cdot Wh^{(k)}_j\big)$ für $j \in \mathcal N(i) \cup \{i\}$, $\alpha^{(k)}_{ij} = \exp(e^{(k)}_{ij}) / \sum_{l} \exp(e^{(k)}_{il})$,
$h'^{(k)}_i = \sum_j \alpha^{(k)}_{ij} Wh^{(k)}_j$; Ausgabe: alle $h'^{(k)}$ nebeneinander (Schicht 1: $K = 4$, $f = 8$, danach ReLU; Schicht 2: $K = 1$, $f$ = Zahl der Klassen).

**Statische Aufmerksamkeit.** $e_{ij}$ ist monoton in $t_j = a^{\text{S}} \cdot Wh_j$ (LeakyReLU ist monoton, der Empfängeranteil $s_i$ ist für alle $j$ derselbe): die Rangfolge der Nachbarn von $i$ nach $\alpha_{ij}$ ist für jedes $i$ die Rangfolge nach $t_j$.

**Verlust.** Kreuzentropie auf den bekannten Kunden plus Gewichtszerfall $\tfrac{\lambda}{2}(\lVert W^{(1)}\rVert^2 + \lVert W^{(2)}\rVert^2)$, $\lambda = 5 \cdot 10^{-4}$; Adam (0,01), 200 Epochen.

**Rückwärtsrechnung** (je Kopf, mit $\delta$ = Gradient an der Ausgabe): $\partial\alpha_{ij} = \delta_i \cdot Wh_j$; Softmax: $\partial e_{ij} = \alpha_{ij}\big(\partial\alpha_{ij} - \sum_l \alpha_{il}\,\partial\alpha_{il}\big)$; LeakyReLU: $\partial z_{ij} = \partial e_{ij}\,\mathbb 1[z_{ij}>0 \text{ oder } 0{,}2]$;
$\partial a^{\text{E}} = \sum_i (\sum_j \partial z_{ij}) Wh_i$, $\partial a^{\text{S}} = \sum_j (\sum_i \partial z_{ij}) Wh_j$; $\partial Wh_j$ = $\sum_i \alpha_{ij}\delta_i$ + die Beiträge der Bewertungen.

**GCN-Vergleich.** $\hat A = D^{-1/2}(A+I)D^{-1/2}$; MLP: $\hat A = I$.

Implementiert in `gat_algorithm.py` (Schichten, Gradienten, Adam, Maße), `gat_scenario.py` (Vehikel), `gat_evaluation.py` (Analyse, vier Experimente).
        """
    )

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning. Interesse an einer maßgeschneiderten Lösung für "
    "Ihr Unternehmen? [Kontakt aufnehmen](https://sebastianhanisch.net/kontakt.html)"
)

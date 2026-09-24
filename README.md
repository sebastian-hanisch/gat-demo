# 🎯 GAT – lernbare Aufmerksamkeit auf Nachbarn

**[→ Demo live ausprobieren](https://sebastianhanisch-gat-demo.streamlit.app/)**

Zweites Stück der **Graph-Neural-Network-Linie** der "Konzepte"-Reihe im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning – und Nachfolger von
[gcn-demo](https://sebastianhanisch-gcn-demo.streamlit.app/) (GCN → GraphSAGE, GAT → GATv2, GIN → Graph Transformer; GraphSAGE, GATv2, GIN und Graph Transformer sind noch nicht gebaut).

Vehikel **D "Liefergebiete"** wie im Vorgänger (Kunden in räumlichen Gebietstypen, vier verrauschte Merkmale, Graph der nächsten Nachbarn, wenige bekannte Etiketten), **erweitert um unzuverlässige Kunden**: ein Anteil der Kunden hat
lückenhafte Telematik – ihre Merkmale sind zusätzlich stark verrauscht, und ein fünftes Merkmal ("Meldeausfall") zeigt es an. Ein GCN mittelt alle Nachbarn gleich, auch die unzuverlässigen. Ein **Graph Attention Network**
(Veličković et al. 2018) lernt für jeden Nachbarn ein Gewicht aus den Merkmalen. Alle Daten sind erzeugt, das Netz ist von Grund auf in numpy geschrieben (Vorwärts- und Rückwärtsrechnung von Hand).

**Bezug zu OR:** Datenqualität ist ein Alltagsproblem der Planung – eine unzuverlässige Telematikmeldung verfälscht, was in Prognose und Tourenplanung als Eingabe ankommt.

## Warum dieses Problem – und was sich gegenüber dem Plan geändert hat

Der Plan der Linie sah für GAT die Kante "behebt die Schwäche des GCN, alle Nachbarn gleich zu behandeln, also auch falsche Nachbarn" vor. Die **Vorab-Messung** hat das **widerlegt**: bei zufällig ersetzten Kanten liegt GAT
**hinter** dem GCN, und die gelernte Aufmerksamkeit legt bei unzuverlässigen Nachbarn im Mittel **mehr** Gewicht auf sie, nicht weniger. Was sich bestätigt hat: dieselbe Architektur mit gelernter Aufmerksamkeit ist deutlich besser
als ohne, aber nur mit genug Etiketten und nicht gleichmäßig. Das Stück misst deshalb, **wann und wie viel** Aufmerksamkeit bringt – und was sie nicht tut. Der Grund für den Misserfolg bei falschen Kanten ist bekannt und
hier messbar: GATs Bewertung ist eine **Summe** aus Empfänger- und Senderanteil ("statische Aufmerksamkeit"), die Rangfolge der Nachbarn hängt nur vom Nachbarn ab, nie vom fragenden Kunden. Das behebt GATv2 (nächstes Stück).

## Modell

- **Vehikel** (`gat_scenario.py`): wie gcn-demo; dazu Anteil unzuverlässiger Kunden (eigener Zufallsstrom, Lage und Grundmerkmale bleiben gleich), Zusatzrauschen und Kennzeichen (1 gegen 0, kleines Rauschen).
- **GAT** (`gat_algorithm.py`): zwei Schichten – 4 Köpfe × 8 Merkmale mit ReLU, dann ein Kopf mit den Klassen-Logits; Bewertung $e_{ij} = \mathrm{LeakyReLU}_{0{,}2}(a_E\cdot Wh_i + a_S\cdot Wh_j)$ über Nachbarn und Schleife,
  Softmax je Kunde; Kreuzentropie auf den bekannten Kunden, Gewichtszerfall $5\cdot10^{-4}$ (nur auf $W$), Adam (0,01), 200 Epochen, **kein Dropout, nichts am Test abgestimmt**. Abweichung vom Aufsatz: ReLU statt ELU, kein Dropout.
- **Vergleiche:** GCN (16 verdeckte Einheiten) und MLP ($\hat A = I$) wie im Vorgänger; **GAT ohne gelernte Aufmerksamkeit** (Bewertungsvektoren bei 0 gehalten: alle Nachbarn zählen gleich, aber dieselben Köpfe und Merkmale – trennt Aufmerksamkeit
  von zusätzlicher Kapazität); **GCN mit bekannter Zuverlässigkeit** (unzuverlässige Nachbarn von Hand auf ein Fünftel herunterwichtet; Referenz, nicht lernbar).

## Methodik

- **Handrechnungen:** GAT-Schicht mit zwei Knoten (Aufmerksamkeit $(1/(1+e),\ e/(1+e))$ in *beiden* Zeilen – die Bewertung ist statisch), LeakyReLU-Zweige, Orakel-Matrix auf einem Pfad, Gewichtsanteil auf unzuverlässige Nachbarn.
- **Gegenproben:** Vorwärtsrechnung gegen eine explizite Schleife (drei Köpfe), **Gradienten aller sechs Parametergruppen gegen zentrale Differenzen** (1, 2 und 4 Köpfe), GCN-Gradienten auch für eine **nicht symmetrische**
  Propagationsmatrix, Zeilen der Aufmerksamkeit summieren sich zu 1 und verschwinden außerhalb der Nachbarschaft, Wiederholbarkeit, Ablation hält die Bewertungsvektoren bei 0 (Aufmerksamkeit exakt gleichverteilt).
- **Maße:** Anteil des Gewichts auf unzuverlässige Nachbarn (ohne den Kunden selbst, je Kunde auf seine Nachbarn normiert); **Rangfolge-Übereinstimmung** – Anteil der Nachbarpaare, deren Aufmerksamkeits-Rangfolge mit der
  Rangfolge der Senderbewertung übereinstimmt (bei GAT immer 1; ein Test zeigt, dass eine dynamische Bewertung nach GATv2-Art das Maß unter 1 drückt).
- **Statistik:** je Messpunkt 8 feste Seeds; Differenzen GAT minus GCN **gepaart je Seed** mit Standardfehler; Gewinnzahl je Seed.
- **Literatur** (nicht nachgebaut): Veličković et al. 2018 ("Graph attention networks", ICLR); Brody/Alon/Yahav 2021 (GATv2, statische Aufmerksamkeit).

## Befunde (gemessen, keine Behauptungen)

| Frage | Befund | Test |
|---|---|---|
| Standardgebiet (200 Kunden, 104 unzuverlässig, 20 Etiketten je Typ, Seed 7) | Ein Einzelgebiet: GAT 88,6 %, GCN 82,9 %, GAT ohne Aufmerksamkeit 80,7 %, MLP 49,3 %, GCN mit bekannter Zuverlässigkeit 78,6 %. | `test_preset_standard` |
| **Was bringt die Aufmerksamkeit?** (Ablation, 8 Gebiete) | GAT 86,6 %, GCN 76,9 %, dieselbe Architektur ohne gelernte Aufmerksamkeit 77,2 %, MLP 50,2 %. Die Aufmerksamkeit bringt **+9,4 ± 1,8 Punkte** gegen die Architektur ohne sie, die zusätzlichen Köpfe allein **+0,4 ± 0,8**: der Vorsprung kommt von der Aufmerksamkeit, nicht von der Kapazität. | `test_ablation_experiment` |
| **Wie viele Etiketten?** (GAT minus GCN, gepaart, 8 Gebiete) | 5 Etiketten je Typ: +6,8 ± 3,9 Punkte (GAT gewinnt in 5 von 8); 10: +3,4 ± 2,6 (6); 20: **+9,7 ± 2,3** (8); 40: +3,7 ± 3,4 (6). In allen vier Stufen nicht schlechter, aber nur bei 20 deutlich über dem zweifachen Standardfehler; **nicht gleichmäßig größer** mit mehr Etiketten. | `test_labels_experiment` |
| Wie viele unzuverlässige Kunden? (20 Etiketten) | Ohne unzuverlässige Kunden: −1,3 ± 1,5 (GAT 93,0 % gegen GCN 94,3 %) – die Aufmerksamkeit bringt nichts. 20 %: +2,5 ± 3,2; 40 %: +1,6 ± 2,9; **60 %: +8,0 ± 2,1** (GAT gewinnt in 7 von 8); 80 %: +2,9 ± 2,9. Die Werte schwanken stärker als eine glatte Kurve; der Verlauf ist nur grob belastbar. | `test_unreliable_experiment`, `test_preset_all_reliable` |
| **Tut die Aufmerksamkeit, was man erwartet?** (Ablation) | **Nein:** im Mittel legt sie **74 %** ihres Nachbargewichts auf unzuverlässige Nachbarn (Anteil unter den Nachbarn: 51 %; bei der Anfangsbelegung vor dem Training 60 %) – sie blendet sie nicht aus, sie bevorzugt sie. Die vier Köpfe unterscheiden sich im Mittel um 16 Punkte. Ein von Hand gesetztes Herunterwichten (Orakel) erreicht nur 78,9 %, GAT 86,6 %. **Warum** GAT trotzdem hilft, wurde nicht bis zur Ursache verfolgt: Aufmerksamkeit ist hier keine einfache Erklärung. | `test_ablation_experiment` |
| Ist die Nachbar-Rangfolge vom Fragenden abhängig? | Nein: in **100 %** der Nachbarpaare stimmt die Rangfolge nach Aufmerksamkeit mit der Rangfolge nach Senderbewertung überein (in allen Experimenten). GAT kann "diesen Nachbarn mag jeder mehr", aber nicht "dieser Nachbar passt zu mir". | `test_neighbor_ranking_is_static_for_gat_and_a_dynamic_score_would_break_it`, alle Experimente |
| **Falsche Kanten?** (keine unzuverlässigen Kunden, 20 Etiketten) | GAT minus GCN: 0 %: −1,3 ± 1,5; 20 %: −1,0 ± 0,6 (Homophilie 81 %); 40 %: −1,6 ± 1,0 (70 %); **60 %: −8,6 ± 2,4** (59 %, GAT gewinnt in 1 von 8). Preset (Homophilie 57,9 %): GAT 76,4 %, GCN 82,9 %, **MLP ohne Nachbarn 85,7 %**. Die Aufmerksamkeit schützt nicht vor falschen Kanten. | `test_wrong_edges_experiment`, `test_preset_wrong_edges` |
| Wenige Etiketten im Einzelgebiet | 15 bekannte Kunden (Preset, Seed 8): GAT 55,7 %, GCN 50,3 %, MLP 49,7 % (Raten 35,7 %) – Vorsprung klein und unsicher. | `test_preset_five_labels` |
| Viele unzuverlässige im Einzelgebiet | 157 von 200 Kunden (Preset, Seed 8): GAT 87,9 %, GCN 80,7 %, ohne Aufmerksamkeit 79,3 %, MLP 38,6 % (Raten 36,4 %). Mit 40 Etiketten und 300 Kunden (Preset, Seed 1): GAT 87,8 %, GCN 77,8 %, ohne Aufmerksamkeit 82,8 %, Orakel 87,8 %. | `test_preset_many_unreliable`, `test_preset_forty_labels` |

Die Presets sind **Einzelgebiete** (Seeds so gewählt, dass die Klassen nicht stark ungleich groß sind; bei den meisten liegt der GAT-Vorsprung nahe dem Median über zwölf Gebiete, nicht aber bei "40 Etiketten, 300 Kunden" - dort gibt es keine Mehr-Seed-Messung); die Mehr-Seed-Zeilen sind die belastbaren Aussagen.

## Ehrliche Grenzen

| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Genug Etiketten, um die Aufmerksamkeit zu lernen** | Mit wenigen Etiketten ist der Vorsprung unscharf; GAT hat mehr Parameter als GCN. | – |
| **Nachbarn unterscheiden sich in ihrem Wert unabhängig vom Fragenden** | Statische Aufmerksamkeit kann nicht beurteilen, ob ein Nachbar zum fragenden Kunden passt; bei falschen Kanten fällt GAT hinter das GCN (gemessen). | GATv2 (nächstes Stück) |
| **Aufmerksamkeit zeigt, was das Netz wichtig findet** | Hier legt sie im Mittel mehr Gewicht auf unzuverlässige Nachbarn und hilft trotzdem; Aufmerksamkeitsgewichte sind keine gesicherte Erklärung. | – |
| **Der ganze Graph ist beim Training bekannt** | Wie GCN transduktiv trainiert; neue Kunden oder sehr große Graphen brauchen Stichproben der Nachbarschaft. Hier nicht gemessen. | GraphSAGE |
| **Erzeugte Daten, acht Gebiete je Messpunkt** | Ein Vehikel mit vier Merkmalen plus Kennzeichen; Standardfehler groß; Verlauf über den Anteil unzuverlässiger Kunden nur grob belastbar; ReLU statt ELU, kein Dropout. | – |

## Tests

Pytest-Suite (`pytest tests/ -v`, rund 8 Minuten wegen der Experimente): Kern per Handrechnung und Gegenprobe (GAT-Schicht, LeakyReLU, Vorwärtsrechnung gegen Schleife, Gradienten, statische Aufmerksamkeit samt Gegenbeispiel, Ablation, Orakel,
Anteil auf unzuverlässige Nachbarn), Vehikel, Auswertung und Experimente (Form, Trends), Preset- und Permalink-Klemmen, AppTest-Rauchtests (jedes Preset, Kartenansichten, Kundenwahl, Extremwerte, vier Experimente auf Abruf) und
`test_claims.py` (jede Zahl aus diesem README; Einzelgebiete mit Bändern von etwa 2 Punkten, Mehr-Seed-Zahlen mit Bändern um den Standardfehler).

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Einstiegspunkt |
| `gat_constants.py` | Regler-Grenzen, Netz-Konstanten, Experiment-Seeds |
| `gat_presets.py` | Permalink/Presets-Mechanik |
| `gat_scenario.py` | Vehikel D mit unzuverlässigen Kunden |
| `gat_algorithm.py` | GAT (Schichten, Gradienten, Adam), GCN/MLP, Orakel, Aufmerksamkeitsmaße |
| `gat_evaluation.py` | Analyse, vier Experimente |
| `gat_visualization.py` | Plotly-Abbildungen (Karte, Aufmerksamkeit je Kunde, Köpfe, Experimente) |

## Bewusst nicht umgesetzt

- Dropout, ELU, Early Stopping, mehr als vier Köpfe und die Feinabstimmung der Hyperparameter.
- Die Ursachenforschung, warum die Aufmerksamkeit unzuverlässige Nachbarn bevorzugt und trotzdem hilft.
- Die übrigen Stücke der Linie (GraphSAGE, GATv2, GIN, Graph Transformer); ein PDF-Export gehört nicht zur Linie.

## Lokal ausführen

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements-dev.txt
streamlit run app.py
```

Gebaut mit Streamlit, Plotly und numpy.

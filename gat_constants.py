"""Konstanten der GAT-Demo: Vehikel D "Liefergebiete" (wie gcn-demo) plus unzuverlässige Kunden (stark verrauschte Merkmale, aber ein sichtbares Qualitätskennzeichen), Regler, Experimente."""

EPS = 1e-9
SEED_MAX = 999999

AREA = 100.0
CLASS_NAMES = ("Innenstadt", "Vorstadt", "Ländlich", "Gewerbegebiet")
FEATURE_NAMES = ("Stopps je Stunde", "Parksuche (min)", "Ladegewicht (kg)", "Zeitfenster-Enge", "Meldeausfall (Kennzeichen)")
N_FEATURES = len(FEATURE_NAMES)

N_MIN, N_MAX, N_STEP, DEFAULT_N = 100, 400, 20, 200
CLASSES_MIN, CLASSES_MAX, DEFAULT_CLASSES = 2, 4, 3
LABELS_MIN, LABELS_MAX, DEFAULT_LABELS = 2, 40, 20
NEIGHBORS_MIN, NEIGHBORS_MAX, DEFAULT_NEIGHBORS = 2, 10, 5
NOISE_MIN, NOISE_MAX, NOISE_STEP, DEFAULT_NOISE = 0.5, 3.0, 0.25, 1.0
UNREL_MIN, UNREL_MAX, UNREL_STEP, DEFAULT_UNREL = 0.0, 0.8, 0.05, 0.5
EXTRA_MIN, EXTRA_MAX, EXTRA_STEP, DEFAULT_EXTRA = 0.0, 8.0, 0.5, 6.0
WRONG_MIN, WRONG_MAX, WRONG_STEP, DEFAULT_WRONG = 0.0, 1.0, 0.05, 0.0

HEADS = 4
HEAD_DIM = 8
HIDDEN_GCN = 16
EPOCHS = 200
LEARNING_RATE = 0.01
WEIGHT_DECAY = 5e-4
LEAKY_SLOPE = 0.2

# --- Experimente (feste Seeds) --------------------------------------------------------------------------------------------------------------

EXP_SEEDS = tuple(range(8))
LABEL_LEVELS = (5, 10, 20, 40)
UNREL_LEVELS = (0.0, 0.2, 0.4, 0.6, 0.8)
WRONG_LEVELS = (0.0, 0.2, 0.4, 0.6)

"""Configuration commune du projet IFT-7030 (entropies EEG et santé mentale).

Module importé par tous les scripts numérotés (01_... à 10_...). Il ne fait aucun
calcul : il regroupe les constantes (bases, canaux, fréquences, bandes, mesures)
et l'organisation des dossiers de sortie, pour qu'un changement soit fait à un
seul endroit.

Organisation des dossiers (par défaut, à côté des scripts) :
    data_eeg/mumtaz   data_eeg/borisov   data_eeg/sakib      données brutes
    sorties/01_donnees/                  signaux bruts sauvegardés (npz + csv)
    sorties/02_controle_qualite/<date>/  contrôles des bases
    sorties/03_extraction/<base>_<date>/ entropies par fenêtre (features_windows.csv)
    sorties/04_fusion/<date>/            matrices par sujet, 3 bases fusionnées
    sorties/05_binaire/<date>/           analyses binaires
    sorties/06_multiclasse/<date>/       analyses 4 classes + contrôles de site
    sorties/07_reseaux_neurones/<date>/  réseaux profonds
    sorties/08_statistiques/<date>/      tests de groupes, tailles d'effet
    sorties/09_interpretabilite/<date>/  importance des variables
    sorties/10_synthese/<date>/          intervalles de confiance, tableaux du rapport
"""
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_ROOT = SCRIPT_DIR / "data_eeg"
SORTIES = SCRIPT_DIR / "sorties"

DOSSIERS = {
    "donnees": "01_donnees",
    "qc": "02_controle_qualite",
    "extraction": "03_extraction",
    "fusion": "04_fusion",
    "binaire": "05_binaire",
    "multiclasse": "06_multiclasse",
    "profond": "07_reseaux_neurones",
    "statistiques": "08_statistiques",
    "interpretabilite": "09_interpretabilite",
    "synthese": "10_synthese",
}

# --------------------------------------------------------------------------
# Bases de données
# --------------------------------------------------------------------------
BASES = ["depression", "schizophrenie", "anxiete"]
SOURCES = {
    "depression": "Mumtaz et al. 2016 (EDF, yeux fermés)",
    "schizophrenie": "Borisov et al. 2005 (.eea, adolescents)",
    "anxiete": "Sakib et al. 2023 (.mat, Emotiv EPOC+)",
}
SOUS_DOSSIER_BRUT = {"depression": "mumtaz", "schizophrenie": "borisov", "anxiete": "sakib"}
FS_ORIGINE = {"depression": 256, "schizophrenie": 128, "anxiete": 128}
FS_CIBLE = 128            # toutes les bases ramenées à 128 Hz avant la DWT
BANDPASS = (0.5, 45.0)    # Hz

MUMTAZ_CONDITION = "EC"   # EC (yeux fermés), EO (yeux ouverts) ou TASK

# Ancienne nomenclature 10-20 → nomenclature 10-10.
ALIASES = {"T3": "T7", "T4": "T8", "T5": "P7", "T6": "P8"}

# Électrodes de chaque base, dans l'ordre des données (noms 10-10).
CANAUX = {
    # Mumtaz : noms lus dans les EDF (« EEG Fp1-LE ») ; 19 électrodes 10-20 gardées.
    "depression": ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
                   "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"],
    # Borisov et al. 2005 : ordre publié des fichiers .eea (SUPPOSÉ, non confirmé dans les fichiers).
    "schizophrenie": ["F7", "F3", "F4", "F8", "T7", "C3", "Cz", "C4", "T8", "P7",
                      "P3", "Pz", "P4", "P8", "O1", "O2"],
    # Emotiv EPOC+ : ordre standard du casque (SUPPOSÉ, non documenté par Sakib et al.).
    "anxiete": ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8",
                "FC6", "F4", "F8", "AF4"],
}
ORDRE_SUPPOSE = {"depression": False, "schizophrenie": True, "anxiete": True}
# Noms Mumtaz tels qu'écrits dans les EDF (10-20).
MUMTAZ_CANAUX_EDF = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T3", "C3", "Cz",
                     "C4", "T4", "T5", "P3", "Pz", "P4", "T6", "O1", "O2"]

# Ordre d'affichage (antéro-postérieur puis gauche-droite).
ORDRE_PREFERE = ["Fp1", "Fp2", "AF3", "AF4", "F7", "F3", "Fz", "F4", "F8", "FC5", "FC6",
                 "T7", "C3", "Cz", "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]

# Enregistrements d'une même personne probable : même groupe de validation.
GROUPES_PRUDENTS = {
    "schizophrenie": {
        "387-02w1": "schizophrenie_387_groupe_prudent_non_confirme",
        "387-03w1": "schizophrenie_387_groupe_prudent_non_confirme",
    },
}

# --------------------------------------------------------------------------
# Caractéristiques
# --------------------------------------------------------------------------
MEASURES = ["SampEn", "DispEn", "FuzzyEn", "Shannon_spectrale", "LZC"]
MESURES_ETENDUES = ["ApEn", "PermEn", "Shannon_amplitude"]
PUISSANCES = ["LogPower", "RelPower"]
MULTIECHELLE = [f"{name}_{scale:02d}" for name in ("RCMSE", "RCMPE") for scale in range(1, 11)]
# DWT db4 niveau 4 à 128 Hz : pywt.wavedec renvoie [A4, D4, D3, D2, D1].
BANDS = ["A4", "D4", "D3", "D2", "D1"]
BANDES_HZ = {"A4": "0-4 Hz (delta)", "D4": "4-8 Hz (thêta)", "D3": "8-16 Hz (alpha)",
             "D2": "16-32 Hz (bêta)", "D1": "32-64 Hz (gamma, tronquée à 45 Hz)"}
AMPLITUDE = "sd_before_standardization"

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
CLASSES = ["temoin", "depression", "schizophrenie", "anxiete"]   # indices 0..3
CLASS_OF_BASE = {"depression": 1, "schizophrenie": 2, "anxiete": 3}
CLASSES_BINAIRES = ["temoin", "trouble"]
NOM_TROUBLE = {"depression": "dépression", "schizophrenie": "schizophrénie", "anxiete": "anxiété"}

GRAINE = 7030

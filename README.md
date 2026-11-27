# Pipeline modulaire : entropies EEG et santé mentale (IFT-7030)

Scripts numérotés, à lancer dans l'ordre depuis ce dossier. Chaque étape lit par défaut
la sortie **la plus récente** de l'étape précédente dans `sorties/` ; toutes acceptent `--help`.

| Étape | Script | Rôle | Sortie principale |
|---|---|---|---|
| 00 | `00_pipeline.py` | lance tout (ou `--etapes 4 5 6`), `--rapide` pour un essai | `sorties/journal_pipeline_*/` |
| 01 | `01_chargement_donnees.py` | lecture EDF / .eea / .mat, µV, doublons (SHA), groupes de validation | `sorties/01_donnees/` |
| 02 | `02_controle_qualite.py` | durées, NaN, canaux plats/saturés, bruit secteur, DSP, **confusions d'acquisition** | `sorties/02_controle_qualite/` |
| 03 | `03_extraction_entropies.py` | filtre 0,5-45 Hz, 128 Hz, fenêtres 10 s, DWT db4 (A4…D1), SampEn, DispEn, FuzzyEn, Shannon spectrale, LZC (multiprocessus) | `sorties/03_extraction/<base>_*/features_windows.csv` |
| 04 | `04_fusion_harmonisation.py` | 10 canaux communs, 4 classes, diagnostic de l'effet de site (η², ACP) | `sorties/04_fusion/` |
| 05 | `05_classification_binaire.py` | trouble / témoins par base : RL, MLP, SVM, forêt, LDA, XGBoost, références amplitude et canaux d'origine, ablation, CV imbriquée, permutations | `sorties/05_binaire/` |
| 06 | `06_classification_multiclasse.py` | 4 classes × 4 harmonisations dans les plis, contrôles de site (a)/(b), permutations globale et intra-base, leave-one-dataset-out | `sorties/06_multiclasse/` |
| 07 | `07_reseaux_neurones.py` | CNN 1D, CNN+attention, EEGNet adapté, FT-Transformer, TabM (binaire et 4 classes) | `sorties/07_reseaux_neurones/` |
| 08 | `08_statistiques_groupes.py` | Mann-Whitney + FDR, δ de Cliff, g de Hedges, similarité des signatures entre troubles | `sorties/08_statistiques/` |
| 09 | `09_interpretabilite.py` | importance par permutation de groupes (mesure, bande, électrode), coefficients RL, SHAP optionnel | `sorties/09_interpretabilite/` |
| 10 | `10_synthese_resultats.py` | IC 95 % bootstrap par sujet, Brier, calibration, comparaisons appariées, tableaux Markdown / LaTeX | `sorties/10_synthese/` |

Modules partagés (non exécutables ; un module Python ne peut pas commencer par un chiffre) :

- `eeg_config.py` : bases, électrodes (ordres supposés), fréquences, bandes, mesures, dossiers ;
- `eeg_utils.py` : messages, dossiers horodatés, validation croisée groupée, métriques, modèles scikit-learn, figures ;
- `eeg_donnees.py` : lecture des sorties 01, 03 et 04, agrégation par sujet, fusion ;
- `eeg_harmonisation.py` : z-score, filtrage, CORAL, filtrage+CORAL, η².

## Démarrage

```bash
pip install numpy scipy pandas scikit-learn matplotlib mne pywavelets
pip install torch            # étape 07
pip install EntropyHub xgboost shap jinja2   # options

# Données attendues : data_eeg/mumtaz/*.edf, data_eeg/borisov/{norm,sch}.zip, data_eeg/sakib/*.mat
python 00_pipeline.py --rapide          # essai complet
python 00_pipeline.py                   # analyse complète (long)
```

Essai sans les vraies données : `python outils/generer_donnees_test.py` puis
`python 00_pipeline.py --donnees data_eeg_test --rapide` (environ 5 min ; résultats sans valeur).

## Réutiliser les extractions existantes

Les étapes 04 à 09 acceptent les anciens dossiers `extraction_corrigee.py` (même format,
colonnes `col01…` converties avec l'ordre supposé) :

```bash
python 04_fusion_harmonisation.py --depression 04_extraction_corrigee/depression_20261122_185618_080052 \
    --schizophrenie 04_extraction_corrigee/schizophrenie_20261123_193938_456160 \
    --anxiete 04_extraction_corrigee/anxiete_20261123_215357_170531
python 05_classification_binaire.py --depression ... --schizophrenie ... --anxiete ...
python 06_classification_multiclasse.py        # lit la fusion la plus récente
```

## Points d'attention

- Ordre des électrodes **supposé** pour Borisov et Sakib (`eeg_config.CANAUX`) ; corriger avec
  `--carte-canaux carte.json` (`{"schizophrenie": [...16 noms], "anxiete": [...14 noms]}`).
- Mumtaz : les EDF sont lus en volts par mne et convertis en µV ; le nombre de canaux d'origine
  (20/22) est gardé et testé comme confusion (02, et `reference_canaux_origine` en 05).
- 387-02w1 et 387-03w1 (schizophrénie) partagent un groupe de validation (`eeg_config.GROUPES_PRUDENTS`).
- Toute harmonisation utilisée pour estimer une performance est ajustée dans chaque pli sur les
  témoins d'entraînement ; les fichiers `harmonise_*.csv` de 04 sont descriptifs.
- Mac / conda : 07 importe PyTorch avant numpy, limite OpenMP à 1 fil et teste chaque réseau
  dans un sous-processus (voir l'en-tête du script).

## Notebook `entropy_classification.ipynb`

Le notebook utilise une analyse distincte de celle des scripts numérotés. La validation imbriquée
garde les fenêtres d'un participant dans le même pli et regroupe par prudence les enregistrements
Borisov `387-02w1` et `387-03w1` ; leur appartenance à une même personne n'est pas confirmée.
Les contrôles recherchent les fenêtres de caractéristiques identiques et, si les signaux sont
disponibles, les copies exactes d'enregistrements ou de segments bruts alignés de 10 secondes.
Ils ne détectent pas toutes les copies transformées, les identités erronées ni les biais de site.

Le notebook compare les caractéristiques d'entropie/complexité seules à ces mêmes caractéristiques
avec les puissances, sur les mêmes partitions groupées. Il effectue trois répétitions par défaut :
ces deux recherches complètes peuvent prendre plusieurs heures. Les nouveaux fichiers
`outputs/window_repeated_*` sont distincts des anciens résultats, qui doivent être recalculés.
Le contrôle de permutation est facultatif et coûteux ; l'analyse SHAP de stabilité réentraîne deux
modèles supplémentaires par base. Aucun de ces contrôles ne remplace une validation externe sur
des participants indépendants.

La dernière section du notebook fusionne les fenêtres sur les dix électrodes supposées communes et
évalue séparément deux classes (témoin / trouble), quatre classes (témoin / dépression /
schizophrénie / anxiété) et la prédiction de la base chez les témoins. Les plis imbriqués sont
stratifiés par base et statut, au niveau des groupes de validation, puis les probabilités des
fenêtres sont moyennées par participant. Les poids d'apprentissage compensent le nombre de fenêtres
par participant et sont équilibrés par classe dans chaque ajustement. Les résultats `pooled_*.csv`
sont exploratoires : chaque trouble provenant d'une seule base, une bonne classification à quatre
classes peut refléter les différences de site plutôt que les diagnostics.

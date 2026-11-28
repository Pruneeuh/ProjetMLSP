# Pipeline modulaire : entropies EEG et santé mentale (IFT-7030)

Scripts numérotés, à lancer dans l'ordre depuis ce dossier. Exécutée seule, chaque étape lit
par défaut la sortie **la plus récente** de l'étape précédente dans `sorties/` ; toutes acceptent `--help`.

| Étape | Script | Rôle | Sortie principale |
|---|---|---|---|
| 00 | `00_pipeline.py` | lance 01–06 et 08–10 (ou `--etapes 4 5 6`), `--rapide` pour un essai | `sorties/journal_pipeline_*/` |
| 01 | `01_chargement_donnees.py` | lecture EDF / .eea / .mat, µV, doublons (SHA), groupes de validation | `sorties/01_donnees/` |
| 02 | `02_controle_qualite.py` | contrôle des signaux, filtrage 0,5–45 Hz, rééchantillonnage à 128 Hz et confusions d'acquisition | `sorties/02_controle_qualite/<date>/` |
| 03 | `03_extraction_entropies.py` | fenêtres 10 s, DWT, entropies et énergies, RCMSE/RCMPE large bande | `sorties/03_extraction/<base>_*/features_windows.csv` |
| 04 | `04_fusion_harmonisation.py` | 10 canaux communs, 4 classes, diagnostic de l'effet de site (η², ACP) | `sorties/04_fusion/` |
| 05 | `05_classification_binaire.py` | trouble / témoins par base : RL, MLP, SVM, kNN, forêt, LDA, XGBoost, LightGBM, références et recherche imbriquée | `sorties/05_binaire/` |
| 06 | `06_classification_multiclasse.py` | 4 classes × 4 harmonisations dans les plis, contrôles de site, permutations facultatives, leave-one-dataset-out | `sorties/06_multiclasse/` |
| 07 | `07_reseaux_neurones.py` | CNN 1D, CNN+attention, EEGNet adapté, FT-Transformer, TabM (binaire et 4 classes) | `sorties/07_reseaux_neurones/` |
| 08 | `08_statistiques_groupes.py` | Mann-Whitney + FDR, δ de Cliff, g de Hedges, similarité des signatures entre troubles | `sorties/08_statistiques/` |
| 09 | `09_interpretabilite.py` | importance par permutation de groupes, coefficients RL, SHAP sur les sujets hors pli | `sorties/09_interpretabilite/` |
| 10 | `10_synthese_resultats.py` | IC 95 %, métriques par classe, Brier, calibration et comparaisons appariées | `sorties/10_synthese/` |

L'étape 01 exclut `387-02w1` par précaution et conserve `387-03w1` lorsqu'il est
exploitable ; l'exclusion est consignée dans les métadonnées. L'étape 02 signale les
problèmes de qualité et confusions possibles mais ne confirme pas l'ordre des électrodes
supposé ni ne retire automatiquement les signaux. Un signal non fini, absent, incompatible
avec les canaux ou trop court pour le filtre arrête l'étape 02 ; l'étape 03 ne remplace plus
les valeurs non finies par zéro. Une alerte d'amplitude élevée n'est pas une preuve de
saturation physique. Les unités Borisov et Sakib et l'ordre de leurs canaux restent supposés.

Le lanceur réutilise automatiquement les sorties 01–04 qu'il a précédemment produites si les
empreintes SHA-256 des entrées et des sorties, le code, les paramètres et les versions Python
et des paquets correspondent. Il transmet les dossiers réutilisés aux étapes suivantes. Les
anciennes sorties sans preuve de provenance ne sont pas réutilisées automatiquement. Les données
source doivent rester disponibles pour vérifier l'étape 01. `--recalculer` force l'exécution
de 01–04 ; les étapes 05–10 ne sont pas mises en cache. Si 01–04 échoue, le lanceur s'arrête,
même avec `--continuer`, pour éviter d'utiliser des résultats périmés.

L'étape 02 écrit les signaux prétraités dans un nouveau dossier et termine par un manifeste
`controle_qualite.json`. L'étape 03 lit ce dossier sans filtrer une deuxième fois et refuse
un dossier incomplet ou incompatible. Les empreintes SHA-256 des signaux source et
prétraités sont contrôlées respectivement en 02 et en 03. Le lanceur transmet explicitement le dossier produit
par 02 à 03. Pour exécuter les étapes
séparément, lancer 02 avant 03 ; `03 --donnees` désigne désormais le dossier de 02.
Le coupe-bande secteur, si nécessaire, se règle en 02 avec `--notch 50` ou `--notch 60` ;
il reste désactivé par défaut. Les paramètres sont fixes et appliqués à chaque enregistrement,
sans apprentissage inter-sujets ; il ne s'agit pas d'une correction garantie des effets de site.

Modules partagés (non exécutables ; un module Python ne peut pas commencer par un chiffre) :

- `eeg_config.py` : bases, électrodes (ordres supposés), fréquences, bandes, mesures, dossiers ;
- `eeg_utils.py` : messages, dossiers horodatés, validation croisée groupée, métriques, modèles scikit-learn, figures ;
- `eeg_donnees.py` : lecture des sorties 01, 03 et 04, agrégation par sujet, fusion ;
- `eeg_harmonisation.py` : z-score, filtrage, CORAL, filtrage+CORAL, η² ;
- `eeg_tuning.py` : recherche groupée et sondes GPU partagées par 05, 06 et 09.

## Démarrage

### Valeria : environnement isolé

Avec les mêmes modules que ceux utilisés lors de la création de l'environnement :

```bash
module load StdEnv/2023 python/3.12.4 scipy-stack/2024a
python 00_package_installs.py
source ~/venvs/projetmlsp-valeria/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
```

Le script crée un nouvel environnement sous `~/venvs/projetmlsp-valeria`, installe les
paquets depuis la wheelhouse Valeria avec `--no-index`, puis vérifie les imports, les
dépendances et un petit entraînement CPU avec explication SHAP/XGBoost. Il refuse de modifier
un chemin existant ; utiliser `--env ~/venvs/un-autre-nom` pour un nouvel emplacement. Les
versions NumPy, SciPy, Matplotlib, SHAP, XGBoost et scikit-learn correspondent à celles
observées dans le notebook Valeria. La disponibilité des wheels et de leurs dépendances
peut varier selon les modules ; en cas d'échec, consulter `avail_wheels` ou le support Valeria.
Aucun téléchargement PyPI ni installation RAPIDS n'est tenté automatiquement.
Cette installation est à effectuer une seule fois : le lanceur `00_pipeline.py` exécute ensuite
les étapes 01 à 06 et 08 à 10 dans l'environnement activé, sans réinstaller les paquets.
L'étape 07 reste accessible explicitement via `--etapes 7 10` ou directement. La synthèse
du lanceur exclut les anciens résultats 07 lorsqu'elle n'a pas été exécutée avec succès.

Le lanceur essaie également RAPIDS (`cuml.accel`) dans les processus 05, 06 et 09, avant
l'import de scikit-learn. RAPIDS doit être installé séparément dans l'environnement activé ;
vérifier avec `avail_wheels cuml` les wheels compatibles avec les modules CUDA et Python
de Valeria. Sa disponibilité dans la wheelhouse n'a pas été vérifiée. Si cuML n'est pas
installé ou si aucun GPU CUDA n'est visible, un message explique le retour au CPU. Une
installation incompatible est une erreur et doit être corrigée.

Si la wheelhouse fournit une version `cuml-cu12` compatible avec l'environnement et le
CUDA de la session, l'installer dans l'environnement activé, puis vérifier les dépendances :

```bash
python -m pip install --no-index --only-binary=:all: cuml-cu12
python -m pip check
```

Cette commande n'est pas adaptée aux autres versions CUDA ni garantie disponible sur
Valeria ; consulter `avail_wheels cuml` avant de l'utiliser.

Quand RAPIDS est actif, ses journaux INFO indiquent quelles opérations tournent sur GPU
ou reviennent au CPU ; ne pas supposer que tous les modèles sont accélérés. LDA et MLP
restent sur CPU avec les implémentations actuelles ; le SVM avec `probability=True` peut
aussi y revenir. Les modèles RAPIDS peuvent différer numériquement de scikit-learn.
XGBoost et LightGBM testent leurs GPU indépendamment ; un échec ultérieur est signalé.
Le prétraitement et certaines opérations d'extraction restent sur CPU. Le mode rapide
ne sélectionne pas XGBoost ni LightGBM. Utiliser `--sans-rapids` sur le lanceur pour
conserver le comportement scikit-learn précédent, ou `--rapids` en lançant 05, 06 ou 09
directement. Vérifier les journaux et la compatibilité SHAP sur la machine cible.

L'étape 03 essaie également CuPy sur un GPU CUDA pour les calculs de paires de SampEn,
FuzzyEn et RCMSE. Elle utilise alors un seul processus ; les autres mesures, la DWT et les
puissances restent sur CPU. Sans CuPy ou GPU visible, elle annonce le retour au CPU ;
une erreur ultérieure n'est pas masquée. `--cpu-entropies` force le CPU dans le lanceur ;
en exécution directe, `03 --calcul cpu`, `auto` ou `cuda` sélectionne le dispositif
(`cuda` échoue si le GPU est indisponible). Le cache du lanceur tient compte du dispositif.

Vérifier `avail_wheels cupy` ; si la wheelhouse fournit une version compatible avec
l'environnement et CUDA 12, l'installation possible dans l'environnement activé est :

```bash
python -m pip install --no-index --only-binary=:all: cupy-cuda12x
python -m pip check
```

Le calcul GPU n'a pas encore été vérifié sur Valeria : comparer ses résultats à ceux du
CPU avant de les interpréter. Il peut aussi être plus lent pour de petits signaux.

Pour Jupyter, après activation, enregistrer un noyau avec un nom qui n'existe pas déjà :

```bash
python -m ipykernel install --user --name projetmlsp-valeria \
    --display-name "ProjetMLSP (Valeria)" --env PYTHONPATH "" \
    --env PYTHONHOME "" --env PYTHONNOUSERSITE 1
```

Charger les mêmes modules dans Jupyter, sélectionner ce noyau et redémarrer le notebook.
La cellule de configuration du notebook affiche désormais l'interpréteur au lieu de lancer
`pip --user`. Les anciens résultats affichés dans cette cellule sont historiques. Le test
du script est effectué sur CPU : la disponibilité GPU de XGBoost, LightGBM ou PyTorch
doit être vérifiée séparément dans une session GPU.

### Installation manuelle

```bash
pip install numpy scipy pandas scikit-learn matplotlib mne pywavelets
pip install torch            # étape 07
pip install EntropyHub xgboost lightgbm shap jinja2

# Données attendues : data_eeg/mumtaz/*.edf, data_eeg/borisov/{norm,sch}.zip, data_eeg/sakib/*.mat
python 00_pipeline.py --rapide          # essai complet
python 00_pipeline.py                   # analyse complète (long)
```

Essai sans les vraies données : `python outils/generer_donnees_test.py` puis
`python 00_pipeline.py --donnees data_eeg_test --rapide` (résultats sans valeur).

## Réutiliser les extractions existantes

Les anciennes extractions `extraction_corrigee.py` peuvent manquer les caractéristiques
étendues et multiéchelles. Dans ce cas, sélectionner explicitement le mode classique
(colonnes `col01…` converties avec l'ordre supposé) :

```bash
python 04_fusion_harmonisation.py --sans-caracteristiques-etendues \
    --depression 04_extraction_corrigee/depression_20261122_185618_080052 \
    --schizophrenie 04_extraction_corrigee/schizophrenie_20261123_193938_456160 \
    --anxiete 04_extraction_corrigee/anxiete_20261123_215357_170531
python 05_classification_binaire.py --caracteristiques classiques --depression ... --schizophrenie ... --anxiete ...
python 06_classification_multiclasse.py --caracteristiques classiques
```

## Caractéristiques et comparaisons

L'étape 03 calcule toujours ApEn, PermEn, Shannon d'amplitude, les énergies DWT et
RCMSE/RCMPE par canal sur le signal filtré, aux échelles 1 à 10. Les fenêtres sont
agrégées par participant ; `features_sujets.csv` contient leur moyenne. RCMSE utilise
m=2 et r=0,15 écart-type du signal de la fenêtre ; RCMPE utilise m=4 et le logarithme
naturel. Les mesures large bande figurent uniquement sur les lignes A4 du fichier long,
mais ne sont **pas** des caractéristiques A4 ; la fusion les étiquette `large_bande`.
Les valeurs indéfinies restent manquantes et l'imputation est ajustée dans les plis.

Par défaut, 04 fusionne les caractéristiques étendues, multiéchelles et les énergies ; 05,
06 et 09 utilisent ce jeu combiné même lorsqu'ils sont lancés séparément. Le lanceur
transmet aussi explicitement ces options,
active l'ablation en 05 et SHAP hors pli en 09 (régression logistique réglée par défaut).
Les tests de permutation restent désactivés ; l'importance par permutation de variables en
09 est une analyse différente. Ces calculs peuvent être très longs, même en mode rapide.
L'étape 07 conserve son tenseur des cinq mesures classiques.

Pour exécuter les étapes séparément :

```bash
pip install EntropyHub
python 02_controle_qualite.py
python 03_extraction_entropies.py
python 04_fusion_harmonisation.py
python 05_classification_binaire.py --caracteristiques etendues
python 05_classification_binaire.py --caracteristiques etendues_puissances
python 06_classification_multiclasse.py --caracteristiques etendues
python 06_classification_multiclasse.py --caracteristiques etendues_puissances
```

Utiliser les mêmes dossiers d'extraction/fusion, modèles, plis, répétitions et graines dans les
deux exécutions pour comparer les caractéristiques sur les mêmes partitions ; choisir les
dossiers explicitement avec `--depression`, `--schizophrenie`, `--anxiete` et `--fusion` si
nécessaire. `etendues` ajoute les trois mesures et les entropies multiéchelles aux cinq mesures classiques ;
`etendues_puissances` ajoute en plus LogPower et RelPower. Les anciens dossiers restent
utilisables en mode `classiques` ; les options étendues exigent de nouvelles extractions.
Dans certains résultats, l'étiquette historique `entropies` désigne le jeu sélectionné :
avec le réglage par défaut, il contient aussi les puissances. La référence d'amplitude est
un modèle distinct.

Les étapes 05 et 06 continuent à moyenner les caractéristiques par participant avant
l'entraînement : elles ne reproduisent pas l'entraînement par fenêtre du notebook. Le
rééchantillonnage à 128 Hz et les unités µV sont conservés. LogPower est log10 de l'énergie
DWT en µV², et non une puissance moyenne ; RelPower est la fraction de l'énergie des cinq
composantes. Les résultats peuvent donc différer du notebook. Les puissances peuvent également
capturer des différences de site ; examiner les contrôles de site et les références d'amplitude.

## Recherche de modèles et interprétation

05 et 06 utilisent par défaut une recherche aléatoire **imbriquée et groupée par sujet** pour
la régression logistique, le MLP, le SVM, kNN, la forêt, LDA, XGBoost et LightGBM. La sélection
de variables est ajustée dans chaque pli interne ; 06 y réajuste aussi l'harmonisation sur les
seuls témoins internes d'entraînement. Le score de recherche est l'exactitude équilibrée par
participant. La référence majoritaire n'est pas réglée. Les modèles et espaces de recherche
s'appuient sur `model_tuning.py`, avec adaptation à la classification multiclasse.

Le profil `comprehensive` par défaut utilise 4 plis internes et 40 candidats (25 pour les arbres)
par pli externe ; cela peut être très long, surtout avec toutes les harmonisations. `quick`
utilise 3 plis internes et 8 candidats (6 pour les arbres). Pour un essai ciblé :

```bash
python 05_classification_binaire.py --bases depression --modeles regression_logistique knn --profil-tuning quick
python 06_classification_multiclasse.py --modeles regression_logistique lightgbm --harmonisations zscore --profil-tuning quick
python 09_interpretabilite.py --taches multiclasse --modele lightgbm --harmonisation zscore --shap --profil-tuning quick
```

`--plis-internes` et `--tirages-tuning` permettent de réduire davantage le coût ; `--sans-tuning`
désactive la recherche et utilise les réglages fixes. Avec `--sans-tuning`, `--c`, `--mlp-couches`
et `--mlp-alpha` règlent les modèles concernés dans 05 et 06. Les fichiers de prédictions restent
compatibles avec 10. Les paramètres choisis sont enregistrés par pli dans 06 et 09, et par
répétition dans 05.

Les étapes sondent les GPU XGBoost et LightGBM puis utilisent le CPU si la sonde échoue ;
les autres modèles restent sur CPU. `--cpu` force tous les modèles sur CPU. Une sonde réussie
ne garantit pas qu'un grand entraînement tienne en mémoire. Les permutations de 06 sont
désactivées par défaut ; `--permutations N` les active et répète toute la recherche pour chaque
permutation. Elles peuvent donc être particulièrement coûteuses.

09 retune le modèle choisi dans chaque pli et explique des participants hors pli (`--shap`).
Pour comparer à 05/06, utiliser les mêmes données, caractéristiques, modèle, plis, graines,
profil et harmonisation. `shap_par_pli.csv` et `shap_valeurs.csv` présentent les valeurs absolues
par classe et variable ; les variables non sélectionnées reçoivent zéro dans le pli concerné.
09 prend en charge les harmonisations `zscore` et `aucune` ; CORAL et le filtrage ne conservent
pas directement les mêmes variables d'un pli à l'autre.
Les échelles SHAP varient selon le modèle ; elles ne sont pas directement comparables entre
familles. Les scores et explications restent exploratoires et ne prouvent pas une causalité.

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

Le notebook utilise une analyse distincte de celle des scripts numérotés. Il exclut par prudence
l'enregistrement Borisov `387-02w1` et conserve `387-03w1` ; leur appartenance à une même
personne n'est pas confirmée. Les scripts numérotés les conservent dans un même groupe
de validation. Les données sources ne sont pas modifiées. Les résultats sauvegardés avant cette
exclusion doivent être recalculés. La validation imbriquée garde les fenêtres d'un participant
dans le même pli.
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

Pour les fenêtres du notebook, `python extract_entropy.py` calcule toutes les entropies, la
complexité et les énergies dans un seul fichier `outputs/<base>_entropy_features.csv` par base.
Le notebook accepte aussi les anciennes paires de fichiers entropy/extra après vérification des
fenêtres et étiquettes. Si seul l'ancien fichier entropy existe, relancer explicitement
`python extract_entropy.py <base>` pour le remplacer ; le calcul peut être long.

La dernière section du notebook fusionne les fenêtres sur les dix électrodes supposées communes et
évalue séparément deux classes (témoin / trouble), quatre classes (témoin / dépression /
schizophrénie / anxiété) et la prédiction de la base chez les témoins. Les plis imbriqués sont
stratifiés par base et statut, au niveau des groupes de validation, puis les probabilités des
fenêtres sont moyennées par participant. Les poids d'apprentissage compensent le nombre de fenêtres
par participant et sont équilibrés par classe dans chaque ajustement. Les résultats `pooled_*.csv`
sont exploratoires : chaque trouble provenant d'une seule base, une bonne classification à quatre
classes peut refléter les différences de site plutôt que les diagnostics.

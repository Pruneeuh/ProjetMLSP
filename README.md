# Entropies EEG et santé mentale — IFT-7030

Pipeline exploratoire de classification EEG pour trois bases : dépression (Mumtaz),
schizophrénie (Borisov) et anxiété (Sakib). Les résultats ne constituent pas un outil
diagnostique clinique.

## Démarrage rapide sur Valeria

Les données doivent déjà être présentes :

```text
data_eeg/
  mumtaz/      fichiers EDF
  borisov/     norm.zip et sch.zip
  sakib/       fichiers MAT
```

Dans une session disposant des ressources nécessaires :

```bash
module load StdEnv/2023 python/3.12.4 scipy-stack/2024a
module load cuda/12.6
source ~/venvs/projetmlsp-valeria/bin/activate
unset PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
```

Si l'environnement n'existe pas, le créer d'abord après avoir chargé les modules :
`python -m venv ~/venvs/projetmlsp-valeria`.

Depuis la racine du dépôt, avec la wheelhouse Valeria configurée et accès à PyPI et GitHub :

```bash
python 00_notebook_install.py requirement.txt
python -c 'import runpy; exec(runpy.run_path("00_package_installs.py")["SMOKE_TEST"])'
python -c "from eeg_gpu_entropies import detecter_cuda; detecter_cuda()"
python 00_pipeline.py --workers 1
```

L'installateur suit les étapes de conversion de l'outil officiel Alliance
`unmanylinuxize.sh`, en utilisant `setrpaths` et une révision épinglée de son outil
`manipulate_wheels.py`. Il installe ensuite les dépendances et vérifie l'import de CuPy.
Il nécessite `git`, `wget` et `setrpaths` dans le `PATH` ainsi que l'environnement
logiciel Alliance. Dans un noyau Python système, il installe dans les paquets utilisateur,
ce qui peut affecter d'autres notebooks du même compte. Redémarrer le noyau après installation.
Cette conversion n'a pas encore été vérifiée sur Valeria : les wheels et leurs dépendances
peuvent varier. Le fichier reflète un
environnement Python 3.12 observé sur Valeria, pas un verrouillage complet ni une garantie
de compatibilité avec Colab. MNE 1.12.1 nécessite PyPI dans la configuration observée ;
MNE 1.9.0 avec le SciPy observé échoue à la lecture EDF. Pour vérifier une lecture réelle :

```bash
python -c "import mne; r = mne.io.read_raw_edf('data_eeg/mumtaz/H S1 EC.edf', preload=False, verbose=False); print('EDF OK:', len(r.ch_names), 'channels'); r.close()"
```

Pour Jupyter, après activation de l'environnement, enregistrer un noyau s'il n'existe pas :

```bash
python -m ipykernel install --user --name projetmlsp-valeria \
    --display-name "ProjetMLSP (Valeria)" --env PYTHONPATH "" \
    --env PYTHONHOME "" --env PYTHONNOUSERSITE 1
```

Démarrer Jupyter avec les modules chargés et sélectionner ce noyau. Le notebook
`00_pipeline_notebook.ipynb` installe les dépendances pour le noyau sélectionné, vérifie
`pip check`, puis lance le pipeline. Après installation, redémarrer le noyau, relancer la
première cellule et passer directement à la cellule du pipeline. Régler le chemin du dépôt
dans la première cellule si nécessaire.

Le calcul complet peut durer longtemps. Pour un essai seulement :

```bash
python 00_pipeline.py --workers 1 --rapide
```

Le mode rapide réduit modèles et réglage, utilise une répétition et désactive LODO : ses
résultats ne sont pas ceux de l'analyse complète. Pour un essai sans données réelles :
`python outils/generer_donnees_test.py`, puis
`python 00_pipeline.py --donnees data_eeg_test --rapide`. Ces résultats sont synthétiques.

## Méthodologie : quoi et pourquoi

```text
01 données → 02 qualité et prétraitement → 03 fenêtres et caractéristiques
→ 04 partition par participant → 05/06 entraînement et test
→ 08 statistiques → 09 interprétation → 10 synthèse
```

Le lanceur `00_pipeline.py` exécute par défaut **01–06 et 08–10**. L'étape 07 est
facultative. Chaque script accepte `--help`.

### 00 — Orchestration

Lance les étapes dans des processus séparés, conserve leurs journaux et transmet les sorties.
Les sorties 01–04 sont réutilisées seulement si leurs données, code, paramètres, versions et
empreintes correspondent. En 05/06, les calculs terminés sont sauvegardés au fur et à
mesure : relancer le même pipeline reprend les modèles compatibles plutôt que de les
réentraîner. Les références, contrôles de site et LODO sont aussi sauvegardés. Les analyses
optionnelles d'ablation et de permutation sont reprises par bloc terminé ; un bloc
interrompu est recalculé. Les anciens résultats sans checkpoints ne peuvent pas être repris.

La reprise exige les mêmes données de fusion, options, code, versions et configuration de
calcul ; chaque lancement écrit un nouveau dossier et copie les modèles repris. Les
checkpoints prennent de l'espace disque et, comme les modèles `joblib`, ne doivent être
chargés que depuis des sources de confiance. Pour les scripts 05/06 lancés directement,
ajouter `--reprendre`. **Pourquoi :** éviter les recalculs sans réutiliser silencieusement
des résultats incompatibles. `--recalculer` force 01–06 ; les données source doivent rester
disponibles pour vérifier le cache. Les étapes 07–10 ne sont pas mises en cache. Une erreur
en 01–04 arrête toujours le lanceur.

### 01 — Chargement et doublons

Lit Mumtaz en yeux fermés, Borisov et Sakib ; conserve respectivement 19, 16 et 14 canaux
selon la configuration. Les signaux Mumtaz lus en volts sont convertis en µV. Les valeurs
Borisov et Sakib sont traitées comme des µV, mais leur unité source et l'ordre de leurs
canaux ne sont pas confirmés. Les doublons exacts détectés par empreinte sont exclus ;
`387-02w1` est exclu par précaution et `387-03w1` est conservé s'il est exploitable.
Les exclusions et groupes de validation sont enregistrés.

**Pourquoi :** travailler sur des données traçables et éviter qu'une copie connue d'un
enregistrement contribue à l'évaluation. Les empreintes ne détectent pas toutes les copies
transformées ni toutes les identités partagées.

### 02 — Qualité et prétraitement

Signale notamment durée, amplitude, canaux plats, bruit secteur et associations entre
variables d'acquisition et étiquette. Filtre chaque enregistrement entre 0,5 et 45 Hz et
le rééchantillonne à 128 Hz. Le coupe-bande secteur est désactivé par défaut ; en
exécution directe, `--notch 50` ou `--notch 60` l'active.

**Pourquoi :** documenter les problèmes et traiter les signaux de manière cohérente avant
l'extraction. Les alertes ne retirent pas automatiquement les données ; un signal invalide
arrête l'étape. Une alerte de « saturation » n'est pas une preuve d'écrêtage physique.
03 vérifie le manifeste et les empreintes des signaux prétraités, sans les filtrer une
seconde fois.

### 03 — Fenêtres et caractéristiques

Découpe chaque signal en fenêtres complètes de **10 secondes sans chevauchement** ; le
reste trop court est écarté. À 128 Hz, une fenêtre contient 1 280 échantillons par canal.
Chaque canal est décomposé par ondelettes db4 en cinq composantes, puis caractérisé par :

- SampEn, DispEn, FuzzyEn, Shannon spectrale et LZC ;
- ApEn, PermEn et Shannon d'amplitude ;
- LogPower et RelPower, calculées à partir des énergies des composantes DWT ;
- RCMSE et RCMPE aux échelles 1–10 sur le signal large bande de la fenêtre.

**Pourquoi :** représenter différentes propriétés de complexité et d'énergie tout au long
du signal plutôt que résumer d'emblée tout l'enregistrement. Les mesures multiscales sont
calculées en large bande, pas sur la seule composante A4. `LogPower` est le log10 de
l'énergie DWT, pas une puissance moyenne ; `RelPower` est la fraction de l'énergie des
cinq composantes. Les valeurs indéfinies restent manquantes jusqu'à l'imputation dans
les plis d'entraînement.

### 04 — Fusion et partition fixe

Crée une matrice par fenêtre, avec tous les canaux de chaque base pour 05 et les dix
canaux supposés communs pour 06. Les groupes de validation sont stratifiés par base et
étiquette, puis répartis une fois en environ **80 % entraînement / 20 % test**. Toutes les
fenêtres et tous les enregistrements liés d'un groupe restent ensemble. La partition est
sauvegardée dans `partition_sujets.csv` et partagée par 05 et 06.

**Pourquoi :** évaluer sur des participants non vus, et non sur d'autres fenêtres d'un
participant déjà appris. Les proportions exactes varient avec les effectifs entiers.
04 produit aussi des agrégats par participant et des harmonisations descriptives ajustées
sur tous les témoins ; ces fichiers descriptifs ne servent pas à entraîner 05/06.

### 05 — Classification binaire

Pour chaque base séparément, les modèles apprennent à distinguer trouble et témoin à
partir des fenêtres d'entraînement. Le réglage des hyperparamètres est effectué uniquement
sur ces participants par validation croisée groupée. Les modèles prédisent ensuite chaque
fenêtre test ; leurs probabilités sont moyennées par participant avant le calcul des
métriques. Des références testent aussi l'amplitude et, quand disponible, le nombre de
canaux d'origine.

**Pourquoi :** mesurer la performance sur des participants réservés et détecter si des
indices d'acquisition suffisent à séparer les groupes. L'ablation est désactivée dans 00 ;
elle reste disponible avec `python 05_classification_binaire.py --ablation` et utilise
uniquement les participants d'entraînement.

### 06 — Classification multiclasse et contrôles

Entraîne des modèles sur les fenêtres communes pour prédire témoin, dépression,
schizophrénie ou anxiété, avec la même partition fixe et une évaluation par participant.
Compare quatre méthodes d'harmonisation (`zscore`, `filtrage`, `coral`,
`filtrage_coral`) ainsi qu'une référence sans harmonisation pour certains modèles.
L'harmonisation utilisée pour le test principal et les plis de réglage est ajustée à partir
des témoins d'entraînement correspondants.

Deux analyses supplémentaires sont activées par défaut sur la seule partition
d'entraînement :

- **Contrôles de site :** prédire la base d'origine, pour tous les participants puis pour
  les témoins seuls, afin de vérifier si des différences de base restent détectables.
- **Leave-one-dataset-out (LODO) :** entraîner un modèle binaire trouble/témoin sur deux
  bases et l'évaluer sur la troisième. L'harmonisation utilise soit une partie de ses
  témoins, exclus de l'évaluation, soit ses données sans étiquette. C'est donc une
  expérience d'adaptation à la base cible, pas une validation externe totalement intacte.

Les tests de permutation d'étiquettes sont disponibles en option, mais leur nombre est
zéro par défaut. **Pourquoi :** les contrôles de site et LODO explorent des biais et la
généralisation au-delà du score multiclasse.

### 07 — Réseaux de neurones, facultatif

Entraîne des références et des réseaux sur des caractéristiques classiques agrégées par
participant, avec sa propre validation croisée. Il n'est pas inclus dans le lancement par
défaut et ne reproduit pas l'entraînement sur fenêtres de 05/06. **Pourquoi :** comparer
une autre famille de modèles sans ajouter son coût au pipeline standard.

### 08 — Statistiques de groupes

Compare les groupes par base sur des caractéristiques classiques agrégées et l'amplitude
(Mann–Whitney, correction FDR et tailles d'effet), puis explore la similarité entre
troubles. **Pourquoi :** décrire les différences observées indépendamment des scores des
classificateurs. Ces analyses restent exploratoires et ne prouvent pas que les différences
sont causées par le trouble.

### 09 — Interprétabilité

Charge les modèles sur fenêtres effectivement ajustés et sauvegardés par 05/06, vérifie
la fusion et reproduit leurs probabilités test avant de les expliquer. Par défaut, il
explique la régression logistique ; `--modele` permet de choisir un autre modèle sauvegardé.
Pour 06, `--harmonisation` accepte `zscore` ou `aucune` : les autres transformations ne
sont pas prises en charge pour cette interprétation par variable. Les références
majoritaire, d'amplitude et de nombre de canaux ne sont pas sauvegardées pour 09.

Le lanceur active SHAP et les importances par permutation de groupes. SHAP est calculé par
fenêtre, puis les valeurs absolues sont moyennées par participant et entre participants :
chaque participant a le même poids, même si les durées diffèrent. Ce n'est pas une valeur
SHAP directement calculée sur la moyenne des probabilités du participant. Les permutations
échangent les trajectoires des variables entre participants ; les longueurs différentes
sont rééchantillonnées par indices relatifs.

**Pourquoi :** explorer les caractéristiques utilisées par les modèles réellement
entraînés, sans considérer les fenêtres d'un même participant comme des personnes
indépendantes. Les explications portent sur le même test que 05/06 : leur utilisation
pour choisir des caractéristiques rend ce test impropre à une évaluation finale
indépendante.

### 10 — Synthèse

Rassemble les prédictions disponibles, métriques par classe, comparaisons et intervalles
bootstrap par participant ; inclut les résultats SHAP ou d'ablation s'ils sont disponibles.
**Pourquoi :** faciliter la lecture et le rapport, sans remplacer une validation externe.
Par défaut, 10 cherche les sorties les plus récentes : vérifier leurs dossiers et leur
provenance avant de comparer des analyses différentes.

## Réglage et interprétation des scores

- Dans 05 et 06, le profil complet utilise **4 plis internes groupés** et jusqu'à
  **40 candidats** par modèle (25 pour les arbres). Imputation, sélection de variables et,
  lorsqu'elle est utilisée, standardisation sont ajustées dans les plis ; le critère interne est
  l'exactitude équilibrée **par fenêtre**, pas par participant.
- Les deux étapes utilisent **3 répétitions par défaut** : elles réajustent les modèles
  avec des graines différentes sur le **même test**, et ne représentent pas trois
  partitions test indépendantes. Les contrôles de site et LODO répètent aussi leurs
  propres plis ou choix de calibration. En exécution directe, `--repetitions` permet de
  modifier ce nombre.
- Les enregistrements longs fournissent plus de fenêtres : ils peuvent peser davantage
  dans l'apprentissage et le réglage. Une fenêtre supplémentaire n'est pas un nouveau
  participant.
- L'exactitude équilibrée est la moyenne des rappels par classe ; l'AUC mesure la qualité
  du classement. Des ablations évaluées sur train en CV ne sont pas directement comparables
  aux scores sur le test réservé.

## Limites importantes

- **Base et diagnostic sont confondus :** chaque trouble provient d'une base différente.
  Les scores multiclasse peuvent refléter le site, le matériel, l'âge ou le protocole.
  Une harmonisation ou un bon score LODO ne prouve pas une signature spécifique du trouble.
- La référence Mumtaz basée sur le nombre de canaux d'origine peut révéler une forte
  confusion d'acquisition. Un bon score d'amplitude peut lui aussi refléter une
  caractéristique technique plutôt qu'un effet clinique.
- La séparation par groupe réduit les fuites connues, mais ne garantit pas leur absence.
  Choisir le meilleur modèle après avoir comparé les scores du même test rend l'estimation
  optimiste. Utiliser les résultats 08 ou les explications 09 pour choisir des variables,
  puis présenter 05/06 sur ce même test comme une évaluation finale indépendante, contamine
  également cette évaluation.
- Les ordres de canaux Borisov et Sakib et leurs unités restent à vérifier ; les valeurs
  d'amplitude et cartes spatiales doivent être interprétées avec prudence.
- Les résultats d'un petit test et les intervalles bootstrap restent incertains. Une
  évaluation finale exige des participants indépendants et idéalement des bases où chaque
  diagnostic est représenté dans plusieurs sites.

## GPU, sorties et options

- 03 essaie CuPy/CUDA pour des calculs de SampEn, FuzzyEn et RCMSE ; le reste de
  l'extraction demeure sur CPU. En mode CUDA, 03 utilise un seul processus. La détection
  réussie n'est pas une garantie de performance ni de mémoire suffisante.
- XGBoost et LightGBM sondent séparément leur GPU. Le lanceur essaie aussi RAPIDS
  `cuml.accel` s'il est installé ; sinon les modèles scikit-learn restent sur CPU.
  Vérifier les journaux : tous les modèles et opérations ne sont pas nécessairement sur GPU.
  `--sans-rapids` désactive RAPIDS et `--cpu-entropies` force l'extraction CPU.
- Les résultats et journaux sont écrits dans `sorties/`. Utiliser `--etapes` pour lancer
  certaines étapes, par exemple `python 00_pipeline.py --etapes 7 10`. Les options propres
  à une étape se règlent en lançant directement son script.
- 05/06 sauvegardent les modèles et leur provenance dans leurs dossiers de sortie ; cela
  utilise de l'espace disque. Les anciennes exécutions sans `modeles.json` doivent être
  relancées avant 09. La fusion d'origine doit rester accessible et inchangée. Charger
  uniquement des modèles `joblib` de confiance, avec un environnement compatible : ces
  fichiers peuvent exécuter du code lors de leur chargement.
- Avec `--continuer`, 09 ou 10 ne réutilise pas des résultats anciens lorsqu'une étape
  05, 06 ou 09 demandée pour ce lancement a échoué.
- `entropy_classification.ipynb` est un workflow distinct. Ses anciennes sorties peuvent
  précéder l'exclusion de Borisov ; ses analyses au niveau participant et au niveau
  fenêtre ne sont pas une comparaison contrôlée à caractéristiques identiques.

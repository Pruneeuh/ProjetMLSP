"""Lance tout le pipeline dans l'ordre (ou une partie), chaque étape dans son propre processus.

    01 chargement des bases brutes        06 classification 4 classes + contrôles de site
    02 contrôle de qualité                07 réseaux de neurones (sur demande seulement)
    03 extraction des entropies           08 statistiques de groupes
    04 fusion + harmonisation             09 interprétabilité
    05 classification binaire             10 synthèse (IC 95 %, tableaux du rapport)

Les sorties 01–04 sont réutilisées si leurs entrées, paramètres, code, dépendances et
empreintes correspondent. --recalculer force leur exécution.
Une étape qui échoue arrête le pipeline (sauf --continuer, sauf pour 01–04) ; le journal de chaque étape est
écrit dans sorties/journal_pipeline_<date>/<étape>.log.

Usage :
    python 00_pipeline.py                          # 01–06 et 08–10, sans réseaux
    python 00_pipeline.py --rapide                 # essai réduit (peu de répétitions/époques)
    python 00_pipeline.py --etapes 4 5 6 10        # seulement certaines étapes
    python 00_pipeline.py --etapes 7 10             # réseaux explicitement demandés
    python 00_pipeline.py --donnees data_eeg_test --rapide    # données synthétiques (outils/generer_donnees_test.py)
    python 00_pipeline.py --workers 4 --sans-reseaux

Les options propres à une étape se règlent en lançant l'étape directement (python 06_... --help).
"""
import argparse
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from eeg_cache_pipeline import CachePipeline, signature, sources_brutes
from eeg_config import BASES

SCRIPT_DIR = Path(__file__).resolve().parent
ETAPES = {
    1: "01_chargement_donnees.py", 2: "02_controle_qualite.py", 3: "03_extraction_entropies.py",
    4: "04_fusion_harmonisation.py", 5: "05_classification_binaire.py", 6: "06_classification_multiclasse.py",
    7: "07_reseaux_neurones.py", 8: "08_statistiques_groupes.py", 9: "09_interpretabilite.py",
    10: "10_synthese_resultats.py",
}
ETAPES_DEFAUT = [e for e in sorted(ETAPES) if e != 7]
RAPIDE = {
    5: ["--repetitions", "1", "--modeles", "regression_logistique", "--profil-tuning", "quick",
        "--plis-internes", "2", "--tirages-tuning", "1"],
    6: ["--repetitions", "1", "--modeles", "regression_logistique", "--harmonisations", "zscore",
        "--profil-tuning", "quick", "--plis-internes", "2", "--tirages-tuning", "1", "--sans-lodo"],
    7: ["--repetitions", "1", "--epoques", "30"],
    9: ["--repetitions", "1", "--tirages", "2", "--profil-tuning", "quick",
        "--plis-internes", "2", "--tirages-tuning", "1"],
    10: ["--bootstrap", "300"],
}
OPTIONS_ANALYSE = {
    4: ["--caracteristiques-etendues"],
    5: ["--caracteristiques", "etendues_puissances", "--ablation"],
    6: ["--caracteristiques", "etendues_puissances"],
    9: ["--caracteristiques", "etendues_puissances", "--shap"],
}


def dernier(parent, prefixe="", contient=None):
    if not parent.is_dir():
        return None
    dirs = sorted(p for p in parent.iterdir() if p.is_dir() and p.name.startswith(prefixe)
                  and (contient is None or (p / contient).is_file()))
    return dirs[-1] if dirs else None


def entrees_etape(etape, args, resultats):
    root = SCRIPT_DIR / "sorties"
    if etape == 1:
        return sources_brutes(args.donnees or SCRIPT_DIR / "data_eeg")
    if etape == 2:
        return resultats.get(1, [root / "01_donnees"])
    if etape == 3:
        qc = resultats.get(2, [dernier(root / "02_controle_qualite", contient="controle_qualite.json")])[0]
        if qc is None:
            raise ValueError("Aucun prétraitement 02 complet ; lancer 02.")
        return [qc]
    if 3 in resultats:
        return resultats[3]
    runs = [dernier(root / "03_extraction", f"{b}_", "parametres_extraction.json") for b in BASES]
    if any(p is None for p in runs):
        raise ValueError("Extractions 03 complètes absentes ; lancer 03.")
    return runs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--etapes", type=int, nargs="+", choices=sorted(ETAPES), default=ETAPES_DEFAUT)
    p.add_argument("--donnees", type=Path, default=None, help="Dossier des données brutes (étape 1).")
    p.add_argument("--workers", type=int, default=None, help="Processus pour l'extraction (étape 3).")
    p.add_argument("--rapide", action="store_true", help="Réglages réduits pour un essai.")
    p.add_argument("--sans-reseaux", action="store_true", help="Sauter l'étape 7 même si elle est demandée.")
    p.add_argument("--continuer", action="store_true", help="Continuer après un échec, sauf pour les étapes 01–04.")
    p.add_argument("--recalculer", action="store_true", help="Recalculer 01–04 même si leurs sorties sont vérifiées.")
    args = p.parse_args()
    etapes = [e for e in args.etapes if not (args.sans_reseaux and e == 7)]
    journal = SCRIPT_DIR / "sorties" / f"journal_pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    journal.mkdir(parents=True, exist_ok=True)
    pretraitement = SCRIPT_DIR / "sorties" / "02_controle_qualite" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    cache = CachePipeline(SCRIPT_DIR / "sorties" / "cache_pipeline.json")
    resultats, reutilisees = {}, set()
    bilan = []
    t_all = time.time()
    for e in etapes:
        cmd = [sys.executable, str(SCRIPT_DIR / ETAPES[e])]
        cmd += OPTIONS_ANALYSE.get(e, [])
        inputs = []
        key = None
        if e <= 4:
            try:
                inputs = entrees_etape(e, args, resultats)
                options = OPTIONS_ANALYSE.get(e, []) + (["--workers", str(args.workers)] if e == 3 and args.workers else [])
                key = signature(e, inputs, options, SCRIPT_DIR)
                cached = None if args.recalculer else cache.trouver(e, key)
            except (OSError, ValueError) as exc:
                print(f"Étape {e:02d} : vérification des entrées impossible : {exc}", flush=True)
                bilan.append((e, ETAPES[e], 1, 0))
                break
            if cached:
                resultats[e] = cached
                reutilisees.add(e)
                bilan.append((e, ETAPES[e], 0, 0))
                print(f"\n# Étape {e:02d} : sorties vérifiées réutilisées : "
                      + ", ".join(str(p) for p in cached), flush=True)
                continue
        if e == 1 and args.donnees:
            cmd += ["--donnees", str(args.donnees)]
        if e == 2:
            cmd += ["--donnees", str(inputs[0]), "--sortie", str(pretraitement)]
        if e == 3:
            cmd += ["--donnees", str(inputs[0])]
        if e == 3 and args.workers:
            cmd += ["--workers", str(args.workers)]
        if e == 4:
            for base, path in zip(BASES, sorted(inputs, key=lambda p: BASES.index(p.name.split("_", 1)[0]))):
                cmd += [f"--{base}", str(path)]
        if e in (5, 7, 8, 9) and 3 in resultats:
            for path in resultats[3]:
                base = path.name.split("_", 1)[0]
                cmd += [f"--{base}", str(path)]
        if e in (6, 7, 8, 9) and 4 in resultats:
            cmd += ["--fusion", str(resultats[4][0])]
        if e == 10 and not any(step == 7 and code == 0 for step, _, code, _ in bilan):
            cmd += ["--sans-profond"]
        if args.rapide:
            cmd += RAPIDE.get(e, [])
        print(f"\n{'#' * 78}\n# Étape {e:02d} : {ETAPES[e]}  {' '.join(cmd[2:])}\n{'#' * 78}", flush=True)
        t0 = time.time()
        sorties = []
        with open(journal / f"{ETAPES[e][:-3]}.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", bufsize=1,
                                    env={**os.environ, "PYTHONIOENCODING": "utf-8"})
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
                if line.strip().startswith("SORTIE : "):
                    sorties.append(Path(line.strip().split(" : ", 1)[1]))
            code = proc.wait()
        if code == 0 and e <= 4:
            if e == 1:
                sorties = [SCRIPT_DIR / "sorties" / "01_donnees"]
            elif e == 2:
                sorties = [pretraitement]
            try:
                if signature(e, inputs, options, SCRIPT_DIR) != key:
                    raise ValueError("les entrées ont changé pendant l'exécution")
                cache.enregistrer(e, key, sorties)
                resultats[e] = sorties
            except (OSError, ValueError) as exc:
                print(f"Étape {e:02d} : sortie non vérifiable : {exc}", flush=True)
                code = 1
        duree = time.time() - t0
        bilan.append((e, ETAPES[e], code, duree))
        if code != 0:
            print(f"\n!! Étape {e:02d} en échec (code {code}) ; journal : {journal / (ETAPES[e][:-3] + '.log')}")
            if e <= 4 or not args.continuer:
                break
    print(f"\n{'=' * 78}\nBILAN DU PIPELINE ({time.time() - t_all:.0f} s)\n{'=' * 78}")
    for e, nom, code, duree in bilan:
        status = "RÉUTILISÉ" if e in reutilisees else ("OK" if code == 0 else f"ÉCHEC ({code})")
        print(f"  {e:02d} {nom:<34s} {status:<12s} {duree:7.0f} s")
    print(f"Journaux : {journal}")
    sys.exit(0 if all(c == 0 for _, _, c, _ in bilan) else 1)


if __name__ == "__main__":
    main()

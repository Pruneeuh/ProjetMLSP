"""Lance tout le pipeline dans l'ordre (ou une partie), chaque étape dans son propre processus.

    01 chargement des bases brutes        06 classification 4 classes + contrôles de site
    02 contrôle de qualité                07 réseaux de neurones
    03 extraction des entropies           08 statistiques de groupes
    04 fusion + harmonisation             09 interprétabilité
    05 classification binaire             10 synthèse (IC 95 %, tableaux du rapport)

Chaque étape lit par défaut la sortie la plus récente de l'étape précédente dans sorties/.
Une étape qui échoue arrête le pipeline (sauf --continuer) ; le journal de chaque étape est
écrit dans sorties/journal_pipeline_<date>/<étape>.log.

Usage :
    python 00_pipeline.py                          # tout, réglages par défaut (long : permutations, réseaux)
    python 00_pipeline.py --rapide                 # essai rapide (peu de répétitions/permutations/époques)
    python 00_pipeline.py --etapes 4 5 6 10        # seulement certaines étapes
    python 00_pipeline.py --donnees data_eeg_test --rapide    # données synthétiques (outils/generer_donnees_test.py)
    python 00_pipeline.py --workers 4 --sans-reseaux

Les options propres à une étape se règlent en lançant l'étape directement (python 06_... --help).
"""
import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
ETAPES = {
    1: "01_chargement_donnees.py", 2: "02_controle_qualite.py", 3: "03_extraction_entropies.py",
    4: "04_fusion_harmonisation.py", 5: "05_classification_binaire.py", 6: "06_classification_multiclasse.py",
    7: "07_reseaux_neurones.py", 8: "08_statistiques_groupes.py", 9: "09_interpretabilite.py",
    10: "10_synthese_resultats.py",
}
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--etapes", type=int, nargs="+", choices=sorted(ETAPES), default=sorted(ETAPES))
    p.add_argument("--donnees", type=Path, default=None, help="Dossier des données brutes (étape 1).")
    p.add_argument("--workers", type=int, default=None, help="Processus pour l'extraction (étape 3).")
    p.add_argument("--rapide", action="store_true", help="Réglages réduits pour un essai.")
    p.add_argument("--sans-reseaux", action="store_true", help="Sauter l'étape 7 (PyTorch).")
    p.add_argument("--continuer", action="store_true", help="Continuer après une étape en échec.")
    args = p.parse_args()
    etapes = [e for e in args.etapes if not (args.sans_reseaux and e == 7)]
    journal = SCRIPT_DIR / "sorties" / f"journal_pipeline_{datetime.now():%Y%m%d_%H%M%S}"
    journal.mkdir(parents=True, exist_ok=True)
    bilan = []
    t_all = time.time()
    for e in etapes:
        cmd = [sys.executable, str(SCRIPT_DIR / ETAPES[e])]
        if e == 1 and args.donnees:
            cmd += ["--donnees", str(args.donnees)]
        if e == 3 and args.workers:
            cmd += ["--workers", str(args.workers)]
        if args.rapide:
            cmd += RAPIDE.get(e, [])
        print(f"\n{'#' * 78}\n# Étape {e:02d} : {ETAPES[e]}  {' '.join(cmd[2:])}\n{'#' * 78}", flush=True)
        t0 = time.time()
        with open(journal / f"{ETAPES[e][:-3]}.log", "w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=SCRIPT_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, bufsize=1)
            for line in proc.stdout:
                print(line, end="", flush=True)
                log.write(line)
            code = proc.wait()
        duree = time.time() - t0
        bilan.append((e, ETAPES[e], code, duree))
        if code != 0:
            print(f"\n!! Étape {e:02d} en échec (code {code}) ; journal : {journal / (ETAPES[e][:-3] + '.log')}")
            if not args.continuer:
                break
    print(f"\n{'=' * 78}\nBILAN DU PIPELINE ({time.time() - t_all:.0f} s)\n{'=' * 78}")
    for e, nom, code, duree in bilan:
        print(f"  {e:02d} {nom:<34s} {'OK' if code == 0 else f'ÉCHEC ({code})':<12s} {duree:7.0f} s")
    print(f"Journaux : {journal}")
    sys.exit(0 if all(c == 0 for _, _, c, _ in bilan) else 1)


if __name__ == "__main__":
    main()

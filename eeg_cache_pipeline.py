"""Vérification des entrées et sorties réutilisables du lanceur."""
import hashlib
import json
import os
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from eeg_config import BASES, SOUS_DOSSIER_BRUT


CODE = {
    1: ["01_chargement_donnees.py", "eeg_config.py", "eeg_utils.py"],
    2: ["02_controle_qualite.py", "eeg_config.py", "eeg_utils.py", "eeg_donnees.py", "eeg_pretraitement.py"],
    3: ["03_extraction_entropies.py", "eeg_config.py", "eeg_utils.py", "eeg_donnees.py", "eeg_pretraitement.py"],
    4: ["04_fusion_harmonisation.py", "eeg_config.py", "eeg_utils.py", "eeg_donnees.py", "eeg_harmonisation.py"],
}
PAQUETS = {
    1: ["numpy", "scipy", "pandas", "mne"],
    2: ["numpy", "scipy", "pandas", "matplotlib", "scikit-learn"],
    3: ["numpy", "scipy", "pandas", "PyWavelets", "EntropyHub"],
    4: ["numpy", "scipy", "pandas", "scikit-learn", "matplotlib"],
}


def empreinte_fichier(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def empreinte_chemin(path):
    path = Path(path).resolve()
    if path.is_file():
        return {str(path): empreinte_fichier(path)}
    if not path.is_dir():
        raise FileNotFoundError(f"Entrée ou sortie introuvable : {path}")
    files = sorted(p for p in path.rglob("*") if p.is_file())
    if not files:
        raise ValueError(f"Dossier vide : {path}")
    return {str(p.relative_to(path)): empreinte_fichier(p) for p in files}


def signature(etape, entrees, options, racine):
    versions = {}
    for package in PAQUETS[etape]:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    data = {
        "schema": 1, "etape": etape, "options": options,
        "python": sys.version, "paquets": versions,
        "code": {name: empreinte_fichier(Path(racine) / name) for name in CODE[etape]},
        "entrees": {str(Path(p).resolve()): empreinte_chemin(p) for p in entrees},
    }
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode("utf-8")).hexdigest()


def verifier_sorties(etape, chemins):
    chemins = [Path(p) for p in chemins]
    if len(chemins) != (len(BASES) if etape == 3 else 1):
        raise ValueError(f"Étape {etape:02d} : nombre de dossiers de sortie inattendu.")
    if etape in (1, 2):
        names = [f"{base}_{suffix}" for base in BASES
                 for suffix in ("signals.npz", "metadata.csv", "canaux.json")]
        names += ["inventaire_chargement.json", "inventaire_chargement.csv"] if etape == 1 else ["controle_qualite.json"]
        required = [(chemins[0], names)]
    elif etape == 3:
        required = [(p, ["features_windows.csv", "features_sujets.csv", "audit_enregistrements.csv",
                         "parametres_extraction.json"]) for p in chemins]
        if {p.name.split("_", 1)[0] for p in chemins} != set(BASES):
            raise ValueError("Étape 03 : sorties manquantes ou bases inattendues.")
    else:
        required = [(chemins[0], ["entropies_sujets.csv", "amplitude_sujets.csv", "meta_sujets.csv",
                                   "entropies_etendues_sujets.csv", "puissances_sujets.csv",
                                   "entropies_multiechelle_sujets.csv", "fusion.json"])]
    for directory, names in required:
        absent = [name for name in names if not (directory / name).is_file() or (directory / name).stat().st_size == 0]
        if absent:
            raise ValueError(f"Étape {etape:02d} : fichiers absents ou vides dans {directory} : {absent}")
    if etape == 1:
        inventory = json.loads((chemins[0] / "inventaire_chargement.json").read_text(encoding="utf-8"))
        resume = inventory.get("resume") if isinstance(inventory, dict) else None
        if (not isinstance(resume, dict) or
                any(not isinstance(resume.get(base), dict) or
                    not isinstance(resume[base].get("retenus"), int) or resume[base]["retenus"] <= 0
                    for base in BASES)):
            raise ValueError("Étape 01 : l'inventaire ne confirme pas des enregistrements pour chaque base.")
    return {str(p.resolve()): empreinte_chemin(p) for p in chemins}


class CachePipeline:
    def __init__(self, path):
        self.path = Path(path)
        self.records = {}
        if self.path.is_file():
            try:
                data = json.loads(self.path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("format invalide")
                self.records = data
            except (OSError, ValueError) as exc:
                print(f"Cache illisible ({self.path}) : {exc} ; les étapes seront recalculées.", flush=True)

    def trouver(self, etape, signature_etape):
        record = self.records.get(signature_etape)
        if not isinstance(record, dict) or record.get("etape") != etape:
            return None
        try:
            paths = [Path(p) for p in record["sorties"]]
            if verifier_sorties(etape, paths) == record["empreintes"]:
                return paths
            print(f"Cache étape {etape:02d} : sorties modifiées ; recalcul.", flush=True)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            print(f"Cache étape {etape:02d} invalide : {exc} ; recalcul.", flush=True)
        return None

    def enregistrer(self, etape, signature_etape, chemins):
        hashes = verifier_sorties(etape, chemins)
        self.records[signature_etape] = {"etape": etape, "sorties": [str(Path(p).resolve()) for p in chemins],
                                         "empreintes": hashes}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp = self.path.with_name(self.path.name + f".{os.getpid()}.tmp")
        try:
            temp.write_text(json.dumps(self.records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            os.replace(temp, self.path)
        finally:
            if temp.exists():
                temp.unlink()


def sources_brutes(root):
    return [Path(root) / SOUS_DOSSIER_BRUT[base] for base in BASES]

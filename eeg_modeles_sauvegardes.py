"""Persistence and provenance for fitted holdout models."""

import json
from pathlib import Path

import joblib

from eeg_cache_pipeline import empreinte_fichier
from eeg_utils import ecrire_json, require


def sauvegarder_modele(dossier, tache, nom, harmonisation, repetition, modele):
    models = Path(dossier) / "modeles"
    models.mkdir(exist_ok=True)
    path = models / f"{tache}__{nom}__{harmonisation}__r{repetition:03d}.joblib"
    joblib.dump(modele, path)
    return dict(tache=tache, modele=nom, harmonisation=harmonisation,
                repetition=repetition, fichier=str(path.relative_to(dossier)))


def empreintes_fusion(fusion, fichiers):
    return {name: empreinte_fichier(Path(fusion) / name) for name in fichiers}


def ecrire_manifeste(dossier, fusion, empreintes, arguments, modeles):
    fusion = Path(fusion).resolve()
    require(empreintes_fusion(fusion, empreintes) == empreintes,
            f"Fusion modifiée pendant l'entraînement : {fusion}.")
    ecrire_json(Path(dossier) / "modeles.json", {
        "fusion": str(fusion), "empreintes": empreintes,
        "arguments": arguments, "modeles": modeles,
    })


def lire_manifeste(dossier, fusion=None):
    dossier = Path(dossier)
    path = dossier / "modeles.json"
    require(path.is_file(), f"{path} absent ; relancer 05 ou 06 pour sauvegarder les modèles.")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    source = Path(manifest["fusion"])
    if fusion is not None:
        require(Path(fusion).resolve() == source.resolve(),
                f"Fusion différente de celle utilisée pour {dossier}.")
    for name, expected in manifest["empreintes"].items():
        path = source / name
        require(path.is_file() and empreinte_fichier(path) == expected,
                f"Fusion absente ou modifiée : {path}.")
    return manifest

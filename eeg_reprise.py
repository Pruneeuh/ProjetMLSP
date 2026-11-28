"""Reprise des calculs terminés, avec vérification des entrées et de l'environnement."""

import hashlib
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import joblib

from eeg_cache_pipeline import empreinte_fichier
from eeg_utils import dossier_etape, ecrire_json, log, to_jsonable, versions


PAQUETS = ("numpy", "scipy", "pandas", "scikit-learn", "joblib", "xgboost", "lightgbm",
           "cuml", "cuml-cu11", "cuml-cu12", "cupy", "cupy-cuda11x", "cupy-cuda12x")


class Reprise:
    def __init__(self, etape, args, empreintes, out):
        root = Path(__file__).resolve().parent
        options = {k: v for k, v in vars(args).items() if k != "reprendre"}
        options["fusion"] = str(Path(args.fusion).resolve())
        provenance = to_jsonable({
            "schema": 1, "etape": etape, "arguments": options, "fusion": empreintes,
            "python": sys.version, "plateforme": platform.platform(),
            "paquets": versions(PAQUETS),
            "code": {p.name: empreinte_fichier(p) for p in sorted(root.glob("*.py"))},
        })
        self.signature = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode("utf-8")).hexdigest()
        self.sources = []
        parent = dossier_etape(etape)
        if args.reprendre and parent.is_dir():
            for path in sorted(parent.iterdir(), reverse=True):
                meta = path / "reprise.json"
                if path.is_dir() and meta.is_file():
                    data = json.loads(meta.read_text(encoding="utf-8"))
                    if data.get("signature") == self.signature:
                        self.sources.append(path)
        self.out = Path(out)
        self.checkpoints = self.out / "reprise"
        self.checkpoints.mkdir()
        ecrire_json(self.out / "reprise.json", {"signature": self.signature, "provenance": provenance})

    @staticmethod
    def nom(cle):
        digest = hashlib.sha256(json.dumps(cle, ensure_ascii=False).encode("utf-8")).hexdigest()
        return digest + ".joblib"

    def charger(self, cle):
        name = self.nom(cle)
        for source in self.sources:
            path = source / "reprise" / name
            if not path.is_file():
                continue
            checkpoint = joblib.load(path)
            if checkpoint["cle"] != cle:
                raise ValueError(f"Checkpoint incohérent : {path}")
            for relative, expected in checkpoint["fichiers"].items():
                src, dst = source / relative, self.out / relative
                if not src.is_file() or empreinte_fichier(src) != expected:
                    raise ValueError(f"Fichier de modèle absent ou modifié : {src}")
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(src, dst)
                if empreinte_fichier(dst) != expected:
                    raise ValueError(f"Copie de modèle incohérente : {dst}")
            self.enregistrer(cle, checkpoint["valeur"], list(checkpoint["fichiers"]))
            log(f"  Reprise : {' / '.join(map(str, cle))}")
            return checkpoint["valeur"]
        return None

    def enregistrer(self, cle, valeur, fichiers=()):
        data = {"cle": cle, "valeur": valeur,
                "fichiers": {name: empreinte_fichier(self.out / name) for name in fichiers}}
        path = self.checkpoints / self.nom(cle)
        temp = path.with_name(path.name + f".{os.getpid()}.tmp")
        try:
            joblib.dump(data, temp)
            os.replace(temp, path)
        finally:
            if temp.exists():
                temp.unlink()

"""Prétraitement déterministe d'un enregistrement EEG individuel."""
import math

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample_poly, sosfiltfilt

from eeg_config import BANDPASS, FS_CIBLE


def bandpass(sig, fs, low=BANDPASS[0], high=BANDPASS[1]):
    high = min(high, 0.95 * fs / 2)
    sos = butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return sosfiltfilt(sos, sig, axis=-1)


def notch(sig, fs, f0):
    b, a = iirnotch(f0, Q=30, fs=fs)
    return filtfilt(b, a, sig, axis=-1)


def reechantillonner(sig, fs, fs_cible=FS_CIBLE):
    if int(round(fs)) == fs_cible:
        return sig
    g = math.gcd(int(round(fs)), fs_cible)
    return resample_poly(sig, fs_cible // g, int(round(fs)) // g, axis=-1)


def pretraiter(sig, fs, canaux, f0=None):
    """Valide et filtre un seul enregistrement ; aucun paramètre inter-sujets n'est appris."""
    if sig is None:
        raise ValueError("signal absent de l'archive")
    sig = np.asarray(sig, dtype=np.float64)
    if sig.ndim != 2 or sig.shape[0] != len(canaux) or sig.shape[1] == 0:
        raise ValueError(f"forme du signal invalide : {sig.shape}, {len(canaux)} canaux attendus")
    if not np.isfinite(sig).all():
        raise ValueError(f"{np.count_nonzero(~np.isfinite(sig))} valeurs non finies")
    if not np.isfinite(fs) or fs <= 0 or not float(fs).is_integer():
        raise ValueError(f"fréquence d'échantillonnage invalide ou non entière : {fs}")
    if 0.95 * fs / 2 < BANDPASS[1]:
        raise ValueError(f"fréquence {fs:g} Hz insuffisante pour le filtre {BANDPASS[1]:g} Hz")
    if f0 is not None and (not np.isfinite(f0) or not 0 < f0 < fs / 2):
        raise ValueError(f"fréquence coupe-bande {f0} incompatible avec {fs:g} Hz")
    filtered = bandpass(sig, fs)
    if f0 is not None:
        filtered = notch(filtered, fs, f0)
    result = reechantillonner(filtered, fs)
    expected = math.ceil(sig.shape[1] * FS_CIBLE / fs)
    if result.shape != (len(canaux), expected) or not np.isfinite(result).all():
        raise ValueError(f"sortie prétraitée invalide : {result.shape}, attendu ({len(canaux)}, {expected})")
    return result

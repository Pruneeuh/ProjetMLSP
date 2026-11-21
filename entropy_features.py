"""Extract EEG entropy features described in the project report.

The script computes five features for every channel of every saved recording:
Sample Entropy, Dispersion Entropy, Fuzzy Entropy, Shannon Entropy, and
Lempel-Ziv Complexity. Signals are reduced to a bounded number of samples
before the quadratic measures are evaluated.

Usage:
    python entropy_features.py --dataset borisov --max-samples 512
    python entropy_features.py --dataset all --output outputs/entropy_features.csv

Dependencies: numpy, pandas, scipy
"""
import argparse
import math
from pathlib import Path
from typing import Dict, Iterable, Iterator, Tuple

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.special import ndtr


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "outputs"
DATASETS = ("mumtaz", "borisov", "sakib")


def _clean_signal(signal: np.ndarray, max_samples: int) -> np.ndarray:
    """Return a finite, approximately uniformly resampled 1-D signal."""
    values = np.asarray(signal, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size < 3:
        raise ValueError("a signal must contain at least three finite samples")
    if values.size > max_samples:
        old_x = np.linspace(0.0, 1.0, values.size)
        new_x = np.linspace(0.0, 1.0, max_samples)
        values = np.interp(new_x, old_x, values)
    return values


def _templates(signal: np.ndarray, dimension: int) -> np.ndarray:
    return np.lib.stride_tricks.sliding_window_view(signal, dimension)


def sample_entropy(signal: np.ndarray, m: int = 2, r_factor: float = 0.2) -> float:
    """Compute SampEn using Chebyshev-distance template matches."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size <= m + 1:
        return float("nan")
    tolerance = r_factor * np.std(x)
    if tolerance <= 0 or not np.isfinite(tolerance):
        return 0.0
    counts = []
    for dimension in (m, m + 1):
        templates = _templates(x, dimension)
        pairs = cKDTree(templates).query_pairs(
            tolerance, p=np.inf, output_type="ndarray"
        )
        counts.append(len(pairs))
    if counts[0] == 0 or counts[1] == 0:
        return float("nan")
    return float(-math.log(counts[1] / counts[0]))


def dispersion_entropy(signal: np.ndarray, bins: int = 10, m: int = 2) -> float:
    """Compute DispEn from normal-CDF symbols and ordinal dispersion patterns."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size < m:
        return float("nan")
    scale = np.std(x)
    if scale <= 0 or not np.isfinite(scale):
        return 0.0
    symbols = np.floor(np.clip(ndtr((x - np.mean(x)) / scale) * bins, 0, bins - 1))
    patterns = _templates(symbols.astype(np.int64), m)
    powers = bins ** np.arange(m - 1, -1, -1)
    codes = patterns @ powers
    _, counts = np.unique(codes, return_counts=True)
    probabilities = counts / counts.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def fuzzy_entropy(
    signal: np.ndarray, m: int = 2, r_factor: float = 0.2, exponent: float = 2.0
) -> float:
    """Compute FuzzyEn with an exponential similarity kernel."""
    x = np.asarray(signal, dtype=np.float64)
    if x.size <= m + 1:
        return float("nan")
    scale = np.std(x)
    if scale <= 0 or not np.isfinite(scale):
        return 0.0
    tolerance = r_factor * scale
    means = []
    for dimension in (m, m + 1):
        templates = _templates(x, dimension)
        templates = templates - templates.mean(axis=1, keepdims=True)
        distances = np.max(
            np.abs(templates[:, None, :] - templates[None, :, :]), axis=2
        )
        np.fill_diagonal(distances, np.inf)
        similarities = np.exp(-((distances / tolerance) ** exponent))
        means.append(float(np.mean(similarities[np.isfinite(similarities)])))
    if means[0] <= 0 or means[1] <= 0:
        return float("nan")
    return float(-math.log(means[1] / means[0]))


def shannon_entropy(signal: np.ndarray, bins: int = 32) -> float:
    """Compute amplitude-distribution Shannon entropy in nats."""
    x = np.asarray(signal, dtype=np.float64)
    if np.ptp(x) == 0:
        return 0.0
    probabilities = np.histogram(x, bins=bins, density=False)[0]
    probabilities = probabilities[probabilities > 0] / x.size
    return float(-np.sum(probabilities * np.log(probabilities)))


def lempel_ziv_complexity(signal: np.ndarray) -> float:
    """Compute normalized binary Lempel-Ziv complexity after median symbolization."""
    symbols = "".join(("1" if value > np.median(signal) else "0") for value in signal)
    n = len(symbols)
    if n < 2:
        return 0.0
    complexity = 1
    index = 0
    length = 1
    while index + length <= n:
        candidate = symbols[index:index + length]
        previous = symbols[:index]
        if candidate not in previous:
            complexity += 1
            index += length
            length = 1
        else:
            length += 1
    return float(complexity * math.log2(n) / n)


def calculate_entropies(
    signal: np.ndarray,
    max_samples: int = 512,
    bins: int = 10,
    m: int = 2,
    r_factor: float = 0.2,
) -> Dict[str, float]:
    """Calculate all report measures for one 1-D channel."""
    x = _clean_signal(signal, max_samples)
    return {
        "sample_entropy": sample_entropy(x, m=m, r_factor=r_factor),
        "dispersion_entropy": dispersion_entropy(x, bins=bins, m=m),
        "fuzzy_entropy": fuzzy_entropy(x, m=m, r_factor=r_factor),
        "shannon_entropy": shannon_entropy(x, bins=max(2, bins * 3)),
        "lz_complexity": lempel_ziv_complexity(x),
    }


def _iter_recordings(
    dataset: str, output_dir: Path
) -> Iterator[Tuple[str, int, str, np.ndarray]]:
    names: Iterable[str] = DATASETS if dataset == "all" else (dataset,)
    for name in names:
        signal_path = output_dir / f"{name}_signals.npz"
        metadata_path = output_dir / f"{name}_metadata.csv"
        if not signal_path.exists() or not metadata_path.exists():
            raise FileNotFoundError(f"missing processed dataset files for '{name}'")
        metadata = pd.read_csv(metadata_path).set_index("subject_id")
        with np.load(signal_path) as recordings:
            for subject_id in recordings.files:
                row = metadata.loc[subject_id]
                yield name, int(row["label"]), subject_id, recordings[subject_id]


def extract_features(
    dataset: str,
    output_dir: Path = OUTPUT_DIR,
    max_samples: int = 512,
    bins: int = 10,
    m: int = 2,
    r_factor: float = 0.2,
) -> pd.DataFrame:
    """Extract one feature row per recording and channel."""
    rows = []
    for dataset_name, label, subject_id, recording in _iter_recordings(dataset, output_dir):
        if recording.ndim != 2:
            raise ValueError(f"{subject_id} must have shape (channels, samples)")
        for channel, signal in enumerate(recording):
            features = calculate_entropies(signal, max_samples, bins, m, r_factor)
            rows.append({
                "dataset": dataset_name,
                "subject_id": subject_id,
                "label": label,
                "channel": channel,
                **features,
            })
        print(f"[{dataset_name}] {subject_id}: {recording.shape[0]} channels")
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=(*DATASETS, "all"), default="all")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "entropy_features.csv")
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--r-factor", type=float, default=0.2)
    args = parser.parse_args()
    if args.max_samples < args.m + 2:
        parser.error("--max-samples must be greater than --m + 1")
    if args.bins < 2 or args.m < 1 or args.r_factor <= 0:
        parser.error("--bins must be >= 2, --m >= 1, and --r-factor > 0")
    features = extract_features(
        args.dataset, args.output_dir, args.max_samples, args.bins, args.m, args.r_factor
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(args.output, index=False)
    print(f"Saved {len(features)} rows to {args.output}")


if __name__ == "__main__":
    main()

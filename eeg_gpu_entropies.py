"""GPU backend for pairwise entropy calculations."""
import importlib.util

import numpy as np


def detecter_cuda():
    if importlib.util.find_spec("cupy") is None:
        print("CuPy absent : calcul des entropies sur CPU.", flush=True)
        return "cpu", None

    import cupy as cp

    try:
        if cp.cuda.runtime.getDeviceCount() == 0:
            print("Aucun GPU CUDA visible : calcul des entropies sur CPU.", flush=True)
            return "cpu", None
        cp.exp(cp.asarray([1.0], dtype=cp.float64)).item()
        properties = cp.cuda.runtime.getDeviceProperties(0)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        info = {"device": name, "cupy": cp.__version__,
                "driver": cp.cuda.runtime.driverGetVersion(),
                "runtime": cp.cuda.runtime.runtimeGetVersion()}
    except cp.cuda.runtime.CUDARuntimeError as exc:
        print(f"GPU CUDA indisponible pour les entropies : {exc} ; utilisation du CPU.", flush=True)
        return "cpu", None
    print(f"Calcul pairwise des entropies sur GPU CUDA : {name} (CuPy {cp.__version__}).", flush=True)
    return "cuda", info


def compter_paires(x, m, r, xp, bloc=256):
    """Count matching m and m+1 templates once per unordered pair."""
    x = xp.asarray(x, dtype=xp.float64)
    templates = xp.lib.stride_tricks.sliding_window_view(x, m + 1)
    n = len(templates)
    bloc = max(1, min(bloc, 1_000_000 // max(n, 1)))
    columns = xp.arange(n)[None, :]
    matches_m = xp.zeros((), dtype=xp.int64)
    matches_next = xp.zeros((), dtype=xp.int64)
    for start in range(0, n, bloc):
        end = min(start + bloc, n)
        distance = xp.abs(templates[start:end, 0, None] - templates[None, :, 0])
        for k in range(1, m):
            xp.maximum(distance, xp.abs(templates[start:end, k, None] - templates[None, :, k]), out=distance)
        mask = (distance <= r) & (xp.arange(start, end)[:, None] < columns)
        matches_m += xp.count_nonzero(mask)
        xp.maximum(distance, xp.abs(templates[start:end, m, None] - templates[None, :, m]), out=distance)
        matches_next += xp.count_nonzero(mask & (distance <= r))
    return int(matches_m.item()), int(matches_next.item())


def fuzzy_entropie(x, m, r, n, xp, bloc=256):
    """Compute FuzzyEn using bounded pairwise blocks on the selected array backend."""
    x = xp.asarray(x, dtype=xp.float64)
    templates = xp.lib.stride_tricks.sliding_window_view(x, m + 1)
    size = len(templates)
    bloc = max(1, min(bloc, 1_000_000 // max(size, 1)))
    sums = []
    for dim in (m, m + 1):
        patterns = templates[:, :dim] - templates[:, :dim].mean(axis=1, keepdims=True)
        total = xp.zeros((), dtype=xp.float64)
        for start in range(0, size, bloc):
            end = min(start + bloc, size)
            distance = xp.abs(patterns[start:end, 0, None] - patterns[None, :, 0])
            for k in range(1, dim):
                xp.maximum(distance, xp.abs(patterns[start:end, k, None] - patterns[None, :, k]), out=distance)
            similarity = xp.exp(-(distance ** n) / r)
            rows = xp.arange(end - start)
            similarity[rows, rows + start] = 0.0
            total += similarity.sum()
        sums.append(float(total.item()) / (size * (size - 1)))
    return float(np.log(sums[0]) - np.log(sums[1])) if sums[0] > 0 and sums[1] > 0 else np.nan

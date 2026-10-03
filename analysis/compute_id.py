
import argparse
from pathlib import Path
import warnings

import numpy as np
from dadapy import IdEstimation


def gride_evolution(data, max_n1=280, upp_bound=100):
    """Return mean n2-neighbor distances and DADApy IDs for n2 = 2*n1.

    DADApy's public scaling method only evaluates powers of two. Its private
    single-scale helper preserves the R script's consecutive neighbor orders.
    Tested with DADApy 0.3.1; keep this dependency pinned when updating.
    DADApy's ratio filtering and finite-sample correction are retained, so the
    estimates need not match intRinsic's maximum-likelihood estimates exactly.
    """
    data = np.asarray(data, dtype=float)
    if data.ndim != 2 or data.shape[1] == 0 or not np.isfinite(data).all():
        raise ValueError("Features must be a finite, two-dimensional numeric array.")
    if not isinstance(max_n1, (int, np.integer)) or max_n1 < 1:
        raise ValueError("max_n1 must be a positive integer.")
    if not np.isfinite(upp_bound) or upp_bound <= 0:
        raise ValueError("upp_bound must be finite and positive.")

    # intRinsic removes duplicate feature vectors before finding neighbors.
    _, indices = np.unique(data, axis=0, return_index=True)
    if len(indices) < len(data):
        warnings.warn("Duplicate feature vectors were removed.", stacklevel=2)
        data = data[np.sort(indices)]
    if len(data) <= 2 * max_n1:
        raise ValueError(
            f"Need at least {2 * max_n1 + 1} distinct feature vectors for "
            f"max_n1={max_n1}; found {len(data)}."
        )

    estimator = IdEstimation(coordinates=data, maxk=2 * max_n1, n_jobs=1)
    estimator.compute_distances()
    distances = estimator.distances  # Column zero is the point itself.
    ids = np.empty(max_n1)
    for i, n1 in enumerate(range(1, max_n1 + 1)):
        ratios = distances[:, 2 * n1] / distances[:, n1]
        ids[i], _ = estimator._compute_id_gride_single_scale(
            d0=0.01,
            d1=min(data.shape[1], upp_bound) + 1,
            mus=ratios,
            n1=n1,
            n2=2 * n1,
            eps=1e-7,
        )
    return distances[:, 2::2].mean(axis=0), ids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--encoded-dir", type=Path,
        default=Path(__file__).resolve().parent / "encoded",
        help="Directory containing the space-delimited encoded feature CSVs.",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Output directory (defaults to --encoded-dir).",
    )
    parser.add_argument("--model", default="enca")
    parser.add_argument("--encoded-features", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 304)))
    parser.add_argument("--max-n1", type=int, default=280)
    parser.add_argument("--upp-bound", type=float, default=100)
    args = parser.parse_args()
    output_dir = args.output_dir or args.encoded_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    for seed in args.seeds:
        suffix = f"{args.model}_{args.encoded_features}_{seed}"
        input_path = args.encoded_dir / f"encoded_{suffix}.csv"
        # Skip the header and basin ID column; retain the raw feature scales.
        table = np.loadtxt(input_path, skiprows=1, ndmin=2)
        data = table[:, 1:]
        if data.shape[1] != args.encoded_features:
            raise ValueError(
                f"{input_path}: expected {args.encoded_features} feature columns, "
                f"found {data.shape[1]}."
            )
        print(f"Seed {seed}")
        print("Mean:", data.mean(axis=0))
        print("Standard deviation:", data.std(axis=0, ddof=1))
        distances, ids = gride_evolution(data, args.max_n1, args.upp_bound)
        output_path = output_dir / f"id_{suffix}.txt"
        np.savetxt(
            output_path, np.column_stack((distances, ids)),
            delimiter=",", header='"d","id"', comments="",
        )
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()

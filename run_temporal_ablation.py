"""
Ablation isolating which part of Quorum's temporal treatment actually helps:

  a. baseline        -- non-temporal features (no diff/RR), uniform random
                         feature selection ('e'), random-shuffle bucketing.
                         (Reproduces the existing non-temporal arm.)
  b. random_selection -- temporal features (diff+RR), uniform random feature
                         selection ('e') instead of the fixed stride ('f'),
                         fixed sequential (disjoint, non-jittered) temporal
                         bucketing. Isolates temporal *bucketing* alone, with
                         full random-selection diversity restored.
  c. jitter           -- same as (b), but temporal bucketing gets a random
                         per-iteration phase offset, so bucket membership
                         varies across ensemble iterations instead of being
                         the identical fixed partition every time.
  d. jitter_overlap    -- same as (c), plus 50% overlapping bucket windows,
                         so each point is compared against multiple
                         different local neighborhoods, further restoring
                         the ensemble-diversity mechanism Quorum's
                         statistics depend on.

All four run at the same (reduced) num_iterations so they're comparable to
each other and cheap enough to run in one sitting.

Usage: python run_temporal_ablation.py <num_qubits> <decoder_option>
           [--num_threads N] [--num_iterations N] [--only a,b,c,d]
"""
import argparse
import time

from Preprocessing.mitbih_preprocess import preprocess_mitbih_multi
from main import run_experiment, MITBIH_RECORDS


CONFIGS = {
    "a": dict(
        name="baseline_nontemporal",
        include_diff=False, include_rr=False,
        feature_strategy="e", temporal=False,
        bucket_overlap=0.0, bucket_jitter=False,
    ),
    "b": dict(
        name="temporal_random_selection",
        include_diff=True, include_rr=True,
        feature_strategy="e", temporal=True,
        bucket_overlap=0.0, bucket_jitter=False,
    ),
    "c": dict(
        name="temporal_jitter",
        include_diff=True, include_rr=True,
        feature_strategy="e", temporal=True,
        bucket_overlap=0.0, bucket_jitter=True,
    ),
    "d": dict(
        name="temporal_jitter_overlap",
        include_diff=True, include_rr=True,
        feature_strategy="e", temporal=True,
        bucket_overlap=0.5, bucket_jitter=True,
    ),
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Temporal bucketing/feature-selection ablation")
    parser.add_argument("num_qubits", type=int, help="Number of qubits to use")
    parser.add_argument("decoder_option", type=int, choices=[1, 2], help="Decoder option: 1 for Qiskit's .inverse(), 2 for manual decoder")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use")
    parser.add_argument("--num_iterations", type=int, default=100, help="Ensemble iterations per config")
    parser.add_argument("--only", type=str, default="a,b,c,d", help="Comma-separated subset of configs to run (a,b,c,d)")
    return parser.parse_args()


def main():
    args = parse_arguments()
    selected = [k.strip() for k in args.only.split(",") if k.strip()]

    total_start = time.time()

    # Preprocess each feature variant once and reuse across configs that need it.
    preprocessed_cache = {}

    for key in selected:
        cfg = CONFIGS[key]
        cache_key = (cfg["include_diff"], cfg["include_rr"])
        if cache_key not in preprocessed_cache:
            print(f"\n=== Preprocessing (include_diff={cfg['include_diff']}, include_rr={cfg['include_rr']}) ===")
            data, high_risk_indices, _ = preprocess_mitbih_multi(
                "Data/MIT_BIH", MITBIH_RECORDS,
                include_diff=cfg["include_diff"], include_rr=cfg["include_rr"],
            )
            print(f"Dataset size: {len(data)}, anomalies: {len(high_risk_indices)}")
            preprocessed_cache[cache_key] = (data, high_risk_indices)

        data, high_risk_indices = preprocessed_cache[cache_key]

        print(f"\n=== Config {key}: {cfg['name']} ===")
        print(f"feature_strategy={cfg['feature_strategy']!r} temporal={cfg['temporal']} "
              f"bucket_overlap={cfg['bucket_overlap']} bucket_jitter={cfg['bucket_jitter']} "
              f"num_iterations={args.num_iterations}")
        start = time.time()
        run_experiment(
            data, high_risk_indices, args.num_qubits, args.decoder_option, args.num_threads,
            f"results/ablation_{key}_{cfg['name']}.pkl",
            feature_strategy=cfg["feature_strategy"], temporal=cfg["temporal"],
            num_iterations=args.num_iterations,
            bucket_overlap=cfg["bucket_overlap"], bucket_jitter=cfg["bucket_jitter"],
        )
        print(f"Config {key} run time: {time.time() - start:.2f}s")

    print(f"\nTotal execution time: {time.time() - total_start:.2f}s")


if __name__ == "__main__":
    main()

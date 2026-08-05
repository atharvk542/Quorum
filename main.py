import argparse
import numpy as np
import pickle
import time
from concurrent.futures import ThreadPoolExecutor
import threading
# from Preprocessing.goldstein_uchida_preprocess import preprocess_goldstein_uchida
from Preprocessing.mitbih_preprocess import preprocess_mitbih_multi
from Preprocessing.ccpp_preprocess import preprocess_ccpp
from data_bucketing import perform_bucketing
from feature_selection import select_features, compute_decorrelation_weights
from Embedding.range_amplitude_enc import create_amplitude_encoding_circuit
# from Ansatzes.ry_cx_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
from Ansatzes.rx_rz_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
# from Ansatzes.ry_rz_ansatz import create_encoder_decoder_circuit, update_circuit_parameters
from swap_test_circuit import create_swap_test_circuit
from qiskit import transpile
from qiskit_aer import AerSimulator

from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError,
                             pauli_error, depolarizing_error, thermal_relaxation_error)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Data Preprocessing and Feature Selection")
    parser.add_argument("num_qubits", type=int, help="Number of qubits to use")
    parser.add_argument("decoder_option", type=int, choices=[1, 2], help="Decoder option: 1 for Qiskit's .inverse(), 2 for manual decoder")
    parser.add_argument("--num_threads", type=int, default=4, help="Number of threads to use")
    return parser.parse_args()


def _transpile_for_simulator(circuits, simulator):
    """
    Transpile against a simulator's basis_gates/coupling_map explicitly rather
    than passing backend=simulator directly. AerSimulator's Target does not
    populate per-qubit qargs when built from explicit basis_gates/coupling_map
    (as configure_noisy_simulator does), which crashes Qiskit's VF2Layout pass
    if handed straight to transpile() as a backend.
    """
    cfg = simulator.configuration()
    return transpile(circuits, basis_gates=cfg.basis_gates, coupling_map=cfg.coupling_map)


def create_realistic_noise_model(num_qubits):
    """
    Create a noise model matching IBM's Brisbane quantum computer specifications.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    NoiseModel: A Qiskit noise model matching Brisbane's error rates
    """
    noise_model = NoiseModel()
    
    # Brisbane specifications
    T1 = 230.42e3  # 230.42  in nanoseconds
    T2 = 143.41e3  # 143.41 in nanoseconds
    
    time_1q = 60 
    time_2q = 660
    time_readout = 1300
    
    # Error rates
    p_sx = 2.274e-4  
    p_cx = 2.903e-3  
    p_readout = 1.38e-2  
    
    # Add single-qubit gate errors
    for qubit in range(num_qubits):
        thermal_error_1q = thermal_relaxation_error(
            T1, T2, time_1q)
        
        depol_error_1q = depolarizing_error(p_sx, 1)
        
        gate_error_1q = thermal_error_1q.compose(depol_error_1q)
        
        noise_model.add_quantum_error(gate_error_1q, ["sx"], [qubit])
        
        readout_error = ReadoutError([[1 - p_readout, p_readout], 
                                    [p_readout, 1 - p_readout]])
        noise_model.add_readout_error(readout_error, [qubit])
        
        meas_thermal_error = thermal_relaxation_error(
            T1, T2, time_readout)
        noise_model.add_quantum_error(meas_thermal_error, ["measure"], [qubit])
    
    # Add two-qubit gate errors
    for q1 in range(num_qubits-1):
        for q2 in range(q1+1, num_qubits):
            thermal_error_q1 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_q2 = thermal_relaxation_error(
                T1, T2, time_2q)
            thermal_error_2q = thermal_error_q1.expand(thermal_error_q2)
            
            depol_error_2q = depolarizing_error(p_cx, 2)
            
            gate_error_2q = thermal_error_2q.compose(depol_error_2q)
            
            noise_model.add_quantum_error(gate_error_2q, ["cx"], [q1, q2])
    
    return noise_model

#mimics Brisbane noise model
def configure_noisy_simulator(num_qubits):
    """
    Configure the AerSimulator with IBM Brisbane noise settings.
    
    Args:
    num_qubits (int): Number of qubits in the system
    
    Returns:
    AerSimulator: Configured noisy simulator matching Brisbane specifications
    """
    noise_model = create_realistic_noise_model(num_qubits)
    
    basis_gates = ['sx', 'rz', 'cx', 'measure']  # Brisbane's basis gates
    simulator = AerSimulator(
        noise_model=noise_model,
        basis_gates=basis_gates,
        coupling_map=[[i, i+1] for i in range(num_qubits-1)] 
    )
    
    return simulator

def process_iteration(iteration, num_qubits, decoder_option, preprocessed_data, high_risk_indices, transpiled_swap_test, simulator, target_proportion, anomaly_likelihood_per_bucket, num_iterations, num_bucketruns, transpiled_ansatz_templates, feature_strategy='e', temporal=False, bucket_overlap=0.0, bucket_jitter=False, num_resamples=1, protected_features=None, decorrelation_weights=None):
    """
    Process a single iteration of the quantum autoencoder optimization.

    Args:
    iteration (int): The current iteration number.
    num_qubits (int): Number of qubits for a single amplitude encoding instance.
    decoder_option (int): Option for decoder circuit (1 or 2).
    preprocessed_data (pd.DataFrame): The preprocessed input data.
    high_risk_indices (list): Indices of high-risk data points.
    transpiled_swap_test (QuantumCircuit): The swap test circuit, already transpiled
        against `simulator` once up front (identical for every iteration/bucket).
    simulator (AerSimulator): The quantum circuit simulator.
    target_proportion (float): The target proportion for optimization.
    anomaly_likelihood_per_bucket (float): The anomaly likelihood per bucket.
    num_iterations (int): Total number of iterations.
    num_bucketruns (int): Number of random angle runs per bucket.
    transpiled_ansatz_templates (dict): compression_level -> (transpiled ansatz
        circuit with symbolic Parameters intact, encoder_params, decoder_params),
        precomputed once per compression level in run_experiment so binding random
        angles here is a cheap assign_parameters instead of a full re-transpile.
    feature_strategy (str): select_features strategy -- 'e' (uniform random,
        matches the paper) for the baseline run, 'corr_aware' (correlation-aware
        biased random selection, see feature_selection.compute_decorrelation_weights)
        for the changed run.
    temporal (bool): If True, bucket points in time order (create_temporal_buckets)
        instead of the random shuffle -- passed through to perform_bucketing.
    bucket_overlap (float): Passed through to perform_bucketing/create_temporal_buckets
        -- fraction of bucket_size consecutive temporal buckets share.
    bucket_jitter (bool): Passed through to perform_bucketing/create_temporal_buckets
        -- randomize the temporal partition's start offset each iteration.
    num_resamples (int): Passed through to select_features -- for strategy 'e',
        draws this many independent random candidate sets and votes across
        them instead of a single draw, to diversify which features get used.
    protected_features (list[str]): Passed through to select_features -- column
        names (e.g. RR_FEATURE_COLUMNS) always included and reserved from
        random/PCA selection, present in every iteration/bucket.
    decorrelation_weights (np.ndarray): Passed through to select_features --
        precomputed sampling weights for feature_strategy in
        ('decorrelated', 'corr_aware'), computed once in run_experiment
        rather than recomputed on every iteration.

    Returns:
    dict: Results of the iteration, including buckets, selected features, and optimization results.
    """
    # Calculate the compression level based on the iteration number
    compression_levels = num_qubits - 1
    iterations_per_level = num_iterations // compression_levels
    compression_level = (iteration // iterations_per_level) + 1
    compression_level = min(compression_level, num_qubits - 1)

    print(f"\nStarting iteration {iteration + 1} with compression_level {compression_level}")

    # Run the preprocessed data through the bucketing algorithm
    target_probability = anomaly_likelihood_per_bucket
    buckets, bucket_size = perform_bucketing(preprocessed_data, high_risk_indices, target_probability,
                                              temporal=temporal, overlap=bucket_overlap, jitter=bucket_jitter)

    print(f"Number of buckets created: {len(buckets)}")
    print(f"Bucket size: {bucket_size}")

    # Run feature selection on the data to select features for amplitude encoding
    selected_data, selected_features = select_features(preprocessed_data, num_qubits, strategy=feature_strategy, num_resamples=num_resamples, protected_features=protected_features, decorrelation_weights=decorrelation_weights)

    print(f"Number of features selected: {len(selected_features)}")
    print("Selected features:", selected_features)

    # Create amplitude encoding circuits for each datapoint+feature set, then
    # transpile them together in one call. Each Initialize gate still needs its
    # own decomposition (the amplitudes differ per datapoint), but transpiling
    # the raw (small) encoding circuits here -- instead of letting simulator.run()
    # implicitly transpile the much larger composed encoder+ansatz+swap circuit
    # on every call -- keeps that decomposition work off the ansatz/swap-test
    # portion, which is handled separately below.
    raw_circuits = {}
    for idx, row in selected_data.iterrows():
        raw_circuits[idx] = create_amplitude_encoding_circuit(row.values, num_qubits)

    indices = list(raw_circuits.keys())
    transpiled = _transpile_for_simulator([raw_circuits[idx] for idx in indices], simulator)
    amplitude_encoding_circuits = dict(zip(indices, transpiled))

    print(f"Created amplitude encoding circuits for {len(amplitude_encoding_circuits)} datapoints")

    # Use the pre-transpiled ansatz template for this compression level.
    # Binding random angles via assign_parameters below is a numeric
    # substitution on already-decomposed basis gates -- no re-synthesis.
    transpiled_ansatz, encoder_params, decoder_params = transpiled_ansatz_templates[compression_level]

    # Run random angle iterations for each bucket
    iteration_results = []
    for bucket_idx, bucket in enumerate(buckets):
        # print(f"\nProcessing bucket {bucket_idx + 1}/{len(buckets)}")
        final_results = []
        for _ in range(num_bucketruns):
            if decoder_option == 1:
                random_angles = np.random.uniform(0, 2*np.pi, len(encoder_params))
            else:
                random_angles = np.random.uniform(0, 2*np.pi, len(encoder_params) + len(decoder_params))

            random_ansatz = update_circuit_parameters(transpiled_ansatz, encoder_params, decoder_params, random_angles)

            # Batch every datapoint in this bucket into a single simulator.run()
            # call instead of one call per datapoint -- this cuts the number of
            # run() invocations (and their per-call dispatch/assembly overhead)
            # from one-per-datapoint down to one-per-bucket-per-bucketrun.
            full_circuits = [
                amplitude_encoding_circuits[idx].compose(random_ansatz).compose(transpiled_swap_test)
                for idx in bucket
            ]
            batch_result = simulator.run(full_circuits, shots=4096).result()
            for i in range(len(full_circuits)):
                counts = batch_result.get_counts(i)
                final_results.append(counts.get('0', 0) / 4096)

        average_proportion = np.mean(final_results)

        bucket_result = {
            'bucket_idx': bucket_idx,
            'final_results': final_results,
            'average_proportion': average_proportion,
            'encoder_params': encoder_params
        }
        # print("done")
        iteration_results.append(bucket_result)

    return {
        'iteration': iteration,
        'buckets': buckets,
        'selected_features': selected_features,
        'bucket_results': iteration_results,
        'high_risk_indices': high_risk_indices,
        'compression_level': compression_level
    }

def run_experiment(preprocessed_data, high_risk_indices, num_qubits, decoder_option, num_threads, output_path, feature_strategy='e', temporal=False, num_iterations=250, bucket_overlap=0.0, bucket_jitter=False, num_resamples=1, protected_features=None):
    num_bucketruns = 1
    target_proportion = 0.50
    anomaly_likelihood_per_bucket = 0.98

    # For the correlation-aware strategy, precompute the sampling weights
    # once here -- the underlying data doesn't change between iterations, so
    # recomputing the correlation matrix inside every one of the (hundreds
    # of) process_iteration calls would just repeat the same O(n_features^2)
    # work for no benefit. Computed only over the non-protected columns,
    # matching what _select_features_core actually draws from.
    decorrelation_weights = None
    if feature_strategy in ('decorrelated', 'corr_aware'):
        candidate_columns = [c for c in preprocessed_data.columns if c not in (protected_features or [])]
        decorrelation_weights = compute_decorrelation_weights(preprocessed_data[candidate_columns])

    swap_test = create_swap_test_circuit(num_qubits)
    simulator = AerSimulator()
    # simulator = configure_noisy_simulator(num_qubits)

    # Pre-transpile the swap test circuit once -- it's identical for every
    # iteration/bucket/datapoint, so leaving it to be re-transpiled inside the
    # hot loop (as part of the composed encoder+ansatz+swap circuit) would just
    # repeat the same structural decomposition on every simulator.run() call.
    transpiled_swap_test = _transpile_for_simulator(swap_test, simulator)

    # Pre-transpile one ansatz template per compression level (there are only
    # num_qubits - 1 of them across the whole run) with symbolic Parameters
    # left intact. Previously a freshly parameter-bound ansatz was rebuilt and
    # implicitly re-transpiled from scratch on every single bucket/bucketrun/
    # datapoint call -- the *values* differ each time but the gate structure
    # for a given compression level never does. Binding angles via
    # assign_parameters on an already-transpiled circuit is just a numeric
    # substitution on existing basis gates, not a re-synthesis.
    transpiled_ansatz_templates = {}
    for compression_level in range(1, num_qubits):
        ansatz, encoder_params, decoder_params = create_encoder_decoder_circuit(
            num_qubits, compression_level, decoder_option
        )
        transpiled_ansatz_templates[compression_level] = (
            _transpile_for_simulator(ansatz, simulator), encoder_params, decoder_params
        )

    all_results = []
    all_results_lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = []
        for iteration in range(num_iterations):
            future = executor.submit(
                process_iteration,
                iteration,
                num_qubits,
                decoder_option,
                preprocessed_data,
                high_risk_indices,
                transpiled_swap_test,
                simulator,
                target_proportion,
                anomaly_likelihood_per_bucket,
                num_iterations,
                num_bucketruns,
                transpiled_ansatz_templates,
                feature_strategy,
                temporal,
                bucket_overlap,
                bucket_jitter,
                num_resamples,
                protected_features,
                decorrelation_weights,
            )
            futures.append(future)

        for future in futures:
            result = future.result()
            with all_results_lock:
                all_results.append(result)

    print("\nAll iterations completed.")

    with open(output_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f"Results saved to {output_path}")


# Only using 2 records (100, 210) risked conclusions being driven by which
# two patients happened to get picked rather than by Quorum itself. Expanded
# to span the three patient/arrhythmia profiles MIT-BIH is normally grouped
# into (per PhysioNet's published record characteristics), so a handful of
# unusual patients can't dominate the result:
#   normal-heavy            : 100, 101, 103
#   PVC-heavy (ventricular) : 106, 119, 208
#   AF / supraventricular   : 201, 210, 221
# These IDs are the standard published categorization, NOT verified against
# what's actually present in this machine's Data/MIT_BIH -- listing that
# directory hung on a Kerberos/AFS password prompt rather than completing.
# preprocess_mitbih_multi already skips any missing record with a [WARNING]
# instead of failing, so a wrong/missing ID here degrades gracefully rather
# than crashing the run -- but double check the printed dataset size/anomaly
# count against expectations before trusting results from this list.
# NOTE: 9 records at num_iterations=500 (vs. the old 2 records @ 250) is a
# large jump in runtime from the ~1 day the 2-record/250-iteration run took --
# consider dropping num_iterations back down for the first pass with this list.
MITBIH_RECORDS = ["100", "106", "208", "210"]


# Fixed seed for both runs below, so the baseline and changed pipelines see
# the same sequence of random draws (bucket shuffles, feature-selection
# draws, ansatz angles) up to the point where their behavior actually
# diverges -- isolating the effect of the RR-protection/corr_aware changes
# themselves rather than confounding it with run-to-run RNG variance.
RANDOM_SEED = 42


def main():
    args = parse_arguments()
    num_qubits = args.num_qubits
    decoder_option = args.decoder_option
    num_threads = args.num_threads

    total_start = time.time()

    # --- Baseline: unchanged Quorum ---
    # No RR features, uniform random feature selection ('e', matching the
    # paper's Fig. 4 subsampling), random-shuffle bucketing.
    print("\n=== Baseline: unchanged Quorum ===")
    baseline_data, baseline_high_risk, _ = preprocess_mitbih_multi(
        "Data/MIT_BIH", MITBIH_RECORDS, include_diff=False, include_rr=False
    )
    print(f"Dataset size: {len(baseline_data)}, anomalies: {len(baseline_high_risk)}")
    np.random.seed(RANDOM_SEED)
    start = time.time()
    run_experiment(baseline_data, baseline_high_risk, num_qubits, decoder_option, num_threads,
                   'results/ensemble_res_baseline.pkl', feature_strategy='e', temporal=False,
                   num_iterations=200, num_resamples=5)
    print(f"Baseline run time: {time.time() - start:.2f}s")

    # --- Changed: protected RR features + correlation-aware selection ---
    # include_rr=True adds rr_prev/rr_next/rr_local_zscore (RR_FEATURE_COLUMNS)
    # to every beat, reserved via protected_features so they're always
    # included and never subject to random selection -- present in every
    # ensemble iteration/bucket, unlike the strategy-selected morphology
    # features. The remaining amplitude-encoding dimensions are drawn with
    # 'corr_aware' instead of uniform random ('e'), biasing away from
    # mutually redundant adjacent sample positions. Bucket size, iteration
    # count, num_resamples, and the record set/seed are all held fixed vs.
    # the baseline above so the two runs are directly comparable.
    print("\n=== Changed: protected RR features + corr_aware selection ===")
    changed_data, changed_high_risk, changed_meta = preprocess_mitbih_multi(
        "Data/MIT_BIH", MITBIH_RECORDS, include_diff=False, include_rr=True
    )
    protected_features = changed_meta.get("rr_feature_columns", [])
    print(f"Dataset size: {len(changed_data)}, anomalies: {len(changed_high_risk)}")
    print(f"Protected RR features: {protected_features}")
    np.random.seed(RANDOM_SEED)
    start = time.time()
    run_experiment(changed_data, changed_high_risk, num_qubits, decoder_option, num_threads,
                   'results/ensemble_res_changed.pkl', feature_strategy='corr_aware', temporal=False,
                   num_iterations=200, num_resamples=5, protected_features=protected_features)
    print(f"Changed run time: {time.time() - start:.2f}s")

    print(f"\nTotal execution time: {time.time() - total_start:.2f}s")

if __name__ == "__main__":
    main()
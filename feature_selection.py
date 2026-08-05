import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def perform_pca(data):
    """
    Perform PCA on the input data.

    Args:
    data (pd.DataFrame): Input data

    Returns:
    PCA: Fitted PCA object
    np.ndarray: Transformed data
    """
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)

    pca = PCA()
    pca_data = pca.fit_transform(scaled_data)

    return pca, pca_data


def compute_decorrelation_weights(data, neighbor_radius=2):
    """
    Assign each column a selection weight inversely related to how redundant
    it is with its immediate neighboring columns, for the 'decorrelated' /
    'corr_aware' strategy.

    Mirrors the EDA notebook's adjacent-position Pearson correlation analysis
    (np.corrcoef over the raw windows array), but instead of partitioning
    into hard high-correlation blocks, turns it into a continuous sampling
    bias: a position's weight is 1 minus its mean |correlation| with the
    neighbor_radius positions on each side, so smooth/redundant stretches of
    the beat window (e.g. adjacent samples on the same slope) are undersampled
    relative to relatively independent positions, while every position keeps
    a nonzero chance of being drawn.

    Args:
    data (pd.DataFrame): Candidate feature columns (e.g. beat-window samples)
        to compute pairwise correlation over. Should NOT include protected
        columns -- select_features strips those before calling this.
    neighbor_radius (int): Number of adjacent columns on each side averaged
        over when scoring a column's redundancy.

    Returns:
    np.ndarray: Selection weights, one per column of `data`, summing to 1.
    """
    values = data.values.astype(float)
    corr = np.corrcoef(values, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0)
    n = corr.shape[0]

    avg_neighbor_corr = np.zeros(n)
    for i in range(n):
        lo, hi = max(0, i - neighbor_radius), min(n, i + neighbor_radius + 1)
        neighbors = [j for j in range(lo, hi) if j != i]
        avg_neighbor_corr[i] = np.mean(np.abs(corr[i, neighbors])) if neighbors else 0.0

    redundancy = np.clip(avg_neighbor_corr, 0.0, 0.999)
    weights = (1.0 - redundancy) + 1e-3   # additive floor keeps every position selectable
    return weights / weights.sum()


def _select_zero_padded(data, num_features):
    """Shared by every strategy: if the budget exceeds the available columns,
    pad with zero-valued columns instead of selecting (nothing to select)."""
    selected_data = data.copy()
    num_zero_features = num_features - data.shape[1]
    for i in range(num_zero_features):
        selected_data[f'zero_feature_{i}'] = 0
    return selected_data, list(range(num_features))


def _select_features_core(data, num_features, strategy='a', num_resamples=1, decorrelation_weights=None):
    """
    Core strategy implementations, operating on a feature set that already
    excludes any protected columns -- see select_features for the reservation
    logic wrapping this.
    """
    original_num_features = data.shape[1]

    if num_features >= original_num_features:
        return _select_zero_padded(data, num_features)

    #uniform random selection of features
    if strategy == 'e':
        if num_resamples <= 1:
            selected_features = np.random.choice(data.columns, num_features, replace=False)
            return data[selected_features], selected_features.tolist()

        # Draw several independent uniform-random candidate sets and tally how
        # often each feature shows up. The final selection samples *proportional*
        # to those vote counts (not a deterministic top-N cutoff) -- a feature
        # drawn 5/5 times is 5x more likely than one drawn once, but taking the
        # top-N outright would let a handful of features that happened to be
        # drawn every time dominate almost every iteration across hundreds of
        # runs, collapsing exactly the diversity this was meant to add.
        vote_counts = pd.Series(0, index=data.columns)
        for _ in range(num_resamples):
            draw = np.random.choice(data.columns, num_features, replace=False)
            vote_counts[draw] += 1

        probabilities = vote_counts.values / vote_counts.values.sum()
        selected_features = np.random.choice(data.columns, num_features, replace=False, p=probabilities)
        return data[selected_features], selected_features.tolist()

    if strategy in ('decorrelated', 'corr_aware'):
        # Correlation-aware random selection (Change 2): biases uniform
        # random draws away from mutually redundant, highly-correlated
        # adjacent sample positions and toward relatively independent ones,
        # instead of treating every position as equally worth selecting.
        weights = decorrelation_weights
        if weights is None or len(weights) != data.shape[1]:
            weights = compute_decorrelation_weights(data)
        selected_indices = np.random.choice(data.shape[1], num_features, replace=False, p=weights)
        selected_features = data.columns[selected_indices]
        return data[selected_features], selected_indices.tolist()

    pca, pca_data = perform_pca(data)
    feature_importance = np.abs(pca.components_).sum(axis=0)

    #Select the top num_features features
    if strategy == 'a':
        selected_indices = feature_importance.argsort()[::-1][:num_features]
    #Select the bottom num_features features
    elif strategy == 'b':
        selected_indices = feature_importance.argsort()[::-1][-num_features:]
    #Select the top half and bottom half of the features
    elif strategy == 'c':
        num_top = num_features // 2
        num_bottom = num_features - num_top
        top_indices = feature_importance.argsort()[::-1][:num_top]
        bottom_indices = feature_importance.argsort()[::-1][-num_bottom:]
        selected_indices = np.concatenate([top_indices, bottom_indices])

    #weighted random selection based on feature importance
    elif strategy == 'd':
        selected_indices = np.random.choice(
            len(feature_importance),
            num_features,
            replace=False,
            p=feature_importance / feature_importance.sum()
        )
    elif strategy == 'f':
    # Temporal: stride evenly across the full ordered feature axis (with a
    # random per-iteration phase offset) instead of one narrow contiguous
    # block. A contiguous block of adjacent raw-signal/diff samples is
    # nearly redundant (a smooth ECG waveform barely changes between
    # neighboring samples) and, being short, almost never reaches the
    # RR-interval column appended at the end -- the single strongest
    # anomaly signal for arrhythmia detection. Striding across the whole
    # axis keeps selection order-aware while spreading coverage across the
    # waveform's full temporal extent, and always keeps the last column.
        stride = original_num_features / num_features
        offset = np.random.uniform(0, stride)
        selected_indices = [int(offset + i * stride) for i in range(num_features - 1)]
        selected_indices.append(original_num_features - 1)
        selected_features = data.columns[selected_indices]
        return data[selected_features], selected_indices
    else:
        raise ValueError("Invalid strategy. Choose 'a', 'b', 'c', 'd', 'e', 'f', 'decorrelated', or 'corr_aware'.")

    selected_features = data.columns[selected_indices]
    return data[selected_features], selected_indices.tolist()


def select_features(data, num_qubits, strategy='a', num_resamples=1, protected_features=None, decorrelation_weights=None):
    """
    Select features based on the specified strategy.

    Args:
    data (pd.DataFrame): Input data
    num_qubits (int): Number of qubits specified in main
    strategy (str): Feature selection strategy (a, b, c, d, e, f, decorrelated, or corr_aware)
    num_resamples (int): Only used by strategy 'e'. If > 1, draws this many
        independent uniform-random candidate sets and votes across them
        instead of taking a single draw -- see below.
    protected_features (list[str] | None): Column names that are always
        included and never subject to random/PCA selection (e.g. the RR
        interval columns produced by mitbih_preprocess -- see
        RR_FEATURE_COLUMNS). A fixed number of the 2**num_qubits - 1
        amplitude-encoding dimensions equal to len(protected_features) is
        reserved for these; the remaining dimensions are chosen by
        `strategy` from the rest of the columns only. Present in every call
        (unlike the strategy-selected columns, which vary call to call).
    decorrelation_weights (np.ndarray | None): Precomputed weights from
        compute_decorrelation_weights, for strategy 'decorrelated'/'corr_aware'.
        Passing this avoids recomputing the correlation matrix on every
        ensemble iteration (the underlying data doesn't change between
        iterations); if None, it's computed on the fly from the
        non-protected columns of `data`.

    Returns:
    pd.DataFrame: Data with selected features (including added 0-features if necessary)
    list: Indices of selected features
    """
    protected_features = [c for c in (protected_features or []) if c in data.columns]
    candidate_data = data.drop(columns=protected_features) if protected_features else data

    num_features = 2**num_qubits - 1
    num_reserved = len(protected_features)
    num_selectable = num_features - num_reserved
    if num_selectable < 0:
        raise ValueError(
            f"num_qubits={num_qubits} gives an amplitude-encoding budget of "
            f"{num_features} dimensions, which is smaller than the "
            f"{num_reserved} reserved protected feature(s) {protected_features}."
        )

    selected_candidate_data, selected_candidate_features = _select_features_core(
        candidate_data, num_selectable, strategy=strategy, num_resamples=num_resamples,
        decorrelation_weights=decorrelation_weights,
    )

    if not protected_features:
        return selected_candidate_data, selected_candidate_features

    selected_data = pd.concat([data[protected_features], selected_candidate_data], axis=1)
    selected_features = list(protected_features) + list(selected_candidate_features)
    return selected_data, selected_features

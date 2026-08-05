import os
import re
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# MIT-BIH annotation labels
# ---------------------------------------------------------------------------
# Beats labelled as anything other than 'N' (normal sinus) are treated as
# anomalies.  The '+' row in the annotations file is a rhythm-change marker,
# not a beat, so it is skipped entirely.
#
# Common anomaly labels in MIT-BIH:
#   V  - premature ventricular contraction
#   A  - atrial premature beat
#   a  - aberrated atrial premature beat
#   J  - nodal (junctional) premature beat
#   S  - supraventricular premature beat
#   F  - fusion of ventricular and normal beat
#   !  - ventricular flutter wave
#   E  - ventricular escape beat
#   j  - nodal escape beat
#   /  - paced beat
#   f  - fusion of paced and normal beat
#   x  - non-conducted P-wave (blocked APC)
#   Q  - unclassifiable beat
#   |  - isolated QRS-like artifact
# ---------------------------------------------------------------------------
NORMAL_LABEL = "N"
SKIP_LABELS  = {"+", "~", "[", "]", "!", "|"}   # rhythm/noise markers, not beats

# Number of beats on each side used to compute the local RR z-score's
# mean/std window (window length = 2 * RR_LOCAL_WINDOW_BEATS + 1).
RR_LOCAL_WINDOW_BEATS = 10

# Column names of the protected RR-derived features appended by
# extract_beat_windows when include_rr=True. These are meant to be reserved
# (always included, never subject to random selection) via select_features's
# protected_features argument -- see feature_selection.py.
RR_FEATURE_COLUMNS = ["rr_prev", "rr_next", "rr_local_zscore"]


def parse_annotations(ann_path: str) -> pd.DataFrame:
    """
    Parse a {number}annotations.txt file into a DataFrame.

    The file has a fixed-width header line followed by data rows:

        Time   Sample #  Type  Sub Chan  Num    Aux
        0:00.050       18     +    0    0    0   (N
        0:00.214       77     N    0    0    0
        ...

    Returns a DataFrame with columns: sample, label
    Only rows whose label is not in SKIP_LABELS are kept.
    """
    rows = []
    with open(ann_path, "r") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            # Header line starts with whitespace + "Time"
            if re.match(r"\s*Time", line):
                continue
            # Split on whitespace; columns are:
            # [time_str, sample_num, type, sub, chan, num, (optional aux)]
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                sample = int(parts[1])
            except ValueError:
                continue
            label = parts[2]
            if label in SKIP_LABELS:
                continue
            rows.append({"sample": sample, "label": label})

    return pd.DataFrame(rows)


def load_signal(csv_path: str) -> pd.DataFrame:
    """
    Load a {number}.csv file.

    Columns (after stripping quotes / whitespace from names):
        sample_num, MLII, V5

    Returns a DataFrame indexed by sample number with columns MLII and V5.
    """
    df = pd.read_csv(csv_path)
    # Strip surrounding quotes and whitespace from column names
    df.columns = [c.strip().strip("'").strip() for c in df.columns]
    # Rename to predictable names
    df = df.rename(columns={"sample #": "sample", "sample#": "sample"})
    df = df.set_index("sample")
    return df


def extract_beat_windows(
    signal_df: pd.DataFrame,
    ann_df: pd.DataFrame,
    window_half: int = 90,
    lead: str = "MLII",
    include_diff: bool = True,
    include_rr: bool = True,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Slice a fixed-length window around each annotated R-peak and build the
    feature matrix that Quorum's amplitude encoding expects.

    Parameters
    ----------
    signal_df   : DataFrame indexed by sample number, columns = lead names.
    ann_df      : DataFrame with columns 'sample' and 'label'.
    window_half : Number of samples on each side of the R-peak.
                  Total window length = 2 * window_half  (default 180 at 360 Hz
                  ≈ 500 ms, comfortably covering one full PQRST complex).
    lead        : Which ECG lead to use as the primary signal.
    include_diff: If True, append the first-difference (slope) of the window
                  as extra features.  Doubles feature count.
    include_rr  : If True, append three protected RR-derived columns (see
                  RR_FEATURE_COLUMNS): the preceding RR interval (rr_prev),
                  the following RR interval (rr_next), both in samples, and
                  a local RR z-score (rr_prev normalized against the mean/std
                  of RR intervals in a window of RR_LOCAL_WINDOW_BEATS beats
                  on each side within this record), which reflects local
                  rhythm irregularity rather than only an absolute interval.

    Returns
    -------
    features_df     : DataFrame where each row is one beat window. Morphology
                      column names are f0, f1, ..., fN; RR columns (when
                      include_rr=True) are named per RR_FEATURE_COLUMNS.
                      The DataFrame index is the sequential beat index (0-based).
    anomaly_indices : List of integer indices (into features_df) of anomalous beats.
    """
    signal = signal_df[lead].values
    n_samples = len(signal)

    beat_samples  = ann_df["sample"].values
    beat_labels   = ann_df["label"].values
    n_beats       = len(beat_samples)

    rows          = []
    beat_idx_list = []   # original beat indices that survived boundary checks

    for i, (s, lbl) in enumerate(zip(beat_samples, beat_labels)):
        start = s - window_half
        end   = s + window_half
        if start < 0 or end > n_samples:
            continue  # skip beats too close to the signal boundary

        window = signal[start:end].astype(float)

        if include_diff:
            diff = np.diff(window, prepend=window[0])  # same length as window
            features = np.concatenate([window, diff])
        else:
            features = window.copy()

        rows.append(features)
        beat_idx_list.append(i)

    # Build the DataFrame
    n_features = len(rows[0]) if rows else 0
    col_names = [f"f{j}" for j in range(n_features)]
    features_df = pd.DataFrame(rows, columns=col_names)

    if include_rr and n_beats > 0:
        # rr_prev/rr_next computed over *all* annotated beats in the record
        # (RR only depends on annotation sample positions, not the signal
        # window), then subset down to the beats that survived the boundary
        # check above. First/last beat in the record has no rr_prev/rr_next
        # respectively -- imputed with the record's median RR interval
        # rather than dropped, matching how the existing pipeline already
        # keeps boundary beats wherever possible.
        rr_prev = np.full(n_beats, np.nan)
        rr_next = np.full(n_beats, np.nan)
        if n_beats > 1:
            diffs = np.diff(beat_samples).astype(float)
            rr_prev[1:] = diffs
            rr_next[:-1] = diffs
            median_rr = float(np.median(diffs))
        else:
            median_rr = 0.0
        rr_prev = np.where(np.isnan(rr_prev), median_rr, rr_prev)
        rr_next = np.where(np.isnan(rr_next), median_rr, rr_next)

        # Local RR z-score: rr_prev normalized against the mean/std of RR
        # intervals in a local window of surrounding beats, so the feature
        # captures local rhythm irregularity rather than only an absolute
        # interval. Falls back to 0 where the local std is 0 or undefined
        # (e.g. near the very start/end of a record).
        rr_prev_series = pd.Series(rr_prev)
        window_size = 2 * RR_LOCAL_WINDOW_BEATS + 1
        local_mean = rr_prev_series.rolling(window=window_size, center=True, min_periods=3).mean()
        local_std  = rr_prev_series.rolling(window=window_size, center=True, min_periods=3).std()
        local_std  = local_std.replace(0, np.nan)
        rr_local_zscore = ((rr_prev_series - local_mean) / local_std).fillna(0.0).values

        features_df["rr_prev"]         = rr_prev[beat_idx_list]
        features_df["rr_next"]         = rr_next[beat_idx_list]
        features_df["rr_local_zscore"] = rr_local_zscore[beat_idx_list]

    # Map original beat indices → row indices in features_df
    orig_to_row = {orig: row for row, orig in enumerate(beat_idx_list)}

    anomaly_indices = [
        orig_to_row[i]
        for i, lbl in enumerate(beat_labels)
        if lbl != NORMAL_LABEL and i in orig_to_row
    ]

    return features_df, sorted(anomaly_indices)


def normalize_for_quorum(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Scale each feature column to [0, 1] using per-column min-max normalization,
    then divide by sqrt(n_features) so the squared-norm of any row is at most 1.

    This matches Quorum's requirement that sum(x_i^2) <= 1, which is needed for
    valid amplitude encoding.
    """
    df = features_df.copy()
    col_min = df.min(axis=0)
    col_max = df.max(axis=0)
    col_range = col_max - col_min
    col_range[col_range == 0] = 1.0   # avoid divide-by-zero for constant columns

    df = (df - col_min) / col_range   # each value in [0, 1]

    # Scale so max possible L2 norm per row is 1
    n_features = df.shape[1]
    df = df / np.sqrt(n_features)

    return df


def _extract_record_features(
    data_dir: str,
    record_id: str,
    window_half: int,
    lead: str,
    include_diff: bool,
    include_rr: bool,
) -> tuple[pd.DataFrame, list[int], dict]:
    """
    Load one record and return its RAW (un-normalized) beat-window feature
    matrix, anomaly indices, and metadata. Shared by preprocess_mitbih and
    preprocess_mitbih_multi so normalization can be applied once, at the
    right point, by each caller.
    """
    csv_path = os.path.join(data_dir, f"{record_id}.csv")
    ann_path = os.path.join(data_dir, f"{record_id}annotations.txt")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not os.path.exists(ann_path):
        raise FileNotFoundError(f"Annotations not found: {ann_path}")

    signal_df = load_signal(csv_path)
    ann_df    = parse_annotations(ann_path)

    features_df, anomaly_indices = extract_beat_windows(
        signal_df,
        ann_df,
        window_half=window_half,
        lead=lead,
        include_diff=include_diff,
        include_rr=include_rr,
    )

    label_counts = (
        ann_df[ann_df["label"] != NORMAL_LABEL]["label"]
        .value_counts()
        .to_dict()
    )

    metadata = {
        "record_id":           record_id,
        "n_beats":             len(features_df),
        "n_anomalies":         len(anomaly_indices),
        "n_features":          features_df.shape[1],
        "window_half":         window_half,
        "lead":                lead,
        "include_diff":        include_diff,
        "include_rr":          include_rr,
        "rr_feature_columns":  list(RR_FEATURE_COLUMNS) if include_rr else [],
        "anomaly_label_counts": label_counts,
    }

    return features_df, anomaly_indices, metadata


def preprocess_mitbih(
    data_dir: str,
    record_id: str | int,
    window_half: int = 90,
    lead: str = "MLII",
    include_diff: bool = True,
    include_rr: bool = True,
) -> tuple[pd.DataFrame, list[int], dict]:
    """
    Main entry point.  Matches the return signature of preprocess_goldstein_uchida:

        preprocessed_data, high_risk_indices, metadata

    Parameters
    ----------
    data_dir    : Path to the folder containing the MIT-BIH files,
                  e.g. "Data/MIT_BIH".
    record_id   : Patient number as an int or string, e.g. 100 or "100".
    window_half : Half-width of the beat window in samples (default 90).
    lead        : ECG lead to use ("MLII" or "V5").
    include_diff: Append first-difference features.
    include_rr  : Append preceding RR interval as a feature.

    Returns
    -------
    preprocessed_data : pd.DataFrame  — normalized beat-window feature matrix.
    high_risk_indices : list[int]     — row indices of anomalous beats.
    metadata          : dict          — record_id, n_beats, n_anomalies,
                                        n_features, window_half, lead,
                                        anomaly_label_counts.
    """
    record_id = str(record_id)
    features_df, anomaly_indices, metadata = _extract_record_features(
        data_dir, record_id, window_half, lead, include_diff, include_rr
    )

    preprocessed_data = normalize_for_quorum(features_df)
    metadata["n_beats"] = len(preprocessed_data)

    return preprocessed_data, anomaly_indices, metadata


# ---------------------------------------------------------------------------
# Multi-record loader — convenience wrapper for loading several patients at
# once and stacking them into a single DataFrame (useful for cross-patient
# experiments).
# ---------------------------------------------------------------------------
def preprocess_mitbih_multi(
    data_dir: str,
    record_ids: list,
    window_half: int = 90,
    lead: str = "MLII",
    include_diff: bool = True,
    include_rr: bool = True,
) -> tuple[pd.DataFrame, list[int], dict]:
    """
    Load multiple MIT-BIH records and concatenate them into one feature matrix.
    The anomaly_indices returned are global row indices into the combined DataFrame.

    Normalization is applied once, on the concatenated raw feature matrix,
    not per-record. Each record's own min/max ranges differ (e.g. baseline
    ECG amplitude/variance), so normalizing per-record before concatenating
    would rescale each record onto its own [0, 1] range independently --
    injecting a record-identity signal that swamps genuine cross-patient
    anomaly signal once buckets mix records together (as the non-temporal,
    randomly-shuffled bucketing does).

    Useful for training / evaluating Quorum across many patients in one run.
    """
    all_features    = []
    all_anomalies   = []
    all_meta        = []
    row_offset      = 0

    for rid in record_ids:
        rid = str(rid)
        try:
            df, anoms, meta = _extract_record_features(
                data_dir, rid, window_half, lead, include_diff, include_rr
            )
        except FileNotFoundError as e:
            print(f"[WARNING] Skipping record {rid}: {e}")
            continue

        all_features.append(df)
        all_anomalies.extend([a + row_offset for a in anoms])
        all_meta.append(meta)
        row_offset += len(df)

    if not all_features:
        raise RuntimeError("No records were successfully loaded.")

    combined_df = normalize_for_quorum(pd.concat(all_features, ignore_index=True))
    combined_meta = {
        "records":            [m["record_id"] for m in all_meta],
        "n_beats":            sum(m["n_beats"] for m in all_meta),
        "n_anomalies":        len(all_anomalies),
        "n_features":         combined_df.shape[1],
        "window_half":        window_half,
        "lead":               lead,
        "rr_feature_columns": all_meta[0]["rr_feature_columns"] if all_meta else [],
        "per_record":         all_meta,
    }

    return combined_df, all_anomalies, combined_meta


# ---------------------------------------------------------------------------
# Quick smoke-test — run directly to verify your Data/MIT_BIH layout
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    data_dir  = sys.argv[1] if len(sys.argv) > 1 else "Data/MIT_BIH"
    record_id = sys.argv[2] if len(sys.argv) > 2 else "100"

    print(f"Loading record {record_id} from {data_dir} ...")
    df, anomalies, meta = preprocess_mitbih(data_dir, record_id)

    print(f"\nRecord        : {meta['record_id']}")
    print(f"Total beats   : {meta['n_beats']}")
    print(f"Anomalies     : {meta['n_anomalies']}")
    print(f"Features/beat : {meta['n_features']}")
    print(f"Anomaly types : {meta['anomaly_label_counts']}")
    print(f"\nFirst 3 anomaly indices : {anomalies[:3]}")
    print(f"\nFeature matrix head:\n{df.head(3)}")
    print(f"\nL2 norm of first row    : {np.linalg.norm(df.iloc[0].values):.4f}")
    print(f"Max L2 norm across rows : {np.linalg.norm(df.values, axis=1).max():.4f}")
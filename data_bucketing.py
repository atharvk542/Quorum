import numpy as np
from typing import List, Tuple

def estimate_bucket_size(p_anomaly: float, target_probability: float, tolerance: float = 1e-6, max_iterations: int = 1000) -> int:
    """
    Estimate the bucket size needed to achieve a target probability
    of containing at least one anomaly.
    
    Args:
    p_anomaly (float): Probability of an anomaly in a single sample
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    tolerance (float): Acceptable error in the probability (default: 1e-6)
    max_iterations (int): Maximum number of iterations (default: 1000)
    
    Returns:
    int: Estimated bucket size needed
    """
    n = 1
    
    for _ in range(max_iterations):
        current_P = 1 - (1 - p_anomaly) ** n
        
        if abs(current_P - target_probability) < tolerance:
            return n
        
        if current_P < target_probability:
            n += 1
        else:
            return n  #round up to ensure we meet/exceed the target probability
    
    raise ValueError(f"Failed to converge after {max_iterations} iterations")

def create_data_buckets(num_datapoints: int, num_anomalies: int, target_probability: float = 0.5) -> List[List[int]]:
    """
    Create buckets of random indices for the dataset.
    
    Args:
    num_datapoints (int): Total number of datapoints in the dataset
    num_anomalies (int): Total number of anomalies in the dataset
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    
    Returns:
    List[List[int]]: List of buckets, where each bucket is a list of indices
    """
    p_anomaly = num_anomalies / num_datapoints
    bucket_size = estimate_bucket_size(p_anomaly, target_probability)
    
    all_indices = list(range(num_datapoints))
    np.random.shuffle(all_indices)
    
    buckets = [all_indices[i:i+bucket_size] for i in range(0, num_datapoints, bucket_size)]
    
    return buckets

def create_temporal_buckets(num_datapoints, num_anomalies,
                            target_probability=0.5,
                            rr_groups=None,
                            overlap=0.0,
                            jitter=False):
    """
    Args:
    overlap (float): Fraction of bucket_size that consecutive buckets share,
        in [0, 1). 0 reproduces the original disjoint partition (each point
        in exactly one bucket); e.g. 0.5 means each point falls into ~2
        overlapping buckets, so it gets compared against multiple different
        local neighborhoods within the same iteration.
    jitter (bool): If True, shift the partition/window start by a random
        offset in [0, stride) each call, so which points share a bucket
        varies across ensemble iterations instead of being the exact same
        fixed partition every time (the original behavior gave every point
        identical bucket-mates on every single iteration -- zero ensemble
        diversity on the bucketing axis).
    """
    p_anomaly = num_anomalies / num_datapoints
    bucket_size = estimate_bucket_size(p_anomaly, target_probability)
    if rr_groups is not None:
        # sort indices within each RR regime, then slice
        ordered = []
        for group in rr_groups:
            ordered.extend(sorted(group))
    else:
        ordered = list(range(num_datapoints))  # time order, no shuffle

    stride = max(1, int(round(bucket_size * (1 - overlap))))
    offset = int(np.random.randint(0, stride)) if jitter else 0

    buckets = []
    for start in range(-offset, num_datapoints, stride):
        window = ordered[max(start, 0):start + bucket_size]
        if window:
            buckets.append(window)
    return buckets

def perform_bucketing(preprocessed_data: np.ndarray, high_risk_indices: List[int], target_probability: float = 0.5, temporal: bool = False, overlap: float = 0.0, jitter: bool = False) -> Tuple[List[List[int]], int]:
    """
    Perform the bucketing process on the preprocessed data.

    Args:
    preprocessed_data (np.ndarray): The preprocessed dataset
    high_risk_indices (List[int]): List of indices of high-risk (anomalous) datapoints
    target_probability (float): Desired probability of having at least one anomaly in a bucket
    temporal (bool): If True, group points in time order via create_temporal_buckets
        (so temporally-clustered anomalies land in the same bucket) instead of the
        random shuffle used by create_data_buckets.
    overlap (float): Passed through to create_temporal_buckets when temporal=True.
    jitter (bool): Passed through to create_temporal_buckets when temporal=True.

    Returns:
    Tuple[List[List[int]], int]: A tuple containing the list of buckets and the bucket size
    """
    num_datapoints = len(preprocessed_data)
    num_anomalies = len(high_risk_indices)

    if temporal:
        buckets = create_temporal_buckets(num_datapoints, num_anomalies, target_probability,
                                           overlap=overlap, jitter=jitter)
    else:
        buckets = create_data_buckets(num_datapoints, num_anomalies, target_probability)
    bucket_size = len(buckets[0])  # all buckets except possibly the last one will have this size

    print(f"Created {len(buckets)} buckets with a target size of {bucket_size} datapoints each.")
    print(f"Probability of at least one anomaly in each bucket: {target_probability}")

    return buckets, bucket_size
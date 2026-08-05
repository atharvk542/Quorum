"""
LSTM autoencoder baseline for MIT-BIH beat anomaly detection.

Classical reference point for Quorum's quantum autoencoder pipeline: a
sequence-to-sequence LSTM autoencoder is trained to reconstruct the raw
per-beat ECG window using ONLY normal ('N') beats (the standard semi-
supervised anomaly-detection setup), then reconstruction error is used as
the anomaly score for beats the model never trained on (held-out normal
beats + every anomalous beat). Scoring/thresholding mirrors
analyze_results.py's threshold_metrics/detection_curve so the numbers are
directly comparable to the Quorum results in results/*.pkl.

Usage: python lstm_baseline.py
"""
import os
import pickle

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, average_precision_score

from Preprocessing.mitbih_preprocess import preprocess_mitbih_multi

# Same record list used by main.py's Quorum runs, so this is an
# apples-to-apples comparison on identical data.
MITBIH_RECORDS = ["100", "101", "103", "106", "119", "208", "201", "210", "221"]

DATA_DIR = "Data/MIT_BIH"
RESULTS_PATH = "results/lstm_baseline.pkl"
SEED = 42

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LSTMAutoencoder(nn.Module):
    """Seq2seq LSTM autoencoder with a linear bottleneck, reconstructing a
    fixed-length univariate ECG beat window."""

    def __init__(self, seq_len, hidden_size=64, latent_dim=16, num_layers=1):
        super().__init__()
        self.seq_len = seq_len

        self.encoder_lstm = nn.LSTM(1, hidden_size, num_layers=num_layers, batch_first=True)
        self.to_latent = nn.Linear(hidden_size, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_size)
        self.decoder_lstm = nn.LSTM(hidden_size, hidden_size, num_layers=num_layers, batch_first=True)
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, 1)
        _, (h_n, _) = self.encoder_lstm(x)
        latent = self.to_latent(h_n[-1])                 # (batch, latent_dim)

        decoder_input = self.from_latent(latent)          # (batch, hidden_size)
        decoder_input = decoder_input.unsqueeze(1).repeat(1, self.seq_len, 1)

        decoded, _ = self.decoder_lstm(decoder_input)
        reconstruction = self.output_layer(decoded)        # (batch, seq_len, 1)
        return reconstruction


def load_dataset():
    preprocessed_data, high_risk_indices, meta = preprocess_mitbih_multi(
        DATA_DIR, MITBIH_RECORDS, include_diff=False, include_rr=False
    )
    X = preprocessed_data.values.astype(np.float32)
    y = np.zeros(len(X), dtype=np.int64)
    y[high_risk_indices] = 1
    print(f"Dataset size: {len(X)}, anomalies: {y.sum()}, seq_len: {X.shape[1]}")
    return X, y


def split_normal_train_mixed_test(X, y, train_frac=0.7, seed=SEED):
    """Semi-supervised split: train ONLY on a fraction of normal beats;
    test on the held-out normal beats plus every anomalous beat."""
    rng = np.random.RandomState(seed)
    normal_idx = np.where(y == 0)[0]
    rng.shuffle(normal_idx)

    n_train = int(len(normal_idx) * train_frac)
    train_idx = normal_idx[:n_train]
    test_idx = np.concatenate([normal_idx[n_train:], np.where(y == 1)[0]])
    rng.shuffle(test_idx)

    return X[train_idx], X[test_idx], y[test_idx]


def train_autoencoder(X_train, seq_len, epochs=30, batch_size=64, lr=1e-3):
    torch.manual_seed(SEED)
    model = LSTMAutoencoder(seq_len).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    train_tensor = torch.from_numpy(X_train).unsqueeze(-1)  # (N, seq_len, 1)
    loader = DataLoader(TensorDataset(train_tensor), batch_size=batch_size, shuffle=True)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(DEVICE)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= len(train_tensor)
        print(f"Epoch {epoch + 1}/{epochs}  reconstruction MSE: {epoch_loss:.6f}")

    return model


@torch.no_grad()
def reconstruction_scores(model, X, batch_size=256):
    model.eval()
    tensor = torch.from_numpy(X).unsqueeze(-1)
    scores = []
    for start in range(0, len(tensor), batch_size):
        batch = tensor[start:start + batch_size].to(DEVICE)
        recon = model(batch)
        per_sample_mse = ((recon - batch) ** 2).mean(dim=(1, 2))
        scores.append(per_sample_mse.cpu().numpy())
    return np.concatenate(scores)


def threshold_metrics(scores, y_true):
    """Flag the top-N highest-scoring points as anomalous, N = true anomaly
    count -- matches analyze_results.py's threshold_metrics so results are
    directly comparable to the Quorum runs."""
    n_anomalies = int(y_true.sum())
    order = np.argsort(-scores)
    y_pred = np.zeros_like(y_true)
    y_pred[order[:n_anomalies]] = 1

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy = (tp + tn) / len(y_true)
    return {"precision": precision, "recall": recall, "f1": f1, "accuracy": accuracy}


def detection_curve(scores, y_true):
    """Fraction of dataset examined (sorted by score desc) vs. fraction of
    true anomalies captured -- matches analyze_results.py's Fig. 9 curve."""
    order = np.argsort(-scores)
    sorted_true = y_true[order]
    cum_anomalies = np.cumsum(sorted_true)
    total_anomalies = y_true.sum()
    frac_dataset = np.arange(1, len(y_true) + 1) / len(y_true)
    frac_anomalies = cum_anomalies / total_anomalies
    return frac_dataset, frac_anomalies


def main():
    np.random.seed(SEED)
    os.makedirs("results", exist_ok=True)

    X, y = load_dataset()
    X_train, X_test, y_test = split_normal_train_mixed_test(X, y)
    print(f"Train (normal only): {len(X_train)}  Test: {len(X_test)} "
          f"({y_test.sum()} anomalies, {len(X_test) - y_test.sum()} normal)")

    model = train_autoencoder(X_train, seq_len=X.shape[1])

    scores = reconstruction_scores(model, X_test)
    metrics = threshold_metrics(scores, y_test)
    roc_auc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)
    curve = detection_curve(scores, y_test)

    print("\n=== LSTM Autoencoder Baseline ===")
    print(f"{'Precision':>12}{'Recall':>10}{'F1':>10}{'Accuracy':>10}{'ROC-AUC':>10}{'PR-AUC':>10}")
    print(f"{metrics['precision']:>12.3f}{metrics['recall']:>10.3f}{metrics['f1']:>10.3f}"
          f"{metrics['accuracy']:>10.3f}{roc_auc:>10.3f}{pr_auc:>10.3f}")

    frac_dataset, frac_anomalies = curve
    idx_10pct = int(0.10 * len(frac_dataset)) - 1
    print(f"\n{frac_anomalies[idx_10pct]:.1%} of anomalies detected within top 10% of scored points")

    with open(RESULTS_PATH, "wb") as f:
        pickle.dump({
            "scores": scores,
            "y_true": y_test,
            "metrics": metrics,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "curve": curve,
        }, f)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()

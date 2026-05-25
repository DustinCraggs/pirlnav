import pandas as pd
import numpy as np
import strictfire
from typing import Optional


def analyze_metrics(
    csv_path: str,
    start_epoch: int = 10,
    end_epoch: int = 20,
    max_epoch: Optional[int] = None,
    smooth_k: int = 5,
):
    """
    Analyzes validation metrics across different architectures from a CSV.

    Args:
        csv_path: Path to the metrics CSV.
        start_epoch: Start of the range for mean/stdev calculation.
        end_epoch: End of the range for mean/stdev calculation.
        max_epoch: Maximum epoch to consider (ignores data beyond this).
        smooth_k: Window size for the moving average.
    """
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter columns:
    filter_cols = ["_MIN", "_MAX", "_step"]
    metric_cols = [
        col for col in df.columns if not any(s in col for s in filter_cols)
    ]
    df = df[metric_cols]

    # Identify the epoch column (assumed first) and architecture columns
    epoch_col = df.columns[0]
    arch_cols = df.columns[1:]

    # 1. Apply max_epoch filter
    if max_epoch is not None:
        df = df[df[epoch_col] <= max_epoch]

    print(
        f"--- Analysis (Up to Epoch {max_epoch if max_epoch else df[epoch_col].max()}) ---"
    )

    results = []

    for arch in arch_cols:
        # Drop NaNs for this specific architecture to handle different lengths
        arch_data = df[[epoch_col, arch]].dropna()

        # 2. Windowed Mean and Stdev (e.g., Epoch 10-20)
        range_mask = (arch_data[epoch_col] >= start_epoch) & (
            arch_data[epoch_col] <= end_epoch
        )
        range_subset = arch_data.loc[range_mask, arch]

        if not range_subset.empty:
            r_mean = range_subset.mean()
            r_std = range_subset.std()
        else:
            r_mean, r_std = np.nan, np.nan

        # 3. Smoothed Mean
        # We calculate the rolling mean and take the last valid value as the 'current' performance
        smoothed_series = arch_data[arch].rolling(window=smooth_k, min_periods=1).mean()
        latest_smoothed = (
            smoothed_series.iloc[-1] if not smoothed_series.empty else np.nan
        )
        peak_smoothed = smoothed_series.max()

        results.append(
            {
                "Architecture": arch,
                f"Mean({start_epoch}-{end_epoch})": r_mean,
                f"Std({start_epoch}-{end_epoch})": r_std,
                f"Smoothed_Last(k={smooth_k})": latest_smoothed,
                f"Smoothed_Peak": peak_smoothed,
                "Total_Epochs": (
                    int(arch_data[epoch_col].max()) if not arch_data.empty else 0
                ),
            }
        )

    # Display results as a formatted table
    results_df = pd.DataFrame(results).set_index("Architecture")
    print(results_df.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    strictfire.StrictFire(analyze_metrics)

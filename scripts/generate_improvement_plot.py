import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from strictfire import StrictFire


def main(plot_type: str = "bar", hide_yaxis: bool = False):
    """Generates performance improvement charts.

    Args:
        plot_type: Choose between:
          - 'bar': Percentage improvement bar chart.
          - 'dumbbell': Absolute value dumbbell transition plot.
          - 'connected_bars': Paired absolute bars joined by improvement slope lines.
        hide_yaxis: If True, hides the Y-axis on the bar chart and appends
          "Improvement" directly to the bar annotations.
    """

    COLOR_BC = (20 / 255, 60 / 255, 120 / 255)  # Dark Blue
    COLOR_OE = (100 / 255, 30 / 255, 140 / 255)  # Deep Purple

    # COLOR_SR_BAR = (140 / 255, 60 / 255, 60 / 255)
    # COLOR_SPL_BAR = (200 / 255, 120 / 255, 120 / 255)

    COLOR_SR_BAR = (45 / 255, 110 / 255, 80 / 255)
    COLOR_SPL_BAR = (95 / 255, 170 / 255, 105 / 255)

    # Annotation Colors for percentage gains
    COLOR_ANNOT_CONN = (34 / 255, 139 / 255, 34 / 255)  # Forest Green
    # COLOR_ANNOT_BAR = (0 / 255, 0 / 255, 0 / 255)  # Darker Green
    COLOR_ANNOT_BAR = (45 / 255, 110 / 255, 80 / 255)
    # COLOR_ANNOT_BAR = (30 / 255, 120 / 255, 30 / 255)  # Darker Green

    # 1. Raw absolute values for Baseline and New runs
    # 5% Dataset Size
    baseline_sr_5 = 22.6
    new_sr_5 = 40.4

    baseline_spl_5 = 9.8635
    new_spl_5 = 17.725

    # 10% Dataset Size
    baseline_sr_10 = 34.55
    new_sr_10 = 48.9

    baseline_spl_10 = 14.509
    new_spl_10 = 22.114

    # Helper function to compute percentage improvement
    def calc_improvement(baseline, new):
        return ((new - baseline) / baseline) * 100

    # Build a structured DataFrame containing both absolute and computed values
    data = [
        {
            "Dataset Size": "5% Dataset",
            "Metric": "SR",
            "BC": baseline_sr_5,
            "ObjectExplore": new_sr_5,
        },
        {
            "Dataset Size": "5% Dataset",
            "Metric": "SPL",
            "BC": baseline_spl_5,
            "ObjectExplore": new_spl_5,
        },
        {
            "Dataset Size": "10% dataset",
            "Metric": "SR",
            "BC": baseline_sr_10,
            "ObjectExplore": new_sr_10,
        },
        {
            "Dataset Size": "10% dataset",
            "Metric": "SPL",
            "BC": baseline_spl_10,
            "ObjectExplore": new_spl_10,
        },
    ]

    df = pd.DataFrame(data)
    df["Improvement (%)"] = df.apply(
        lambda r: calc_improvement(r["BC"], r["ObjectExplore"]), axis=1
    )

    if plot_type.lower() == "bar":
        # --- ORIGINAL PLOT REBUILT IN PURE MATPLOTLIB ---
        sns.set_theme(style="whitegrid", font_scale=1.8)
        fig, ax = plt.subplots(figsize=(11, 7))

        datasets = df["Dataset Size"].unique()
        x_indices = np.arange(len(datasets))

        bar_width = 0.35
        gap = 0.1
        offset = (bar_width + gap) / 2

        df_sr = df[df["Metric"] == "SR"]
        df_spl = df[df["Metric"] == "SPL"]

        bars_sr = ax.bar(
            x_indices - offset,
            df_sr["Improvement (%)"],
            width=bar_width,
            color=COLOR_SR_BAR,
            label="SR",
        )

        bars_spl = ax.bar(
            x_indices + offset,
            df_spl["Improvement (%)"],
            width=bar_width,
            color=COLOR_SPL_BAR,
            label="SPL",
        )

        # Axis styling & layout
        ax.set_xticks(x_indices)
        clean_labels = [d.replace("dataset", "Dataset") for d in datasets]
        ax.set_xticklabels(clean_labels, fontweight="bold")

        # Increased upper limit slightly to comfortably clear larger multi-line fonts
        ax.set_ylim(0, 100)

        # Handle Y-Axis display logic based on CLI toggle
        if hide_yaxis:
            ax.yaxis.set_visible(False)
            ax.yaxis.grid(False)  # Turn off horizontal grid lines for clean canvas
            ax.spines["left"].set_visible(False)
        else:
            ax.set_ylabel("Percentage Improvement", fontweight="bold")
            ax.yaxis.set_major_formatter(
                plt.FuncFormatter(lambda y, _: f"{int(y)}%")
            )

        # ax.set_title(
        #     "Performance Improvement of ObjectExplore over BC", pad=25, fontweight="bold"
        # )

        # Annotate bars dynamically with increased font size and "+" prefix
        for bar in bars_sr:
            height = bar.get_height()
            label_text = (
                f"+{round(height, 1)}%\nImprovement" if hide_yaxis else f"+{round(height, 1)}%"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                label_text,
                ha="center",
                va="bottom",
                fontsize=20,  # Increased font size
                fontweight="bold",
                color=COLOR_ANNOT_BAR,
            )

        for bar in bars_spl:
            height = bar.get_height()
            label_text = (
                f"+{round(height, 1)}%\nImprovement" if hide_yaxis else f"+{round(height, 1)}%"
            )
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height + 2,
                label_text,
                ha="center",
                va="bottom",
                fontsize=20,  # Increased font size
                fontweight="bold",
                color=COLOR_ANNOT_BAR,
            )

        ax.legend(frameon=True, loc="upper right")

        output_filename = "percentage_improvement_bar_chart.png"
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        print(f"Success: Generated perfectly spaced bar chart -> {output_filename}")

    elif plot_type.lower() == "dumbbell":
        # --- DUMBBELL PLOT ---
        sns.set_theme(style="whitegrid", font_scale=1.8)
        fig, ax = plt.subplots(figsize=(12, 7))

        df["Label"] = df["Dataset Size"] + " (" + df["Metric"] + ")"
        y_pos = np.arange(len(df))

        ax.hlines(
            y=y_pos,
            xmin=df["BC"],
            xmax=df["ObjectExplore"],
            color="gray",
            alpha=0.4,
            linewidth=5,
        )
        ax.scatter(df["BC"], y_pos, color=COLOR_BC, s=250, label="BC", zorder=5)
        ax.scatter(
            df["ObjectExplore"],
            y_pos,
            color=COLOR_OE,
            s=250,
            label="ObjectExplore",
            zorder=5,
        )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(df["Label"])
        ax.invert_yaxis()
        ax.set_xlabel("Absolute Performance Score")
        ax.set_title(
            "Absolute Performance Transitions & Gains", pad=25, fontweight="bold"
        )

        for i, row in df.iterrows():
            mid_x = (row["BC"] + row["ObjectExplore"]) / 2
            imp = row["Improvement (%)"]
            ax.text(
                mid_x,
                i - 0.2,
                f"+{imp:.1f}%",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=COLOR_ANNOT_CONN,
            )

        ax.legend(loc="lower right", frameon=True)
        output_filename = "performance_dumbbell_plot.png"
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        print(f"Success: Generated dumbbell plot -> {output_filename}")

    elif plot_type.lower() in ["connected_bars", "connected"]:
        # --- PAIRED ABSOLUTE BARS WITH CONNECTING SLOPE ---
        sns.set_theme(style="whitegrid", font_scale=1.6)
        fig, ax = plt.subplots(figsize=(13, 7.5))

        group_centers = [0, 1.5, 4.0, 5.5]
        bar_width = 0.35
        paired_offset = bar_width / 2 + 0.01

        bc_labeled = False
        oe_labeled = False

        for idx, row in df.iterrows():
            center = group_centers[idx]
            x_bc = center - paired_offset
            x_oe = center + paired_offset
            y_bc = row["BC"]
            y_oe = row["ObjectExplore"]
            imp = row["Improvement (%)"]

            ax.bar(
                x_bc,
                y_bc,
                width=bar_width,
                color=COLOR_BC,
                label="BC" if not bc_labeled else "",
            )
            ax.bar(
                x_oe,
                y_oe,
                width=bar_width,
                color=COLOR_OE,
                label="ObjectExplore" if not oe_labeled else "",
            )
            bc_labeled = True
            oe_labeled = True

            ax.plot(
                [x_bc, x_oe],
                [y_bc, y_oe],
                color="black",
                linestyle="--",
                linewidth=2,
                marker="o",
                markersize=5,
                zorder=4,
            )

            ax.text(
                x_bc - 0.05,
                y_bc + 1,
                f"{y_bc:.1f}",
                ha="right",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#444",
            )
            ax.text(
                x_oe + 0.05,
                y_oe + 1,
                f"{y_oe:.1f}",
                ha="left",
                va="bottom",
                fontsize=13,
                fontweight="bold",
                color="#444",
            )

            ax.text(
                center,
                y_oe + 4.5,
                f"+{imp:.1f}%",
                ha="center",
                va="bottom",
                fontsize=16,
                fontweight="bold",
                color=COLOR_ANNOT_CONN,
            )

        ax.set_ylim(0, 65)
        ax.set_ylabel("Absolute Performance Score", fontweight="bold")
        ax.set_title(
            "Absolute Performance & Relative Trajectory", pad=20, fontweight="bold"
        )

        custom_labels = [
            f"{row['Metric']}\n({row['Dataset Size'].replace('dataset', 'Dataset')})"
            for _, row in df.iterrows()
        ]
        ax.set_xticks(group_centers)
        ax.set_xticklabels(custom_labels, fontweight="bold")

        ax.legend(loc="upper left", frameon=True)

        output_filename = "performance_connected_bars.png"
        plt.savefig(output_filename, bbox_inches="tight", dpi=300)
        print(f"Success: Generated connected absolute bars -> {output_filename}")

    else:
        print(
            f"Error: Invalid plot_type '{plot_type}'. Choose 'bar', 'dumbbell', or 'connected_bars'."
        )


if __name__ == "__main__":
    StrictFire(main)
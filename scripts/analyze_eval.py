import json
import strictfire
from collections import defaultdict
import numpy as np
import sys
from typing import Dict, Any, Tuple


def load_stats(path: str) -> Dict[str, Any]:
    """Loads JSON stats from a file."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        sys.exit(1)


def get_per_file_stats(
    data: Dict[str, Any],
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    """Computes overall success rate and per-category stats."""
    total_episodes = len(data)
    if total_episodes == 0:
        return 0.0, {}

    success_count = 0
    cat_data = defaultdict(list)

    for key, val in data.items():
        success = val.get("success", 0.0)
        spl = val.get("spl", 0.0)
        category = val.get("ep_info", {}).get("object_category", "unknown")

        success_count += success
        cat_data[category].append({"success": success, "spl": spl})

    overall_sr = success_count / total_episodes

    cat_stats = {}
    for cat, items in cat_data.items():
        count = len(items)
        sr = sum(i["success"] for i in items) / count
        mean_spl = sum(i["spl"] for i in items) / count
        cat_stats[cat] = {"sr": sr, "spl": mean_spl, "count": count}

    return overall_sr, cat_stats


def fill_missing_categories(data1: Dict[str, Any], data2: Dict[str, Any]):
    """Fills in missing object_category by cross-referencing files."""
    for key in data1:
        cat1 = data1[key].get("ep_info", {}).get("object_category")
        cat2 = data2.get(key, {}).get("ep_info", {}).get("object_category")

        final_cat = cat1 or cat2 or "unknown"

        # Ensure ep_info and object_category exist in both
        if "ep_info" not in data1[key]:
            data1[key]["ep_info"] = {}
        data1[key]["ep_info"]["object_category"] = final_cat

        if key in data2:
            if "ep_info" not in data2[key]:
                data2[key]["ep_info"] = {}
            data2[key]["ep_info"]["object_category"] = final_cat


def main(path1: str, path2: str):
    """
    Analyzes and compares two evaluation JSON files.

    Args:
        path1: Path to the first JSON file.
        path2: Path to the second JSON file.
    """
    data1 = load_stats(path1)
    data2 = load_stats(path2)

    keys1 = set(data1.keys())
    keys2 = set(data2.keys())

    # Check for 2000 unique episodes
    if len(keys1) != 2000:
        print(f"Error: {path1} has {len(keys1)} episodes, expected 2000.")
        sys.exit(1)
    if len(keys2) != 2000:
        print(f"Error: {path2} has {len(keys2)} episodes, expected 2000.")
        sys.exit(1)

    if keys1 != keys2:
        print(f"Error: The sets of episode keys in both files do not match.")
        # Optional: print symmetric difference if needed for debugging
        sys.exit(1)

    # Fill missing categories before computing stats
    fill_missing_categories(data1, data2)

    # Individual stats
    sr1, cat1 = get_per_file_stats(data1)
    sr2, cat2 = get_per_file_stats(data2)

    sr1 *= 100
    sr2 *= 100

    print(f"=== Results for: {path1} ===")
    print(f"Overall Success Rate: {sr1:.2f}")
    print(f"{'Category':<20} {'SR':<10} {'Mean SPL':<10} {'Count':<10}")
    print("-" * 50)
    for cat in sorted(cat1.keys()):
        s = cat1[cat]
        print(
            f"{cat:<20} {s['sr'] * 100:<10.2f} {s['spl'] * 100:<10.2f} {s['count']:<10}"
        )
    print("\n")

    print(f"=== Results for: {path2} ===")
    print(f"Overall Success Rate: {sr2:.2f}")
    print(f"{'Category':<20} {'SR':<10} {'Mean SPL':<10} {'Count':<10}")
    print("-" * 50)
    for cat in sorted(cat2.keys()):
        s = cat2[cat]
        print(
            f"{cat:<20} {s['sr'] * 100:<10.2f} {s['spl'] * 100:<10.2f} {s['count']:<10}"
        )
    print("\n")

    # Intersection analysis
    common_success_keys = [
        k for k in keys1 if data1[k]["success"] == 1.0 and data2[k]["success"] == 1.0
    ]

    print(f"=== Intersection Analysis ===")
    print(f"Number of episodes successful in BOTH: {len(common_success_keys)}")

    if common_success_keys:
        spls1 = [data1[k]["spl"] for k in common_success_keys]
        spls2 = [data2[k]["spl"] for k in common_success_keys]

        mean_spl1 = np.mean(spls1) * 100
        mean_spl2 = np.mean(spls2) * 100

        print(f"Mean SPL (File 1) over intersection: {mean_spl1:.2f}")
        print(f"Mean SPL (File 2) over intersection: {mean_spl2:.2f}")
        print(f"SPL Delta: {mean_spl2 - mean_spl1:.2f}")
    else:
        print("No episodes succeeded in both files.")


if __name__ == "__main__":
    strictfire.StrictFire(main)

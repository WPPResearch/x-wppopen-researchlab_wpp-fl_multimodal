import os
from collections import defaultdict

import pandas as pd


def main():
    """Calculates the mean frequency of each relationship type across all subgraph CSVs."""
    # --- Path and File Checks ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    subgraphs_dir = os.path.join(project_root, "subgraphs")
    all_edges_path = os.path.join(project_root, "all_edges.csv")

    if not os.path.exists(all_edges_path):
        print(f"Error: 'all_edges.csv' not found at '{all_edges_path}'.")
        return

    if not os.path.exists(subgraphs_dir):
        print(f"Error: Subgraphs directory not found at '{subgraphs_dir}'.")
        print("Please run 'poetry run generate-subgraphs <number>' first.")
        return

    subgraph_files = [
        os.path.join(subgraphs_dir, f)
        for f in os.listdir(subgraphs_dir)
        if f.endswith(".csv") and f.split(".")[0].isdigit()
    ]
    if not subgraph_files:
        print(f"No valid subgraph CSVs found in {subgraphs_dir}")
        return

    total_rel_counts = defaultdict(float)
    file_count = 0

    for file_path in subgraph_files:
        try:
            df = pd.read_csv(file_path)
            if "relationship" in df.columns:
                rel_counts = df["relationship"].value_counts()
                for rel, count in rel_counts.items():
                    total_rel_counts[rel] += count
                file_count += 1
            else:
                pass  # Silently skip files without the 'relationship' column

        except pd.errors.EmptyDataError:
            pass  # Silently skip empty files
        except Exception as e:
            print(f"An unexpected error occurred while processing {os.path.basename(file_path)}: {e}")

    if file_count > 0:
        mean_rel_counts = {rel: total / file_count for rel, total in total_rel_counts.items()}

        sorted_leaderboard = sorted(mean_rel_counts.items(), key=lambda item: item[1], reverse=True)

        print("Mean Relationship Frequency Leaderboard:")
        for rel, mean_freq in sorted_leaderboard:
            print(f"{rel}: {mean_freq:.2f}")

        num_items = len(sorted_leaderboard)
        if num_items >= 4:
            print("\n--- Leaderboard Summary ---")
            print("\nHighest 2:")
            for rel, mean_freq in sorted_leaderboard[:2]:
                print(f"{rel}: {mean_freq:.2f}")

            mid_index = num_items // 2
            middle_4 = sorted_leaderboard[max(0, mid_index - 2) : min(num_items, mid_index + 2)]
            print("\nMiddle 4:")
            for rel, mean_freq in middle_4:
                print(f"{rel}: {mean_freq:.2f}")

            print("\nLowest 2:")
            for rel, mean_freq in sorted_leaderboard[-2:]:
                print(f"{rel}: {mean_freq:.2f}")


if __name__ == "__main__":
    main()

import csv
import os
import re
import statistics


def get_subgraph_stats():
    """Calculates and returns statistics from generated subgraphs."""
    # --- Configuration ---
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    subgraphs_dir = os.path.join(project_root, "subgraphs")
    original_csv_path = os.path.join(project_root, "all_edges.csv")

    # 1. Discover all valid relationship types
    rel_types = set()
    if not os.path.exists(original_csv_path):
        print(f"Error: Original CSV not found at {original_csv_path}")
        return None

    with open(original_csv_path, "r") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # skip header
        except StopIteration:
            print("Error: Original CSV is empty.")
            return None
        for row in reader:
            if len(row) >= 3 and "_to_" in row[2]:
                parts = row[2].strip().split("_to_")
                if len(parts) == 2:
                    rel_types.add(tuple(parts))

    # 2. Find and process subgraphs
    if not os.path.exists(subgraphs_dir):
        print(f"Error: Subgraphs directory not found at '{subgraphs_dir}'. Please run generate-subgraphs.")
        return None

    subgraph_files = [f for f in os.listdir(subgraphs_dir) if re.match(r"^\d+\.csv$", f)]
    if not subgraph_files:
        print(f"Error: No numeric subgraphs found in {subgraphs_dir}")
        return None

    all_pos_ratios, all_neg_ratios, all_mis_ratios, all_node_counts = [], [], [], []
    print(f"Processing {len(subgraph_files)} subgraphs...")

    for f_name in sorted(subgraph_files):
        # ... (rest of the processing logic remains the same) ...
        path = os.path.join(subgraphs_dir, f_name)
        nodes_by_role = {}
        actual_pos, actual_neg = 0, 0
        with open(path, "r") as f:
            reader = csv.reader(f)
            try:
                next(reader)
            except StopIteration:
                continue
            for row in reader:
                if len(row) < 4:
                    continue
                src, tgt, rel, sign = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                if sign == "+":
                    actual_pos += 1
                elif sign == "-":
                    actual_neg += 1
                if "_to_" in rel:
                    r1, r2 = rel.split("_to_")
                    nodes_by_role.setdefault(r1, set()).add(src)
                    nodes_by_role.setdefault(r2, set()).add(tgt)

        unique_nodes = set().union(*nodes_by_role.values())
        all_node_counts.append(len(unique_nodes))

        possible = sum(len(nodes_by_role.get(r1, set())) * len(nodes_by_role.get(r2, set())) for r1, r2 in rel_types)
        if possible > 0:
            all_pos_ratios.append(actual_pos / possible)
            all_neg_ratios.append(actual_neg / possible)
            all_mis_ratios.append((possible - actual_pos - actual_neg) / possible)

    if not all_pos_ratios:
        print("No valid data in subgraphs to calculate statistics.")
        return None

    # 4. Compute and return final stats
    return {
        "pos_avg": statistics.mean(all_pos_ratios) * 100,
        "neg_avg": statistics.mean(all_neg_ratios) * 100,
        "mis_avg": statistics.mean(all_mis_ratios) * 100,
        "pos_std": (statistics.stdev(all_pos_ratios) * 100) if len(all_pos_ratios) > 1 else 0.0,
        "neg_std": (statistics.stdev(all_neg_ratios) * 100) if len(all_neg_ratios) > 1 else 0.0,
        "mis_std": (statistics.stdev(all_mis_ratios) * 100) if len(all_mis_ratios) > 1 else 0.0,
        "nodes_avg": statistics.mean(all_node_counts),
        "nodes_std": statistics.stdev(all_node_counts) if len(all_node_counts) > 1 else 0.0,
        "subgraph_count": len(subgraph_files),
    }


def main():
    """Calculates and prints statistics from generated subgraphs."""
    stats = get_subgraph_stats()
    if stats:
        print("-" * 30)
        print(f"pos_avg: {stats['pos_avg']:.2f}% (std: {stats['pos_std']:.2f}%)")
        print(f"neg_avg: {stats['neg_avg']:.2f}% (std: {stats['neg_std']:.2f}%)")
        print(f"mis_avg: {stats['mis_avg']:.2f}% (std: {stats['mis_std']:.2f}%)")
        print(f"nodes_avg: {stats['nodes_avg']:.2f} (std: {stats['nodes_std']:.2f})")
        print("-" * 30)


if __name__ == "__main__":
    main()

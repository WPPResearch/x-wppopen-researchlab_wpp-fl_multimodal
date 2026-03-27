import csv
import os
import random
import shutil
import sys


def main():
    # --- Configuration & Setup ---

    # Get the number of subgraphs from the command-line argument
    num_subgraphs = 2  # Default value
    if len(sys.argv) > 1:
        try:
            num_subgraphs = int(sys.argv[1])
            if num_subgraphs < 1:
                raise ValueError("Number of subgraphs must be at least 1.")
        except ValueError as e:
            print(f"Error: Invalid argument for number of subgraphs. {e}")
            sys.exit(1)

    # Define the number of nodes to sample for each entity type
    ranges = {"brand": (3, 5), "content": (3, 5), "platform": (1, 1), "geo": (2, 5), "audience": (3, 5)}

    # Use relative paths for portability
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # This assumes the script is in a 'scripts' subdirectory
    base_dir = project_root
    out_dir = os.path.join(base_dir, "subgraphs")

    original_csv_path = os.path.join(base_dir, "all_edges.csv")

    # Clean and recreate the output directory
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    # --- Main Logic ---

    # Dictionary to hold sets of nodes for each entity type
    nodes_by_type = {k: set() for k in ranges.keys()}

    # 1. READ ORIGINAL GRAPH FROM DISK
    print(f"Reading from {original_csv_path}...")
    all_edges = []
    try:
        with open(original_csv_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if len(row) >= 4:
                    src, tgt, rel, sign = row[0].strip(), row[1].strip(), row[2].strip(), row[3].strip()
                    all_edges.append((src, tgt, rel, sign))

                    # Populate potential nodes for sampling based on relationship
                    if "_to_" in rel:
                        parts = rel.split("_to_")
                        if len(parts) == 2:
                            t1, t2 = parts[0], parts[1]
                            if t1 in nodes_by_type:
                                nodes_by_type[t1].add(src)
                            if t2 in nodes_by_type:
                                nodes_by_type[t2].add(tgt)
    except FileNotFoundError:
        print(f"Error: Input file not found at '{original_csv_path}'. Please ensure the file exists.")
        sys.exit(1)

    # Convert sets to lists for sampling
    node_lists_by_type = {k: list(v) for k, v in nodes_by_type.items()}
    for k, v in node_lists_by_type.items():
        print(f"Found {len(v)} nodes for type {k}")

    # Define the output filenames
    subgraph_names = [f"{i + 1:02d}.csv" for i in range(num_subgraphs)]

    # 2. GENERATE RANDOMIZED SUBGRAPHS
    # Run the sampling and mapping logic for each required output file
    for out_name in subgraph_names:
        selected_by_type = {t: set() for t in ranges}

        # Randomly select a target count for each node type based on configured min/max range
        for t, (min_val, max_val) in ranges.items():
            count = random.randint(min_val, max_val)
            # Sample 'count' number of nodes from our population
            sampled = random.sample(node_lists_by_type[t], min(count, len(node_lists_by_type[t])))
            # Update the typed selection of nodes for this current subgraph
            selected_by_type[t].update(sampled)

        out_csv = os.path.join(out_dir, out_name)
        edge_count = 0
        unique_nodes_in_subgraph = set()

        # 3. WRITE SUBGRAPH TO DISK
        with open(out_csv, "w", newline="") as f_out:
            writer = csv.writer(f_out)
            writer.writerow(["source", "target", "relationship", "sign"])

            for src, tgt, rel, sign in all_edges:
                if "_to_" in rel:
                    parts = rel.split("_to_")
                    if len(parts) == 2:
                        t1, t2 = parts[0], parts[1]
                        # Only keep edge if BOTH nodes are in their respective role's selected pool
                        if t1 in selected_by_type and t2 in selected_by_type:
                            if src in selected_by_type[t1] and tgt in selected_by_type[t2]:
                                writer.writerow([src, tgt, rel, sign])
                                edge_count += 1
                                unique_nodes_in_subgraph.add(src)
                                unique_nodes_in_subgraph.add(tgt)

        print(
            f"FINISHED: Generated {out_csv} with "
            f"{len(unique_nodes_in_subgraph)} unique nodes total and {edge_count} edges."
        )


if __name__ == "__main__":
    main()

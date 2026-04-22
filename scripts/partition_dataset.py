import json
import os
import random
import shutil
from collections import defaultdict

import click
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger


# ----------------------------
# Partitioning
# ----------------------------
def homogeneous_partitioning(targets: np.ndarray, num_clients: int):
    """
    Homogeneous split: each client gets (as equal as possible) per-label samples.
    Works with string labels.
    """
    indices_by_class = defaultdict(list)
    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    for label in indices_by_class:
        np.random.shuffle(indices_by_class[label])

    client_partitions = [[] for _ in range(num_clients)]
    class_splits = {label: np.array_split(indices, num_clients) for label, indices in indices_by_class.items()}

    for _, splits in class_splits.items():
        for client_id in range(num_clients):
            client_partitions[client_id].extend(splits[client_id].tolist())

    for client_id in range(num_clients):
        random.shuffle(client_partitions[client_id])

    return client_partitions


def heterogeneous_partitioning(targets: np.ndarray, num_clients: int, alpha: float):
    """
    Heterogeneous split: Dirichlet label-skew across clients (non-IID).
    Works with string labels.
    """
    unique_labels = np.unique(targets)
    indices_by_class = defaultdict(list)

    for idx, label in enumerate(targets):
        indices_by_class[label].append(idx)

    for label in indices_by_class:
        np.random.shuffle(indices_by_class[label])

    client_partitions = [[] for _ in range(num_clients)]

    for label in unique_labels:
        class_indices = indices_by_class[label]
        n = len(class_indices)

        proportions = np.random.dirichlet([alpha] * num_clients)
        split_counts = (proportions * n).astype(int)
        split_counts[-1] += n - np.sum(split_counts)  # rounding fix

        start = 0
        for cid, count in enumerate(split_counts):
            if count > 0:
                client_partitions[cid].extend(class_indices[start : start + count])
            start += count

    for cid in range(num_clients):
        random.shuffle(client_partitions[cid])

    return client_partitions


# ----------------------------
# Filesystem helpers
# ----------------------------
def clear_directory(dir_path: str):
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    os.makedirs(dir_path, exist_ok=True)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# ----------------------------
# JSON array load/save
# ----------------------------
def load_json_array(path: str):
    """
    Loads rows from:
      1) JSON array: [ {...}, {...}, ... ]
      2) Single JSON object: {...}   -> returns [obj]
      3) Concatenated JSON objects / JSONL / NDJSON:
           {...}\n{...}\n...
         or even:
           {...} {...} {...}

    Returns: List[dict]
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found: {path}")

    # First try standard JSON (array or object)
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, list):
            if not all(isinstance(r, dict) for r in obj):
                bad = next((type(r) for r in obj if not isinstance(r, dict)), None)
                raise ValueError(f"{path} JSON array must contain objects (dict). Found element of type {bad}.")
            return obj

        if isinstance(obj, dict):
            # Single row stored as one object
            return [obj]

        raise ValueError(f"{path} must be a JSON array (list) or object (dict). Got: {type(obj)}")

    except json.JSONDecodeError:
        # Fall back to parsing concatenated JSON objects
        pass

    # Parse concatenated objects using raw_decode repeatedly
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    decoder = json.JSONDecoder()
    rows = []
    i = 0
    n = len(text)

    while True:
        # Skip whitespace
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break

        obj, next_i = decoder.raw_decode(text, i)

        if not isinstance(obj, dict):
            raise ValueError(
                f"{path}: expected JSON object at position {i}, got {type(obj)}. "
                "If your file is a JSON array, it should start with '['."
            )

        rows.append(obj)
        i = next_i

    if not rows:
        raise ValueError(f"{path} appears empty or could not be parsed into any JSON objects.")

    return rows


def save_json_array(path: str, rows):
    ensure_dir(os.path.dirname(path) or ".")
    if path.endswith(".jsonl"):
        with open(path, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    else:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f, ensure_ascii=False, indent=2)


# ----------------------------
# Heatmap
# ----------------------------
def save_partition_heatmap(image_path: str, targets: np.ndarray, num_clients: int, client_partitions):
    targets = np.asarray(targets)
    unique_labels = np.unique(targets)

    label_to_col = {lab: i for i, lab in enumerate(unique_labels)}
    counts = np.zeros((num_clients, len(unique_labels)), dtype=int)

    for cid in range(num_clients):
        client_labels = targets[client_partitions[cid]]
        labs, cts = np.unique(client_labels, return_counts=True)
        for lab, ct in zip(labs, cts):
            counts[cid, label_to_col[lab]] = int(ct)

    df = pd.DataFrame(counts, columns=[str(ul) for ul in unique_labels])

    plt.figure(figsize=(max(12, len(unique_labels) * 0.6), 8))
    sns.heatmap(
        df.T,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar_kws={"label": "Label Count"},
        linewidths=0.5,
        square=False,
    )
    plt.title("Label Distribution per Client")
    plt.xlabel("Client ID")
    plt.ylabel("Label")
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    plt.savefig(image_path, bbox_inches="tight")
    plt.close()

    logger.info(f"Heatmap saved under {image_path}")


@click.command()
@click.option("--version", required=True, help='Subfolder under datasets/, e.g. "V16"')
@click.option("--datasets_dir", default="./datasets", show_default=True)
@click.option("--train_file", default="train.json", show_default=True)
@click.option("--test_file", default="test.json", show_default=True)
@click.option("--label_col", default="label", show_default=True)
@click.option("--num_clients", default=10, show_default=True)
@click.option("--split", default="homogeneous", show_default=True, help="homogeneous|heterogeneous")
@click.option("--alpha", default=1.0, show_default=True, help="Dirichlet alpha (heterogeneous only)")
@click.option("--seed", default=42, show_default=True)
def main(
    version: str,
    datasets_dir: str,
    train_file: str,
    test_file: str,
    label_col: str,
    num_clients: int,
    split: str,
    alpha: float,
    seed: int,
):
    random.seed(seed)
    np.random.seed(seed)

    base_dir = os.path.join(datasets_dir, version)

    def find_file(directory, filename):
        for ext in [".json", ".jsonl"]:
            path = os.path.join(directory, filename.replace(".json", ext))
            if os.path.exists(path):
                return path
        return os.path.join(directory, filename)  # Fallback to default

    train_path = find_file(base_dir, train_file)
    test_path = find_file(base_dir, test_file)

    logger.info(f"Loading from: {base_dir}")
    logger.info(f"Train: {train_path}")
    logger.info(f"Test : {test_path}")

    train_rows = load_json_array(train_path)
    test_rows = load_json_array(test_path)

    # Preserve the input extension for output files
    train_ext = os.path.splitext(train_path)[1]
    test_ext = os.path.splitext(test_path)[1]

    if len(train_rows) == 0:
        raise ValueError("Train split is empty.")
    if len(test_rows) == 0:
        logger.warning("Test split is empty (server dataset will be empty).")

    # Validate labels exist
    missing = [i for i, r in enumerate(train_rows) if label_col not in r]
    if missing:
        raise ValueError(f"{len(missing)} train rows missing '{label_col}'. First few indices: {missing[:10]}")

    targets = np.array([r[label_col] for r in train_rows], dtype=object)
    logger.info(f"Train rows={len(train_rows)} | Unique labels={len(np.unique(targets))}")

    if split == "homogeneous":
        client_partitions = homogeneous_partitioning(targets, num_clients)
    elif split == "heterogeneous":
        client_partitions = heterogeneous_partitioning(targets, num_clients, alpha)
    else:
        raise ValueError("Invalid type. Use 'homogeneous' or 'heterogeneous'.")

    if split == "heterogeneous":
        run_folder = f"split={split}_clients={num_clients}_alpha={alpha:.3g}"
    else:
        run_folder = f"split={split}_clients={num_clients}"

    # Output dirs
    base_out = os.path.join("data", version, run_folder)

    server_out = os.path.join(base_out, "server", "raw")
    client_out = os.path.join(base_out, "client", "raw")

    clear_directory(server_out)
    clear_directory(client_out)

    # Save server test (preserving format)
    server_test_out = os.path.join(server_out, f"server_test{test_ext}")
    save_json_array(server_test_out, test_rows)
    logger.info(f"Server test saved: {server_test_out} (rows={len(test_rows)})")

    # Save clients train partitions (kept as JSON)
    client_samples = {}
    for cid in range(num_clients):
        idxs = client_partitions[cid]
        client_samples[cid] = len(idxs)
        rows = [train_rows[i] for i in idxs]
        out_path = os.path.join(client_out, f"client_{cid}_train{train_ext}")
        save_json_array(out_path, rows)

    with open(os.path.join(client_out, "client_samples.json"), "w", encoding="utf-8") as f:
        json.dump(client_samples, f, indent=2)

    for cid in range(num_clients):
        logger.info(f"Client {cid}: {client_samples[cid]} rows")

    # Heatmap
    heatmap_path = os.path.join(client_out, "partition_distribution.png")
    save_partition_heatmap(heatmap_path, targets, num_clients, client_partitions)

    logger.info("Done.")


if __name__ == "__main__":
    main()

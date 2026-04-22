# Synthetic Data Processing Project

This project contains a suite of scripts for generating, analyzing, and evaluating synthetic subgraphs from a larger graph dataset. It is managed with [Poetry](https://python-poetry.org/) to ensure a reproducible environment.

## Getting Started

### Prerequisites

- [Python 3.12+](https://www.python.org/)
- [Poetry](https://python-poetry.org/)

### Installation

1.  **Navigate to the project directory**:

    ```bash
    cd synthetic-data-processing
    ```

2.  **Install dependencies**:

    This command will create a virtual environment and install all the necessary packages listed in `pyproject.toml`.

    ```bash
    poetry install
    ```

## Data Setup

Before running any scripts, you must place a file named `all_edges.csv` in the root of the `synthetic-data-processing` directory. This file should contain the complete graph data from which subgraphs will be generated. The `all_edges.csv` file is expected to be provided by the user.


## Scripts

All scripts are run from the root of the `synthetic data processing` directory using the `poetry run` command.

### 1. `generate-subgraphs`

Generates a specified number of random subgraphs from the main `all_edges.csv` file.

-   **Command**:

    ```bash
    poetry run generate-subgraphs <number_of_subgraphs>
    ```

-   **Example**:

    ```bash
    poetry run generate-subgraphs 100
    ```

-   **Output**: Creates a `subgraphs/` directory containing the generated CSV files (e.g., `01.csv`, `02.csv`, etc.).

### 2. `calculate-stats`

Calculates and displays aggregate statistics (average and standard deviation for positive, negative, and missing edges) across all generated subgraphs.

-   **Command**:

    ```bash
    poetry run calculate-stats
    ```

### 3. `calculate-label-ranges`

Calculates and prints the label ranges for generating training data points based on the statistics of the generated subgraphs.

-   **Command**:

    ```bash
    poetry run calculate-label-ranges --alpha <value> --beta <value> [--noise <value>]
    ```

-   **Arguments**:
    -   `--alpha`: Multiplier for the standard deviation to define the target ranges.
    -   `--beta`: Multiplier for the standard deviation to define the tolerance window.
    -   `--noise` (optional): The percentage of noise to apply.

-   **Example**:

    ```bash
    poetry run calculate-label-ranges --alpha 2.0 --beta 0.5 --noise 10
    ```

### 4. `calculate-mean-rel-freq`

Calculates the mean frequency of each relationship type across all generated subgraphs and displays a leaderboard.

-   **Command**:

    ```bash
    poetry run calculate-mean-rel-freq
    ```

## Typical Workflow

1.  **Generate Subgraphs**: Start by creating a new set of subgraphs.

    ```bash
    poetry run generate-subgraphs 10000
    ```

2.  **Calculate Statistics**: Analyze the generated subgraphs to get their statistical profile.

    ```bash
    poetry run calculate-stats
    ```

3.  **Determine Label Ranges**: Use the calculated stats to determine the ranges for your training data labels.

    ```bash
    poetry run calculate-label-ranges --alpha 2.5 --beta 0.5
    ```

4.  **Analyze Relationship Frequencies**: Optionally, you can also analyze the relationship frequencies within the generated subgraphs.

    ```bash
    poetry run calculate-mean-rel-freq
    ```

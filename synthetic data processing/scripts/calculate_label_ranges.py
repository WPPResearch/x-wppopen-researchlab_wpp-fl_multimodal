import argparse
import sys

from scripts.calculate_subgraph_stats import get_subgraph_stats


def print_ranges(pos_avg, neg_avg, mis_avg, pos_std, neg_std, mis_std, alpha, beta, noise, subgraph_count):
    print(f"--- Subgraph Label Ranges (based on {subgraph_count} subgraphs) ---")
    print(f"Parameters: alpha={alpha}, beta={beta}, noise={noise if noise is not None else 0}%")
    print(f"Baselines: pos_avg={pos_avg:.2f}%, neg_avg={neg_avg:.2f}%, mis_avg={mis_avg:.2f}%")
    print(f"           pos_std={pos_std:.2f}%, neg_std={neg_std:.2f}%, mis_std={mis_std:.2f}%\n")

    # ... (rest of the print_ranges function is unchanged) ...
    if noise is None or noise == 0:
        print("[ NOISE-FREE POINTS ]\n")
        pos_target = pos_avg + alpha * pos_std
        pos_min = max(0.0, pos_target - 5.0)
        pos_max = min(100.0, pos_target + 5.0)
        neg_lower = max(0.0, (neg_avg - beta * neg_std) - 5.0)
        neg_upper = min(100.0, (neg_avg + beta * neg_std) + 5.0)
        print("1. Noise-Free Positive Point:")
        print(f"   - Positive edges: between {pos_min:.2f}% and {pos_max:.2f}% (pos_avg + {alpha} * pos_std +/- 5%)")
        print(
            f"   - Negative edges: between {neg_lower:.2f}% and {neg_upper:.2f}% (neg_avg +/- {beta} * neg_std +/- 5%)"
        )
        print("")
        neg_target = neg_avg + alpha * neg_std
        neg_min = max(0.0, neg_target - 5.0)
        neg_max = min(100.0, neg_target + 5.0)
        pos_lower = max(0.0, (pos_avg - beta * pos_std) - 5.0)
        pos_upper = min(100.0, (pos_avg + beta * pos_std) + 5.0)
        print("2. Noise-Free Negative Point:")
        print(f"   - Negative edges: between {neg_min:.2f}% and {neg_max:.2f}% (neg_avg + {alpha} * neg_std +/- 5%)")
        print(
            f"   - Positive edges: between {pos_lower:.2f}% and {pos_upper:.2f}% (pos_avg +/- {beta} * pos_std +/- 5%)"
        )
        print("")
        mis_lower = max(0.0, (mis_avg - beta * mis_std) - 5.0)
        mis_upper = min(100.0, (mis_avg + beta * mis_std) + 5.0)
        print("3. Noise-Free Neutral Point:")
        print(
            f"   - Positive edges: between {pos_lower:.2f}% and {pos_upper:.2f}% (pos_avg +/- {beta} * pos_std +/- 5%)"
        )
        print(
            f"   - Negative edges: between {neg_lower:.2f}% and {neg_upper:.2f}% (neg_avg +/- {beta} * neg_std +/- 5%)"
        )
        print(
            f"   - Missing edges:  between {mis_lower:.2f}% and {mis_upper:.2f}% (mis_avg +/- {beta} * mis_std +/- 5%)"
        )
        print("")
    else:
        print(f"[ POINTS WITH {noise}% NOISE ]\n")
        pos_target = pos_avg + alpha * pos_std
        pos_min = max(0.0, pos_target - 5.0)
        pos_max = min(100.0, pos_target + 5.0)
        target_neg = neg_avg * (1 + noise / 100.0)
        neg_lower = max(0.0, target_neg - 5.0)
        neg_upper = min(100.0, target_neg + 5.0)
        print(f"1. Positive Point with {noise}% Noise:")
        print(f"   - Positive edges: between {pos_min:.2f}% and {pos_max:.2f}% (pos_avg + {alpha} * pos_std +/- 5%)")
        print(
            f"   - Negative edges: between {neg_lower:.2f}% and {neg_upper:.2f}% (neg_avg * (1 + {noise}/100) +/- 5%)"
        )
        print("")
        neg_target = neg_avg + alpha * neg_std
        neg_min = max(0.0, neg_target - 5.0)
        neg_max = min(100.0, neg_target + 5.0)
        target_pos = pos_avg * (1 + noise / 100.0)
        pos_lower = max(0.0, target_pos - 5.0)
        pos_upper = min(100.0, target_pos + 5.0)
        print(f"2. Negative Point with {noise}% Noise:")
        print(f"   - Negative edges: between {neg_min:.2f}% and {neg_max:.2f}% (neg_avg + {alpha} * neg_std +/- 5%)")
        print(
            f"   - Positive edges: between {pos_lower:.2f}% and {pos_upper:.2f}% (pos_avg * (1 + {noise}/100) +/- 5%)"
        )
        print("")
        y = noise / 200.0
        target_pos_neutral = pos_avg * (1 + y)
        target_neg_neutral = neg_avg * (1 + y)
        pos_lower_neu = max(0.0, target_pos_neutral - 5.0)
        pos_upper_neu = min(100.0, target_pos_neutral + 5.0)
        neg_lower_neu = max(0.0, target_neg_neutral - 5.0)
        neg_upper_neu = min(100.0, target_neg_neutral + 5.0)
        print(f"3. Neutral Point with {noise}% Noise:")
        print(
            f"   - Positive edges: between {pos_lower_neu:.2f}% and {pos_upper_neu:.2f}%"
            f" (pos_avg * (1 + {noise}/200) +/- 5%)"
        )
        print(
            f"   - Negative edges: between {neg_lower_neu:.2f}% and {neg_upper_neu:.2f}%"
            f" (neg_avg * (1 + {noise}/200) +/- 5%)"
        )
        print("")


def main():
    """Calculates stats and then prints label ranges."""
    parser = argparse.ArgumentParser(
        description="Calculate ranges for pos/neg/neutral subgraphs based on calculated baseline stats."
    )
    parser.add_argument(
        "--alpha", type=float, required=True, help="Alpha multiplier for positiveness threshold (e.g. 1, 1.5, 2)"
    )
    parser.add_argument(
        "--beta", type=float, required=True, help="Beta multiplier for tolerance window (e.g. 0.25, 0.5)"
    )
    parser.add_argument(
        "--noise", type=float, default=None, help="Noise percentage (x). If None, prints noise-free ranges."
    )

    args = parser.parse_args()

    # Automatically calculate the subgraph stats first
    stats = get_subgraph_stats()

    if stats:
        print_ranges(
            stats["pos_avg"],
            stats["neg_avg"],
            stats["mis_avg"],
            stats["pos_std"],
            stats["neg_std"],
            stats["mis_std"],
            args.alpha,
            args.beta,
            args.noise,
            stats["subgraph_count"],
        )
    else:
        print("Could not calculate label ranges because subgraph statistics failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

import math
from typing import List, Tuple

import matplotlib.pyplot as plt


def compute_precision_recall_at_k(binary_relevances: List[int]) -> Tuple[List[float], List[float]]:
    """
    Compute precision and recall at each cutoff k = 1..N for a single ranked list
    using binary relevances (1=relevant, 0=non-relevant).
    """
    precisions: List[float] = []
    recalls: List[float] = []

    total_relevant: int = sum(binary_relevances)
    hit_so_far: int = 0

    for k in range(1, len(binary_relevances) + 1):
        hit_so_far += binary_relevances[k - 1]
        precision_at_k: float = hit_so_far / k
        recall_at_k: float = (hit_so_far / total_relevant) if total_relevant > 0 else 0.0
        precisions.append(precision_at_k)
        recalls.append(recall_at_k)

    return precisions, recalls


def plot_pr_curve(recalls: List[float], precisions: List[float], out_path: str) -> None:
    plt.figure(figsize=(6, 4))
    # Connect points in order of k
    plt.plot(recalls, precisions, marker="o", linestyle="-", color="#1f77b4", label="PR curve (k=1..N)")
    # Also scatter points for clarity
    plt.scatter(recalls, precisions, color="#1f77b4")

    for idx, (r, p) in enumerate(zip(recalls, precisions), start=1):
        plt.annotate(str(idx), (r, p), textcoords="offset points", xytext=(5, 5), fontsize=8, color="#333333")

    plt.title("Precision-Recall Curve (Exercise 2)")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.xlim(0.0, 1.02)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", linewidth=0.6)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def main() -> None:
    # Binary relevance for Exercise 2 top-10, as specified in the assignment
    # rel = [1,1,1,1,0,1,0,1,0,1]
    binary_relevances: List[int] = [1, 1, 1, 1, 0, 1, 0, 1, 0, 1]

    precisions, recalls = compute_precision_recall_at_k(binary_relevances)
    out_path: str = "pr_curve_ex2.png"
    plot_pr_curve(recalls, precisions, out_path)
    print(f"Saved PR curve to {out_path}")


if __name__ == "__main__":
    main()



from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    metrics_path = Path("runs/training_metrics.jsonl")
    if not metrics_path.exists():
        raise SystemExit(f"Missing {metrics_path}. Run training first.")

    steps = []
    mean_rewards = []
    entropies = []

    with metrics_path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            steps.append(int(row["step"]))
            mean_rewards.append(float(row.get("mean_reward", 0.0)))
            entropies.append(float(row.get("entropy", 0.0)))

    out_dir = Path("plots")
    out_dir.mkdir(exist_ok=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, mean_rewards, linewidth=2)
    plt.title("Training progress: mean reward (rolling)")
    plt.xlabel("training step")
    plt.ylabel("mean reward")
    plt.grid(True, alpha=0.25)
    p1 = out_dir / "reward_curve.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, entropies, linewidth=2)
    plt.title("Training stability: entropy")
    plt.xlabel("training step")
    plt.ylabel("entropy")
    plt.grid(True, alpha=0.25)
    p2 = out_dir / "entropy_curve.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()

    print(f"Wrote plots: {p1} and {p2}")


if __name__ == "__main__":
    main()


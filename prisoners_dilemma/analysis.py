"""Load Prisoner's Dilemma eval logs and produce analysis + visualizations."""

import json
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _read_eval_file(fpath: Path) -> tuple[dict, list[dict]]:
    """Read an .eval file (zip archive with header.json + samples/*.json)."""
    if zipfile.is_zipfile(fpath):
        with zipfile.ZipFile(fpath) as z:
            header = json.loads(z.read("header.json"))
            samples = []
            for name in sorted(z.namelist()):
                if name.startswith("samples/") and name.endswith(".json"):
                    samples.append(json.loads(z.read(name)))
            return header, samples
    else:
        data = json.loads(fpath.read_text())
        return data, data.get("samples", [])


def load_logs(log_dir: str | None = None) -> pd.DataFrame:
    """Load all PD eval logs into a DataFrame."""
    if log_dir is None:
        if Path("logs").exists():
            log_dir = "logs"
        else:
            log_dir = os.path.expanduser("~/.inspect_ai/logs")

    log_path = Path(log_dir)
    rows = []

    for fpath in sorted(log_path.rglob("*.eval")):
        try:
            header, samples = _read_eval_file(fpath)
        except Exception:
            continue

        task_name = header.get("eval", {}).get("task", "")
        if not task_name.startswith("pd_"):
            continue

        model = header.get("eval", {}).get("model", "")
        model_roles = header.get("eval", {}).get("model_roles", {})
        partner_raw = model_roles.get("partner", "")
        partner = partner_raw.get("model", "") if isinstance(partner_raw, dict) else partner_raw
        config = header.get("eval", {}).get("config", {})
        temperature = config.get("temperature", None)

        for sample in samples:
            score_data = {}
            scores = sample.get("scores", {})
            for scorer_name, score_obj in scores.items():
                score_data = score_obj
                break

            metadata = score_data.get("metadata", {})

            rows.append({
                "log_file": fpath.name,
                "task": task_name,
                "model_a": model,
                "model_b": partner,
                "temperature": temperature,
                "num_rounds": metadata.get("num_rounds", 0),
                "cooperation_rate_a": metadata.get("cooperation_rate_a", 0),
                "cooperation_rate_b": metadata.get("cooperation_rate_b", 0),
                "mutual_cooperation_rate": metadata.get("mutual_cooperation_rate", 0),
                "mutual_defection_rate": metadata.get("mutual_defection_rate", 0),
                "total_score_a": metadata.get("total_score_a", 0),
                "total_score_b": metadata.get("total_score_b", 0),
                "all_choices_a": metadata.get("all_choices_a", []),
                "all_choices_b": metadata.get("all_choices_b", []),
                "strategy_a": metadata.get("strategy_a", ""),
                "strategy_b": metadata.get("strategy_b", ""),
            })

    return pd.DataFrame(rows)


def short_name(model_id: str) -> str:
    """Extract short model name from OpenRouter ID."""
    parts = model_id.replace("openrouter/", "").split("/")
    return parts[-1] if parts else model_id


def pair_label(row: pd.Series) -> str:
    a, b = short_name(row["model_a"]), short_name(row["model_b"])
    if a == b:
        return f"{a}\n(self)"
    return f"{a}\nvs {b}"


# ── 1. Cooperation rate by model ─────────────────────────────────────────────

def cooperation_by_model(df: pd.DataFrame, output_path: str = "coop_by_model.png") -> None:
    """Bar chart of average cooperation rate per model (as both A and B)."""
    if df.empty:
        return

    # Gather cooperation rates for each model in either role
    model_coop: dict[str, list[float]] = {}
    for _, row in df.iterrows():
        ma = short_name(row["model_a"])
        mb = short_name(row["model_b"])
        model_coop.setdefault(ma, []).append(row["cooperation_rate_a"])
        model_coop.setdefault(mb, []).append(row["cooperation_rate_b"])

    models = sorted(model_coop.keys())
    means = [np.mean(model_coop[m]) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(models, means, color="mediumseagreen", edgecolor="black")
    ax.set_ylabel("Average Cooperation Rate")
    ax.set_title("Cooperation Rate by Model")
    ax.set_ylim(0, 1)
    for i, v in enumerate(means):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=10)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 2. Cooperation rate by round count ───────────────────────────────────────

def cooperation_by_rounds(df: pd.DataFrame, output_path: str = "coop_by_rounds.png") -> None:
    """Bar chart comparing cooperation rate across 1, 3, 10 round games."""
    if df.empty:
        return

    df = df.copy()
    # Average cooperation of both players
    df["avg_coop"] = (df["cooperation_rate_a"] + df["cooperation_rate_b"]) / 2

    grouped = df.groupby("num_rounds")["avg_coop"].mean().sort_index()

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar([str(r) for r in grouped.index], grouped.values,
                  color=["steelblue", "mediumseagreen", "coral"], edgecolor="black")
    ax.set_xlabel("Number of Rounds")
    ax.set_ylabel("Average Cooperation Rate")
    ax.set_title("Cooperation Rate by Game Length")
    ax.set_ylim(0, 1)
    for bar, v in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.02, f"{v:.0%}",
                ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 3. Strategy over time (10-round games) ──────────────────────────────────

def strategy_over_time(df: pd.DataFrame, output_path: str = "strategy_over_time.png") -> None:
    """Line chart of cooperation probability per round number for 10-round games."""
    df10 = df[df["num_rounds"] == 10].copy()
    if df10.empty:
        return

    max_rounds = 10
    round_coop = {r: [] for r in range(1, max_rounds + 1)}

    for _, row in df10.iterrows():
        choices_a = row["all_choices_a"]
        choices_b = row["all_choices_b"]
        for r in range(min(len(choices_a), max_rounds)):
            round_coop[r + 1].append(1 if choices_a[r] == "cooperate" else 0)
            round_coop[r + 1].append(1 if choices_b[r] == "cooperate" else 0)

    rounds = list(range(1, max_rounds + 1))
    rates = [np.mean(round_coop[r]) if round_coop[r] else 0 for r in rounds]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rounds, rates, marker="o", linewidth=2, color="steelblue")
    ax.fill_between(rounds, rates, alpha=0.2, color="steelblue")
    ax.set_xlabel("Round Number")
    ax.set_ylabel("Cooperation Rate")
    ax.set_title("Cooperation Rate Over Time (10-Round Games)")
    ax.set_ylim(0, 1)
    ax.set_xticks(rounds)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 4. Mutual cooperation heatmap ────────────────────────────────────────────

def mutual_cooperation_heatmap(df: pd.DataFrame, output_path: str = "mutual_coop_heatmap.png") -> None:
    """4x4 heatmap of mutual cooperation rate per model pair."""
    if df.empty:
        return

    models = sorted(set(df["model_a"].map(short_name)) | set(df["model_b"].map(short_name)))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        rate = group["mutual_cooperation_rate"].mean()
        sa, sb = short_name(ma), short_name(mb)
        matrix.loc[sa, sb] = rate
        matrix.loc[sb, sa] = rate

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".0%", cmap="YlGn",
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Mutual Cooperation Rate by Model Pair")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 5. Score heatmap ─────────────────────────────────────────────────────────

def score_heatmap(df: pd.DataFrame, output_path: str = "score_heatmap.png") -> None:
    """4x4 heatmap of average total score per model (as model A)."""
    if df.empty:
        return

    models = sorted(set(df["model_a"].map(short_name)) | set(df["model_b"].map(short_name)))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        sa, sb = short_name(ma), short_name(mb)
        avg_a = group["total_score_a"].mean()
        avg_b = group["total_score_b"].mean()
        matrix.loc[sa, sb] = avg_a
        matrix.loc[sb, sa] = avg_b

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".1f", cmap="YlOrRd",
                ax=ax, square=True)
    ax.set_title("Average Total Score by Model Pair\n(row = model, column = partner)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 6. Strategy detection summary ────────────────────────────────────────────

def strategy_summary(df: pd.DataFrame, output_path: str = "strategy_summary.png") -> None:
    """Bar chart of detected strategy patterns."""
    if df.empty:
        return

    # Combine strategies from both roles
    strategies = list(df["strategy_a"]) + list(df["strategy_b"])
    strat_counts = pd.Series(strategies).value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))
    strat_counts.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Count")
    ax.set_title("Detected Strategy Patterns (All Models)")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    for i, v in enumerate(strat_counts.values):
        ax.text(i, v + 0.3, str(v), ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Run all ──────────────────────────────────────────────────────────────────

def run_analysis(log_dir: str | None = None, output_dir: str = "pd_plots") -> None:
    """Run all PD analyses and save plots."""
    df = load_logs(log_dir)
    print(f"Loaded {len(df)} PD game samples from logs.")
    if df.empty:
        print("No Prisoner's Dilemma eval data found. Run evaluations first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    cooperation_by_model(df, f"{output_dir}/coop_by_model.png")
    cooperation_by_rounds(df, f"{output_dir}/coop_by_rounds.png")
    strategy_over_time(df, f"{output_dir}/strategy_over_time.png")
    mutual_cooperation_heatmap(df, f"{output_dir}/mutual_coop_heatmap.png")
    score_heatmap(df, f"{output_dir}/score_heatmap.png")
    strategy_summary(df, f"{output_dir}/strategy_summary.png")

    # Print summary
    print(f"\n{'='*60}")
    print("PRISONER'S DILEMMA SUMMARY")
    print(f"{'='*60}")
    print(f"  Total games: {len(df)}")
    print(f"  Unique model pairs: {df.groupby(['model_a', 'model_b']).ngroups}")
    avg_coop = (df["cooperation_rate_a"].mean() + df["cooperation_rate_b"].mean()) / 2
    print(f"  Overall cooperation rate: {avg_coop:.1%}")
    print(f"  Mutual cooperation rate: {df['mutual_cooperation_rate'].mean():.1%}")
    print(f"  Mutual defection rate: {df['mutual_defection_rate'].mean():.1%}")

    print(f"\nBy round count:")
    for nrounds, group in df.groupby("num_rounds"):
        gc = (group["cooperation_rate_a"].mean() + group["cooperation_rate_b"].mean()) / 2
        print(f"  {nrounds} rounds: coop={gc:.0%}, mutual_coop={group['mutual_cooperation_rate'].mean():.0%}")

    print(f"\nPer model pair:")
    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        sa, sb = short_name(ma), short_name(mb)
        ca = group["cooperation_rate_a"].mean()
        cb = group["cooperation_rate_b"].mean()
        mc = group["mutual_cooperation_rate"].mean()
        print(f"  {sa} vs {sb}: coop_a={ca:.0%} coop_b={cb:.0%} mutual={mc:.0%}")

    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Prisoner's Dilemma eval results")
    parser.add_argument("--log-dir", default=None, help="Path to inspect logs directory")
    parser.add_argument("--output-dir", default="pd_plots", help="Directory for output plots")
    args = parser.parse_args()
    run_analysis(log_dir=args.log_dir, output_dir=args.output_dir)

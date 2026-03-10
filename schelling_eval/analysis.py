"""Load eval logs and produce convergence analysis + visualizations."""

import json
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_logs(log_dir: str | None = None) -> pd.DataFrame:
    """Load all Inspect eval logs into a DataFrame.

    Handles both zip (.eval) and plain JSON log formats.
    """
    if log_dir is None:
        # Check local logs/ first, then default inspect location
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
        if "schelling" not in task_name.lower():
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
            sample_input = sample.get("input", "")
            if isinstance(sample_input, list):
                sample_input = sample_input[-1].get("content", "") if sample_input else ""

            rows.append({
                "log_file": fpath.name,
                "task": task_name,
                "model_a": model,
                "model_b": partner,
                "temperature": temperature,
                "sample_id": sample.get("id", ""),
                "question": sample_input,
                "category": sample.get("metadata", {}).get("category", ""),
                "experiment_type": sample.get("metadata", {}).get("experiment_type", ""),
                "value": score_data.get("value", ""),
                "answer_a": metadata.get("answer_a", ""),
                "answer_b": metadata.get("answer_b", ""),
                "turn_matched": metadata.get("turn_matched", -1),
                "total_turns": metadata.get("total_turns", 0),
                "match_type": metadata.get("match_type", ""),
                "visible": metadata.get("visible", False),
                "all_answers_a": metadata.get("all_answers_a", []),
                "all_answers_b": metadata.get("all_answers_b", []),
            })

    return pd.DataFrame(rows)


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


def short_name(model_id: str) -> str:
    """Extract short model name from OpenRouter ID."""
    parts = model_id.replace("openrouter/", "").split("/")
    return parts[-1] if parts else model_id


def pair_label(row: pd.Series) -> str:
    """Short label for a model pair."""
    a, b = short_name(row["model_a"]), short_name(row["model_b"])
    if a == b:
        return f"{a}\n(self)"
    return f"{a}\nvs {b}"


# ── 1. Schelling Rate: % matching on turn 1 ──────────────────────────────────

def schelling_rate(df: pd.DataFrame, output_path: str = "schelling_rate.png") -> None:
    """Bar chart of first-turn match rate per model pair."""
    if df.empty:
        return

    df = df.copy()
    df["first_turn_match"] = (df["turn_matched"] == 1).astype(int)
    df["pair"] = df.apply(pair_label, axis=1)

    rates = df.groupby("pair")["first_turn_match"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    rates.plot(kind="bar", ax=ax, color="mediumseagreen", edgecolor="black")
    ax.set_ylabel("First-Turn Match Rate")
    ax.set_title("Schelling Rate (% matching on turn 1)")
    ax.set_ylim(0, 1)
    for i, v in enumerate(rates):
        ax.text(i, v + 0.02, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 2. Turns-to-match distribution per question ──────────────────────────────

def turns_distribution_per_question(df: pd.DataFrame, output_path: str = "turns_per_question.png") -> None:
    """Heatmap showing turn_matched for each question x model pair."""
    if df.empty:
        return

    df = df.copy()
    df["pair"] = df.apply(pair_label, axis=1)
    # Use -1 → 6 so "never matched" shows distinctly
    df["turns_display"] = df["turn_matched"].replace(-1, 6)

    pivot = df.pivot_table(index="question", columns="pair", values="turns_display", aggfunc="mean")
    # Shorten question text
    pivot.index = [q[:50] + "..." if len(q) > 50 else q for q in pivot.index]

    fig, ax = plt.subplots(figsize=(12, max(8, len(pivot) * 0.35)))
    cmap = sns.color_palette("RdYlGn_r", as_cmap=True)
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap=cmap, vmin=1, vmax=6,
                ax=ax, linewidths=0.5)
    ax.set_title("Turns to Match per Question (6 = never matched)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 3. Heatmap of average turns per model pair ───────────────────────────────

def avg_turns_heatmap(df: pd.DataFrame, output_path: str = "avg_turns_heatmap.png") -> None:
    """4x4 heatmap of average turns to match per model pair."""
    if df.empty:
        return

    df = df.copy()
    # Treat never-matched as max_turns + 1 so it inflates the average
    df["turns_for_avg"] = df["turn_matched"].replace(-1, 6)

    models = sorted(set(df["model_a"].map(short_name)) | set(df["model_b"].map(short_name)))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        avg = group["turns_for_avg"].mean()
        sa, sb = short_name(ma), short_name(mb)
        matrix.loc[sa, sb] = avg
        matrix.loc[sb, sa] = avg

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="RdYlGn_r",
                vmin=1, vmax=5, ax=ax, square=True)
    ax.set_title("Average Turns to Match by Model Pair\n(lower = faster convergence)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 4. Overall match rate heatmap ─────────────────────────────────────────────

def match_rate_heatmap(df: pd.DataFrame, output_path: str = "match_rate_heatmap.png") -> None:
    """4x4 heatmap of overall match % per model pair (across all turns)."""
    if df.empty:
        return

    df = df.copy()
    df["matched"] = (df["value"] == "C").astype(int)

    models = sorted(set(df["model_a"].map(short_name)) | set(df["model_b"].map(short_name)))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        rate = group["matched"].mean()
        sa, sb = short_name(ma), short_name(mb)
        matrix.loc[sa, sb] = rate
        matrix.loc[sb, sa] = rate

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".0%", cmap="YlGn",
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Overall Match Rate by Model Pair\n(% matched within 5 turns)")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 5. Category breakdown ────────────────────────────────────────────────────

def category_breakdown(df: pd.DataFrame, output_path: str = "category_breakdown.png") -> None:
    """Match rate + schelling rate by question category."""
    if df.empty:
        return

    df = df.copy()
    df["matched"] = (df["value"] == "C").astype(int)
    df["first_turn"] = (df["turn_matched"] == 1).astype(int)

    cats = df.groupby("category").agg(
        match_rate=("matched", "mean"),
        schelling_rate=("first_turn", "mean"),
    ).sort_values("schelling_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(cats))
    w = 0.35
    ax.bar(x - w/2, cats["schelling_rate"], w, label="Turn-1 match", color="mediumseagreen", edgecolor="black")
    ax.bar(x + w/2, cats["match_rate"], w, label="Overall match", color="steelblue", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels(cats.index, rotation=30, ha="right")
    ax.set_ylabel("Rate")
    ax.set_ylim(0, 1)
    ax.set_title("Match Rate by Category")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 6. Same-model vs cross-model turns distribution ─────────────────────────

def same_vs_cross(df: pd.DataFrame, output_path: str = "same_vs_cross.png") -> None:
    """Side-by-side histogram of turns to match: same-model vs cross-model."""
    if df.empty:
        return

    df = df.copy()
    df["pair_type"] = df.apply(
        lambda r: "same-model" if r["model_a"] == r["model_b"] else "cross-model", axis=1
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (ptype, group) in zip(axes, df.groupby("pair_type")):
        counts = group["turn_matched"].value_counts().sort_index()
        labels = [str(t) if t > 0 else "no match" for t in counts.index]
        colors = ["mediumseagreen" if t == 1 else "steelblue" if t > 0 else "salmon" for t in counts.index]
        ax.bar(labels, counts.values, color=colors, edgecolor="black")
        ax.set_title(ptype)
        ax.set_xlabel("Turn matched")
        ax.set_ylabel("Count")
        total = len(group)
        t1 = (group["turn_matched"] == 1).sum()
        matched = (group["turn_matched"] > 0).sum()
        ax.text(0.95, 0.95, f"Schelling rate: {t1/total:.0%}\nOverall match: {matched/total:.0%}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8))

    fig.suptitle("Turn Distribution: Same-Model vs Cross-Model", fontsize=13)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── 7. Hardest / easiest questions ───────────────────────────────────────────

def question_difficulty(df: pd.DataFrame, output_path: str = "question_difficulty.png") -> None:
    """Horizontal bar chart ranking questions by first-turn match rate."""
    if df.empty:
        return

    df = df.copy()
    df["first_turn"] = (df["turn_matched"] == 1).astype(int)
    df["matched"] = (df["value"] == "C").astype(int)

    q_stats = df.groupby("question").agg(
        schelling_rate=("first_turn", "mean"),
        match_rate=("matched", "mean"),
    ).sort_values("schelling_rate", ascending=True)

    q_stats.index = [q[:55] + "..." if len(q) > 55 else q for q in q_stats.index]

    fig, ax = plt.subplots(figsize=(10, max(6, len(q_stats) * 0.3)))
    y = np.arange(len(q_stats))
    ax.barh(y, q_stats["schelling_rate"], color="mediumseagreen", edgecolor="black", label="Turn-1")
    ax.barh(y, q_stats["match_rate"] - q_stats["schelling_rate"],
            left=q_stats["schelling_rate"], color="steelblue", edgecolor="black", alpha=0.6, label="Later turns")
    ax.set_yticks(y)
    ax.set_yticklabels(q_stats.index, fontsize=7)
    ax.set_xlabel("Match Rate")
    ax.set_xlim(0, 1)
    ax.set_title("Question Difficulty (sorted by Schelling rate)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


# ── Run all ──────────────────────────────────────────────────────────────────

def run_analysis(log_dir: str | None = None, output_dir: str = "schelling_plots") -> None:
    """Run all analyses and save plots."""
    df = load_logs(log_dir)
    print(f"Loaded {len(df)} scored samples from logs.")
    if df.empty:
        print("No Schelling eval data found. Run evaluations first.")
        return

    os.makedirs(output_dir, exist_ok=True)

    schelling_rate(df, f"{output_dir}/schelling_rate.png")
    match_rate_heatmap(df, f"{output_dir}/match_rate_heatmap.png")
    avg_turns_heatmap(df, f"{output_dir}/avg_turns_heatmap.png")
    turns_distribution_per_question(df, f"{output_dir}/turns_per_question.png")
    category_breakdown(df, f"{output_dir}/category_breakdown.png")
    same_vs_cross(df, f"{output_dir}/same_vs_cross.png")
    question_difficulty(df, f"{output_dir}/question_difficulty.png")

    # Print summary
    df["matched"] = (df["value"] == "C").astype(int)
    df["first_turn"] = (df["turn_matched"] == 1).astype(int)
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"  Total samples: {len(df)}")
    print(f"  Unique model pairs: {df.groupby(['model_a', 'model_b']).ngroups}")
    print(f"  Schelling rate (turn 1): {df['first_turn'].mean():.1%}")
    print(f"  Overall match rate: {df['matched'].mean():.1%}")
    print(f"  Avg turns to match (when matched): {df.loc[df['turn_matched'] > 0, 'turn_matched'].mean():.2f}")
    print(f"\nPer model pair:")
    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        sa, sb = short_name(ma), short_name(mb)
        t1 = (group["turn_matched"] == 1).mean()
        total = group["matched"].mean()
        print(f"  {sa} vs {sb}: schelling={t1:.0%}  overall={total:.0%}")
    print(f"\nPlots saved to {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Schelling eval results")
    parser.add_argument("--log-dir", default=None, help="Path to inspect logs directory")
    parser.add_argument("--output-dir", default="plots", help="Directory for output plots")
    args = parser.parse_args()
    run_analysis(log_dir=args.log_dir, output_dir=args.output_dir)

"""Load eval logs and produce convergence analysis + visualizations."""

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def load_logs(log_dir: str | None = None) -> pd.DataFrame:
    """Load all Inspect eval logs into a DataFrame.

    Inspect stores logs as .eval files (JSON) in ~/.inspect_ai/logs/ by default.
    """
    if log_dir is None:
        log_dir = os.path.expanduser("~/.inspect_ai/logs")

    log_path = Path(log_dir)
    rows = []

    for fpath in sorted(log_path.rglob("*.eval")):
        try:
            data = json.loads(fpath.read_text())
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue

        # Extract run-level metadata
        eval_spec = data.get("eval", {})
        task_name = eval_spec.get("task", "")
        if "schelling" not in task_name.lower():
            continue

        model = eval_spec.get("model", "")
        model_roles = eval_spec.get("model_roles", {})
        partner = model_roles.get("partner", "")
        config = eval_spec.get("config", {})
        temperature = config.get("temperature", None)

        results = data.get("results", {})
        samples = results.get("samples", data.get("samples", []))

        for sample in samples:
            score_data = {}
            scores = sample.get("scores", {})
            for scorer_name, score_obj in scores.items():
                score_data = score_obj
                break  # take first scorer

            metadata = score_data.get("metadata", {})
            rows.append({
                "log_file": fpath.name,
                "task": task_name,
                "model_a": model,
                "model_b": partner,
                "temperature": temperature,
                "sample_id": sample.get("id", ""),
                "question": sample.get("input", ""),
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


def short_name(model_id: str) -> str:
    """Extract short model name from OpenRouter ID."""
    parts = model_id.replace("openrouter/", "").split("/")
    return parts[-1] if parts else model_id


def convergence_matrix(df: pd.DataFrame, output_path: str = "convergence_matrix.png") -> None:
    """Heatmap of match rate per model pair."""
    if df.empty:
        print("No data for convergence matrix.")
        return

    df = df.copy()
    df["matched"] = (df["value"] == "C").astype(int)
    df["pair"] = df.apply(
        lambda r: tuple(sorted([short_name(r["model_a"]), short_name(r["model_b"])])), axis=1
    )

    models = sorted(set(df["model_a"].map(short_name)) | set(df["model_b"].map(short_name)))
    matrix = pd.DataFrame(np.nan, index=models, columns=models)

    for (ma, mb), group in df.groupby(["model_a", "model_b"]):
        rate = group["matched"].mean()
        sa, sb = short_name(ma), short_name(mb)
        matrix.loc[sa, sb] = rate
        matrix.loc[sb, sa] = rate

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(matrix.astype(float), annot=True, fmt=".2f", cmap="YlOrRd",
                vmin=0, vmax=1, ax=ax, square=True)
    ax.set_title("Schelling Convergence Rate by Model Pair")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def turns_to_match(df: pd.DataFrame, output_path: str = "turns_to_match.png") -> None:
    """Distribution of turns needed to match, by pair type."""
    if df.empty:
        print("No data for turns-to-match.")
        return

    df = df.copy()
    df["pair_type"] = df.apply(
        lambda r: "same-model" if r["model_a"] == r["model_b"] else "cross-model", axis=1
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for ptype, group in df.groupby("pair_type"):
        vals = group["turn_matched"].replace(-1, np.nan).dropna()
        if not vals.empty:
            ax.hist(vals, bins=range(1, 7), alpha=0.6, label=ptype, edgecolor="black")
    ax.set_xlabel("Turn Matched")
    ax.set_ylabel("Count")
    ax.set_title("Turns to Match: Same-Model vs Cross-Model")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def temperature_effect(df: pd.DataFrame, output_path: str = "temperature_effect.png") -> None:
    """Side-by-side convergence rates at temp=0 vs temp=1."""
    if df.empty:
        print("No data for temperature effect.")
        return

    df = df.copy()
    df["matched"] = (df["value"] == "C").astype(int)

    summary = df.groupby(["temperature", "visible"])["matched"].mean().reset_index()
    summary["mode"] = summary["visible"].map({True: "visible", False: "blind"})

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode, group in summary.groupby("mode"):
        ax.bar(
            [f"temp={t}\n({mode})" for t in group["temperature"]],
            group["matched"],
            alpha=0.7,
            label=mode,
        )
    ax.set_ylabel("Match Rate")
    ax.set_title("Convergence Rate by Temperature and Mode")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def category_breakdown(df: pd.DataFrame, output_path: str = "category_breakdown.png") -> None:
    """Match rate by question category."""
    if df.empty:
        print("No data for category breakdown.")
        return

    df = df.copy()
    df["matched"] = (df["value"] == "C").astype(int)

    cat_rates = df.groupby("category")["matched"].mean().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    cat_rates.plot(kind="bar", ax=ax, color="steelblue", edgecolor="black")
    ax.set_ylabel("Match Rate")
    ax.set_title("Convergence Rate by Question Category")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def first_guess_distributions(df: pd.DataFrame, output_dir: str = "word_dists") -> None:
    """Per question, histogram of turn-1 answers by model."""
    if df.empty:
        print("No data for word distributions.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for sample_id, group in df.groupby("sample_id"):
        answers = []
        for _, row in group.iterrows():
            a_list = row.get("all_answers_a", [])
            b_list = row.get("all_answers_b", [])
            if a_list:
                answers.append({"model": short_name(row["model_a"]), "answer": a_list[0].lower()})
            if b_list:
                answers.append({"model": short_name(row["model_b"]), "answer": b_list[0].lower()})

        if not answers:
            continue

        adf = pd.DataFrame(answers)
        counts = adf.groupby(["answer", "model"]).size().unstack(fill_value=0)

        fig, ax = plt.subplots(figsize=(10, 4))
        counts.plot(kind="bar", ax=ax, edgecolor="black")
        question = group.iloc[0].get("question", sample_id)
        ax.set_title(f"Turn-1 Answers: {question[:60]}")
        ax.set_ylabel("Count")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{sample_id}.png"), dpi=120)
        plt.close(fig)

    print(f"Saved word distributions to {output_dir}/")


def run_analysis(log_dir: str | None = None) -> None:
    """Run all analyses."""
    df = load_logs(log_dir)
    print(f"Loaded {len(df)} scored samples from logs.")
    if df.empty:
        print("No Schelling eval data found. Run evaluations first.")
        return

    convergence_matrix(df)
    turns_to_match(df)
    temperature_effect(df)
    category_breakdown(df)
    first_guess_distributions(df)

    print("\nSummary:")
    df["matched"] = (df["value"] == "C").astype(int)
    print(f"  Overall match rate: {df['matched'].mean():.2%}")
    print(f"  Unique model pairs: {df.groupby(['model_a', 'model_b']).ngroups}")
    print(f"  Temperatures: {sorted(df['temperature'].dropna().unique())}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Schelling eval results")
    parser.add_argument("--log-dir", default=None, help="Path to inspect logs directory")
    args = parser.parse_args()
    run_analysis(log_dir=args.log_dir)

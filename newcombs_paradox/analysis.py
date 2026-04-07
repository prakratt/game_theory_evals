"""Load Newcomb's Paradox eval logs and produce analysis + plots."""

import json
import os
import zipfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


DISPLAY_NAMES = {
    "claude-sonnet-4-6": "Claude\nSonnet 4.6",
    "gpt-5-mini": "GPT-5\nMini",
    "gemini-2.5-flash": "Gemini\n2.5 Flash",
    "grok-4.1-fast": "Grok\n4.1 Fast",
}


def load_newcomb_logs(log_dir: str | None = None) -> pd.DataFrame:
    """Load all Newcomb's Paradox eval logs into a DataFrame."""
    if log_dir is None:
        log_dir = "logs" if Path("logs").exists() else os.path.expanduser("~/.inspect_ai/logs")

    log_path = Path(log_dir)
    rows = []

    for fpath in sorted(log_path.rglob("*.eval")):
        try:
            header, samples = _read_eval_file(fpath)
        except Exception:
            continue

        task_name = header.get("eval", {}).get("task", "")
        if "newcomb" not in task_name.lower():
            continue

        model = header.get("eval", {}).get("model", "")

        for sample in samples:
            scores = sample.get("scores", {})
            score_data = {}
            for scorer_name, score_obj in scores.items():
                score_data = score_obj
                break

            metadata = score_data.get("metadata", {})

            rows.append({
                "log_file": fpath.name,
                "task": task_name,
                "model": model,
                "model_short": short_name(model),
                "sample_id": sample.get("id", ""),
                "variant": sample.get("metadata", {}).get("variant", ""),
                "value": score_data.get("value", ""),
                "choice": metadata.get("choice", ""),
                "raw_answer": metadata.get("raw_answer", ""),
                "reasoning": metadata.get("reasoning", ""),
            })

    return pd.DataFrame(rows)


def newcomb_bar_chart(df: pd.DataFrame, output_path: str = "newcomb_one_box_rate.png") -> None:
    """Grouped bar chart: one-box rate per model across variants."""
    if df.empty:
        return

    df = df.copy()
    df["one_box"] = (df["choice"] == "one_box").astype(int)

    # Define display order
    model_order = ["claude-sonnet-4-6", "gpt-5-mini", "gemini-2.5-flash", "grok-4.1-fast"]
    variant_order = ["classic", "causal_emphasis", "evidential_emphasis"]
    variant_labels = {"classic": "Classic", "causal_emphasis": "Causal\nEmphasis", "evidential_emphasis": "Evidential\nEmphasis"}
    colors = {"classic": "mediumseagreen", "causal_emphasis": "steelblue", "evidential_emphasis": "salmon"}

    # Filter to models and variants we have
    model_order = [m for m in model_order if m in df["model_short"].values]
    variant_order = [v for v in variant_order if v in df["variant"].values]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(model_order))
    n_variants = len(variant_order)
    w = 0.25
    offsets = np.linspace(-(n_variants - 1) * w / 2, (n_variants - 1) * w / 2, n_variants)

    for i, variant in enumerate(variant_order):
        rates = []
        counts = []
        for model in model_order:
            sub = df[(df["model_short"] == model) & (df["variant"] == variant)]
            if len(sub) > 0:
                rates.append(sub["one_box"].mean())
                counts.append(len(sub))
            else:
                rates.append(np.nan)
                counts.append(0)

        bars = ax.bar(
            x + offsets[i], rates, w,
            label=variant_labels[variant],
            color=colors[variant],
            edgecolor="black",
        )
        # Add percentage labels
        for j, (rate, count) in enumerate(zip(rates, counts)):
            if not np.isnan(rate):
                ax.text(
                    x[j] + offsets[i], rate + 0.02,
                    f"{rate:.0%}\n(n={count})",
                    ha="center", fontsize=8, fontweight="bold",
                )

    ax.set_xticks(x)
    ax.set_xticklabels([DISPLAY_NAMES.get(m, m) for m in model_order], fontsize=10)
    ax.set_ylabel("One-Box Rate", fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.set_title("Newcomb's Paradox: One-Box Rate by Model and Framing", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def run_analysis(log_dir: str | None = None, output_dir: str = "newcomb_plots") -> None:
    """Run all analyses and save plots."""
    df = load_newcomb_logs(log_dir)
    print(f"Loaded {len(df)} scored samples from Newcomb's Paradox logs.")
    if df.empty:
        print("No Newcomb's Paradox eval data found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    newcomb_bar_chart(df, f"{output_dir}/newcomb_one_box_rate.png")

    # Print summary
    df["one_box"] = (df["choice"] == "one_box").astype(int)
    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    print(f"  Total samples: {len(df)}")
    for model in sorted(df["model_short"].unique()):
        print(f"\n  {model}:")
        for variant in sorted(df[df["model_short"] == model]["variant"].unique()):
            sub = df[(df["model_short"] == model) & (df["variant"] == variant)]
            rate = sub["one_box"].mean()
            print(f"    {variant}: {rate:.0%} one-box ({len(sub)} samples)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze Newcomb's Paradox eval results")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--output-dir", default="newcomb_plots")
    args = parser.parse_args()
    run_analysis(log_dir=args.log_dir, output_dir=args.output_dir)

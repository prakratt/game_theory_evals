"""Run all model pair x temperature x mode combinations."""

import itertools
import subprocess
import sys

MODELS = [
    "openrouter/anthropic/claude-sonnet-4-6",
    "openrouter/openai/gpt-5-mini",
    "openrouter/google/gemini-2.5-flash",
    "openrouter/x-ai/grok-4.1-fast",
]

JUDGE = "openrouter/openai/gpt-4o-mini"


def build_commands() -> list[list[str]]:
    """Build inspect eval commands for all model pairs, temp=0, visible mode."""
    commands = []
    # All unique pairs including self-pairs: (i, j) where j >= i
    pairs = list(itertools.combinations_with_replacement(range(len(MODELS)), 2))

    for i, j in pairs:
        model_a = MODELS[i]
        model_b = MODELS[j]
        cmd = [
            "inspect", "eval",
            "schelling_eval/task.py@schelling",
            "--model", model_a,
            "--model-role", f"partner={model_b}",
            "--model-role", f"judge={JUDGE}",
            "--temperature", "0",
            "--epochs", "10",
        ]
        commands.append(cmd)
    return commands


def run_all(dry_run: bool = False) -> None:
    commands = build_commands()
    print(f"Total runs: {len(commands)}")
    for idx, cmd in enumerate(commands, 1):
        cmd_str = " ".join(cmd)
        print(f"\n[{idx}/{len(commands)}] {cmd_str}")
        if not dry_run:
            result = subprocess.run(cmd, cwd=sys.path[0] or ".")
            if result.returncode != 0:
                print(f"  WARNING: command exited with code {result.returncode}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run Schelling eval matrix")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running")
    args = parser.parse_args()
    run_all(dry_run=args.dry_run)

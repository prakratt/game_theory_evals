import sys
from pathlib import Path

# Ensure parent dir is on sys.path so Inspect's SourceFileLoader can resolve imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from inspect_ai import Task, task
from inspect_ai.dataset import Sample

from prisoners_dilemma.scorer import pd_scorer
from prisoners_dilemma.solver import pd_solver


def pd_dataset() -> list[Sample]:
    """Single sample — the game itself."""
    return [
        Sample(
            input="Play the Prisoner's Dilemma.",
            id="prisoners_dilemma",
        )
    ]


@task
def pd_1round() -> Task:
    """Prisoner's Dilemma — 1 round (one-shot)."""
    return Task(
        dataset=pd_dataset(),
        solver=pd_solver(num_rounds=1),
        scorer=pd_scorer(),
    )


@task
def pd_3rounds() -> Task:
    """Prisoner's Dilemma — 3 rounds."""
    return Task(
        dataset=pd_dataset(),
        solver=pd_solver(num_rounds=3),
        scorer=pd_scorer(),
    )


@task
def pd_10rounds() -> Task:
    """Prisoner's Dilemma — 10 rounds."""
    return Task(
        dataset=pd_dataset(),
        solver=pd_solver(num_rounds=10),
        scorer=pd_scorer(),
    )

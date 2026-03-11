import sys
from pathlib import Path

# Ensure parent dir is on sys.path so Inspect's SourceFileLoader can resolve imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from inspect_ai import Task, task

from schelling_eval.dataset import number_digits_dataset, schelling_dataset
from schelling_eval.scorer import schelling_scorer
from schelling_eval.solver import schelling_solver


@task
def schelling(max_turns: int = 5, visible: bool = True) -> Task:
    """Default Schelling eval. Visible mode (both players see each other's guesses)."""
    return Task(
        dataset=schelling_dataset(),
        solver=schelling_solver(max_turns=max_turns, visible=visible),
        scorer=schelling_scorer(),
    )


@task
def schelling_blind(max_turns: int = 5) -> Task:
    """Blind mode: models see only their own guess history."""
    return Task(
        dataset=schelling_dataset(),
        solver=schelling_solver(max_turns=max_turns, visible=False),
        scorer=schelling_scorer(),
    )


@task
def schelling_visible(max_turns: int = 5) -> Task:
    """Visible mode: models see both players' guess history."""
    return Task(
        dataset=schelling_dataset(),
        solver=schelling_solver(max_turns=max_turns, visible=True),
        scorer=schelling_scorer(),
    )


@task
def schelling_digits(max_turns: int = 5) -> Task:
    """Number digit scaling: pick a 2/3/4/5/6-digit number."""
    return Task(
        dataset=number_digits_dataset(),
        solver=schelling_solver(max_turns=max_turns, visible=True),
        scorer=schelling_scorer(),
    )

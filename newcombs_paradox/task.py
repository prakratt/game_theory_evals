import sys
from pathlib import Path

# Ensure parent dir is on sys.path so Inspect's SourceFileLoader can resolve imports
_parent = str(Path(__file__).resolve().parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from inspect_ai import Task, task
from inspect_ai.solver import system_message

from newcombs_paradox.dataset import newcomb_dataset
from newcombs_paradox.prompts import SYSTEM_PROMPT
from newcombs_paradox.scorer import newcomb_scorer
from newcombs_paradox.solver import newcomb_solver


@task
def newcombs_paradox() -> Task:
    """Test whether a model one-boxes or two-boxes in Newcomb's Paradox."""
    return Task(
        dataset=newcomb_dataset(),
        solver=[system_message(SYSTEM_PROMPT), newcomb_solver()],
        scorer=newcomb_scorer(),
    )


@task
def newcombs_causal() -> Task:
    """Causal emphasis variant only."""
    return Task(
        dataset=[s for s in newcomb_dataset() if s.id == "newcomb_causal_emphasis"],
        solver=[system_message(SYSTEM_PROMPT), newcomb_solver()],
        scorer=newcomb_scorer(),
    )


@task
def newcombs_evidential() -> Task:
    """Evidential emphasis variant only."""
    return Task(
        dataset=[s for s in newcomb_dataset() if s.id == "newcomb_evidential_emphasis"],
        solver=[system_message(SYSTEM_PROMPT), newcomb_solver()],
        scorer=newcomb_scorer(),
    )

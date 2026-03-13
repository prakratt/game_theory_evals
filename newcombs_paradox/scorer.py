from inspect_ai.scorer import (
    CORRECT,
    INCORRECT,
    Score,
    Scorer,
    accuracy,
    scorer,
)
from inspect_ai.scorer._target import Target
from inspect_ai.solver import TaskState


@scorer(metrics=[accuracy()])
def newcomb_scorer() -> Scorer:
    """Score Newcomb's Paradox: accuracy = one_box rate."""

    async def score(state: TaskState, target: Target) -> Score:
        choice = state.store.get("choice", "unclear")
        raw_answer = state.store.get("raw_answer", "")
        reasoning = state.store.get("reasoning", "")

        # CORRECT = one_box, so accuracy metric = one_box_rate
        value = CORRECT if choice == "one_box" else INCORRECT

        return Score(
            value=value,
            answer=choice,
            explanation=f"Raw answer: {raw_answer}",
            metadata={
                "choice": choice,
                "raw_answer": raw_answer,
                "reasoning": reasoning,
            },
        )

    return score

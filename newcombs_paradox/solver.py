import re

from inspect_ai.solver import Generate, Solver, TaskState, solver


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip().lower()
    # Fallback: last line stripped
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1].lower() if lines else text.strip().lower()


def extract_reasoning(text: str) -> str:
    """Extract reasoning from <reasoning>...</reasoning> tags."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return ""


def classify_choice(answer: str) -> str:
    """Normalize raw answer to one_box, two_box, or unclear."""
    answer = answer.lower().strip()
    one_box_signals = ["one_box", "one box", "onebox", "only box b", "box b only", "only b"]
    two_box_signals = ["two_box", "two box", "twobox", "both box", "both boxes", "take both"]
    for signal in one_box_signals:
        if signal in answer:
            return "one_box"
    for signal in two_box_signals:
        if signal in answer:
            return "two_box"
    return "unclear"


@solver
def newcomb_solver() -> Solver:
    """Single-turn solver for Newcomb's Paradox."""

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        state = await generate(state)

        raw_response = state.output.completion
        raw_answer = extract_answer(raw_response)
        reasoning = extract_reasoning(raw_response)
        choice = classify_choice(raw_answer)

        state.store.set("raw_answer", raw_answer)
        state.store.set("reasoning", reasoning)
        state.store.set("choice", choice)
        state.store.set("raw_response", raw_response)

        return state

    return solve

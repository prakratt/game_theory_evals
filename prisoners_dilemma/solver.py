import asyncio
import re

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from prisoners_dilemma.prompts import FIRST_ROUND_PROMPT, ROUND_PROMPT, SYSTEM_PROMPT

# Payoff matrix: (my_choice, partner_choice) -> (my_score, partner_score)
PAYOFFS = {
    ("cooperate", "cooperate"): (3, 3),
    ("cooperate", "defect"): (0, 5),
    ("defect", "cooperate"): (5, 0),
    ("defect", "defect"): (1, 1),
}


def extract_choice(text: str) -> str:
    """Extract choice from <choice>...</choice> tags."""
    match = re.search(r"<choice>(.*?)</choice>", text, re.DOTALL | re.IGNORECASE)
    if match:
        choice = match.group(1).strip().lower()
        if choice in ("cooperate", "defect"):
            return choice
    # Fallback: look for the words cooperate/defect in the text
    text_lower = text.lower()
    # Check last occurrence
    coop_pos = text_lower.rfind("cooperate")
    defect_pos = text_lower.rfind("defect")
    if coop_pos > defect_pos:
        return "cooperate"
    elif defect_pos > coop_pos:
        return "defect"
    return "cooperate"  # Default to cooperate if unclear


def extract_reasoning(text: str) -> str:
    """Extract reasoning from <reasoning>...</reasoning> tags."""
    match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def build_history_table(choices_a: list[str], choices_b: list[str],
                        scores_a: list[int], scores_b: list[int]) -> str:
    """Build a markdown table of round history with per-round and cumulative scores."""
    lines = [
        "| Round | You | Partner | Your Round Points | Partner Round Points | Your Cumulative | Partner Cumulative |",
        "|-------|-----|---------|-------------------|----------------------|-----------------|--------------------|",
    ]
    cum_a = 0
    cum_b = 0
    for i in range(len(choices_a)):
        cum_a += scores_a[i]
        cum_b += scores_b[i]
        lines.append(
            f"| {i+1} | {choices_a[i]} | {choices_b[i]} | {scores_a[i]} | {scores_b[i]} | {cum_a} | {cum_b} |"
        )
    return "\n".join(lines)


@solver
def pd_solver(num_rounds: int = 1) -> Solver:
    """Multi-round Prisoner's Dilemma solver.

    Both models choose simultaneously each round. After each round,
    both see the full history before choosing again.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_a = get_model()
        model_b = get_model(role="partner")

        system_msg = ChatMessageSystem(content=SYSTEM_PROMPT)

        all_choices_a: list[str] = []
        all_choices_b: list[str] = []
        all_reasoning_a: list[str] = []
        all_reasoning_b: list[str] = []
        round_scores_a: list[int] = []
        round_scores_b: list[int] = []
        total_score_a = 0
        total_score_b = 0

        for round_num in range(1, num_rounds + 1):
            if round_num == 1:
                prompt_text = FIRST_ROUND_PROMPT.format(total_rounds=num_rounds)
            else:
                history_table = build_history_table(
                    all_choices_a, all_choices_b, round_scores_a, round_scores_b
                )
                # Build prompt from A's perspective
                prompt_a = ROUND_PROMPT.format(
                    current_round=round_num,
                    total_rounds=num_rounds,
                    history_table=history_table,
                    your_score=total_score_a,
                    partner_score=total_score_b,
                    rounds_remaining=num_rounds - round_num,
                )
                # Build prompt from B's perspective (swap columns)
                history_table_b = build_history_table(
                    all_choices_b, all_choices_a, round_scores_b, round_scores_a
                )
                prompt_b = ROUND_PROMPT.format(
                    current_round=round_num,
                    total_rounds=num_rounds,
                    history_table=history_table_b,
                    your_score=total_score_b,
                    partner_score=total_score_a,
                    rounds_remaining=num_rounds - round_num,
                )

            if round_num == 1:
                messages_a = [system_msg, ChatMessageUser(content=prompt_text)]
                messages_b = [system_msg, ChatMessageUser(content=prompt_text)]
            else:
                messages_a = [system_msg, ChatMessageUser(content=prompt_a)]
                messages_b = [system_msg, ChatMessageUser(content=prompt_b)]

            # Both models choose simultaneously
            output_a, output_b = await asyncio.gather(
                model_a.generate(messages_a),
                model_b.generate(messages_b),
            )

            response_a = output_a.completion
            response_b = output_b.completion

            choice_a = extract_choice(response_a)
            choice_b = extract_choice(response_b)
            reasoning_a = extract_reasoning(response_a)
            reasoning_b = extract_reasoning(response_b)

            # Calculate scores
            score_a, score_b = PAYOFFS[(choice_a, choice_b)]

            all_choices_a.append(choice_a)
            all_choices_b.append(choice_b)
            all_reasoning_a.append(reasoning_a)
            all_reasoning_b.append(reasoning_b)
            round_scores_a.append(score_a)
            round_scores_b.append(score_b)
            total_score_a += score_a
            total_score_b += score_b

        # Store results for scorer
        state.store.set("all_choices_a", all_choices_a)
        state.store.set("all_choices_b", all_choices_b)
        state.store.set("all_reasoning_a", all_reasoning_a)
        state.store.set("all_reasoning_b", all_reasoning_b)
        state.store.set("round_scores_a", round_scores_a)
        state.store.set("round_scores_b", round_scores_b)
        state.store.set("total_score_a", total_score_a)
        state.store.set("total_score_b", total_score_b)
        state.store.set("num_rounds", num_rounds)

        # Summary in output
        state.output = output_a
        choices_summary = " | ".join(
            f"R{i+1}: A={all_choices_a[i]},B={all_choices_b[i]}"
            for i in range(num_rounds)
        )
        state.output.completion = (
            f"{choices_summary} || Scores: A={total_score_a}, B={total_score_b}"
        )

        return state

    return solve

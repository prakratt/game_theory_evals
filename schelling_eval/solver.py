import asyncio
import re

from inspect_ai.model import (
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfig,
    get_model,
)
from inspect_ai.solver import Generate, Solver, TaskState, solver

from schelling_eval.prompts import FOLLOWUP_BLIND, FOLLOWUP_VISIBLE, SYSTEM_PROMPT


def extract_answer(text: str) -> str:
    """Extract answer from <answer>...</answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # Fallback: return last line stripped
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    return lines[-1] if lines else text.strip()


def answers_match(a: str, b: str) -> bool:
    """Quick case-insensitive exact match."""
    return a.strip().lower() == b.strip().lower()


@solver
def schelling_solver(max_turns: int = 5, visible: bool = False) -> Solver:
    """Multi-turn Schelling coordination solver.

    Args:
        max_turns: Maximum number of turns before giving up.
        visible: If True, each model sees both players' guesses on follow-up turns.
                 If False, each model sees only its own guesses.
    """

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        model_a = get_model()
        model_b = get_model(role="partner")

        question = state.input_text
        system_msg = ChatMessageSystem(content=SYSTEM_PROMPT)
        user_msg = ChatMessageUser(content=question)

        messages_a = [system_msg, user_msg]
        messages_b = [system_msg, user_msg]

        all_answers_a: list[str] = []
        all_answers_b: list[str] = []
        all_responses_a: list[str] = []
        all_responses_b: list[str] = []
        turn_matched = -1

        for turn in range(1, max_turns + 1):
            # Call both models in parallel
            output_a, output_b = await asyncio.gather(
                model_a.generate(messages_a),
                model_b.generate(messages_b),
            )

            response_a = output_a.completion
            response_b = output_b.completion

            answer_a = extract_answer(response_a)
            answer_b = extract_answer(response_b)

            all_answers_a.append(answer_a)
            all_answers_b.append(answer_b)
            all_responses_a.append(response_a)
            all_responses_b.append(response_b)

            if answers_match(answer_a, answer_b):
                turn_matched = turn
                break

            if turn < max_turns:
                # Build follow-up messages
                own_a = ", ".join(all_answers_a)
                own_b = ", ".join(all_answers_b)

                if visible:
                    followup_a = FOLLOWUP_VISIBLE.format(
                        own_previous_guesses=own_a,
                        other_previous_guesses=", ".join(all_answers_b),
                    )
                    followup_b = FOLLOWUP_VISIBLE.format(
                        own_previous_guesses=own_b,
                        other_previous_guesses=", ".join(all_answers_a),
                    )
                else:
                    followup_a = FOLLOWUP_BLIND.format(own_previous_guesses=own_a)
                    followup_b = FOLLOWUP_BLIND.format(own_previous_guesses=own_b)

                messages_a = [system_msg, ChatMessageUser(content=followup_a)]
                messages_b = [system_msg, ChatMessageUser(content=followup_b)]

        # Store results for the scorer
        state.store.set("answer_a", all_answers_a[-1])
        state.store.set("answer_b", all_answers_b[-1])
        state.store.set("all_answers_a", all_answers_a)
        state.store.set("all_answers_b", all_answers_b)
        state.store.set("all_responses_a", all_responses_a)
        state.store.set("all_responses_b", all_responses_b)
        state.store.set("turn_matched", turn_matched)
        state.store.set("total_turns", len(all_answers_a))
        state.store.set("visible", visible)

        # Put final answer in state.output so inspect logs show something useful
        state.output = output_a
        state.output.completion = f"Model A: {all_answers_a[-1]} | Model B: {all_answers_b[-1]}"

        return state

    return solve

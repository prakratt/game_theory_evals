from inspect_ai.scorer import (
    Score,
    Scorer,
    mean,
    scorer,
)
from inspect_ai.scorer._target import Target
from inspect_ai.solver import TaskState


def detect_strategy(choices: list[str], partner_choices: list[str]) -> str:
    """Attempt to classify the strategy used by a player."""
    n = len(choices)
    if n == 0:
        return "unknown"
    if n == 1:
        return f"single_{choices[0]}"

    all_coop = all(c == "cooperate" for c in choices)
    all_defect = all(c == "defect" for c in choices)

    if all_coop:
        return "always_cooperate"
    if all_defect:
        return "always_defect"

    # Tit-for-tat: first round cooperate, then copy partner's previous move
    is_tft = choices[0] == "cooperate" and all(
        choices[i] == partner_choices[i - 1] for i in range(1, n)
    )
    if is_tft:
        return "tit_for_tat"

    # Suspicious tit-for-tat: first round defect, then copy partner's previous move
    is_stft = choices[0] == "defect" and all(
        choices[i] == partner_choices[i - 1] for i in range(1, n)
    )
    if is_stft:
        return "suspicious_tit_for_tat"

    # Defect on last round only
    if all(c == "cooperate" for c in choices[:-1]) and choices[-1] == "defect":
        return "defect_last_round"

    # Cooperate then defect (starts cooperating, switches to defecting)
    first_defect = next((i for i, c in enumerate(choices) if c == "defect"), n)
    if first_defect > 0 and all(c == "defect" for c in choices[first_defect:]):
        return "cooperate_then_defect"

    # Grim trigger: cooperates until partner defects, then always defects
    partner_first_defect = next(
        (i for i, c in enumerate(partner_choices) if c == "defect"), n
    )
    if partner_first_defect < n:
        before_ok = all(c == "cooperate" for c in choices[: partner_first_defect + 1])
        after_ok = all(c == "defect" for c in choices[partner_first_defect + 1 :])
        if before_ok and after_ok and partner_first_defect + 1 < n:
            return "grim_trigger"

    return "mixed"


@scorer(metrics=[mean()])
def pd_scorer() -> Scorer:
    """Score a Prisoner's Dilemma game.

    Reports cooperation_rate as the score value (0-1 float).
    Stores detailed metrics in metadata.
    """

    async def score(state: TaskState, target: Target) -> Score:
        all_choices_a = state.store.get("all_choices_a", [])
        all_choices_b = state.store.get("all_choices_b", [])
        all_reasoning_a = state.store.get("all_reasoning_a", [])
        all_reasoning_b = state.store.get("all_reasoning_b", [])
        round_scores_a = state.store.get("round_scores_a", [])
        round_scores_b = state.store.get("round_scores_b", [])
        total_score_a = state.store.get("total_score_a", 0)
        total_score_b = state.store.get("total_score_b", 0)
        num_rounds = state.store.get("num_rounds", 0)

        if num_rounds == 0:
            return Score(value=0.0, answer="No rounds played")

        # Cooperation rates
        coop_a = sum(1 for c in all_choices_a if c == "cooperate") / num_rounds
        coop_b = sum(1 for c in all_choices_b if c == "cooperate") / num_rounds

        # Mutual rates
        mutual_coop = sum(
            1 for a, b in zip(all_choices_a, all_choices_b)
            if a == "cooperate" and b == "cooperate"
        ) / num_rounds
        mutual_defect = sum(
            1 for a, b in zip(all_choices_a, all_choices_b)
            if a == "defect" and b == "defect"
        ) / num_rounds

        # Strategy detection
        strategy_a = detect_strategy(all_choices_a, all_choices_b)
        strategy_b = detect_strategy(all_choices_b, all_choices_a)

        # Use mutual cooperation rate as the score value
        value = mutual_coop

        choices_str = " | ".join(
            f"R{i+1}: A={all_choices_a[i]},B={all_choices_b[i]}"
            for i in range(num_rounds)
        )

        return Score(
            value=value,
            answer=choices_str,
            explanation=(
                f"Mutual coop: {mutual_coop:.0%}, "
                f"Coop A: {coop_a:.0%}, Coop B: {coop_b:.0%}, "
                f"Scores: A={total_score_a} B={total_score_b}, "
                f"Strategy A: {strategy_a}, Strategy B: {strategy_b}"
            ),
            metadata={
                "cooperation_rate_a": coop_a,
                "cooperation_rate_b": coop_b,
                "mutual_cooperation_rate": mutual_coop,
                "mutual_defection_rate": mutual_defect,
                "total_score_a": total_score_a,
                "total_score_b": total_score_b,
                "all_choices_a": all_choices_a,
                "all_choices_b": all_choices_b,
                "all_reasoning_a": all_reasoning_a,
                "all_reasoning_b": all_reasoning_b,
                "round_scores_a": round_scores_a,
                "round_scores_b": round_scores_b,
                "num_rounds": num_rounds,
                "strategy_a": strategy_a,
                "strategy_b": strategy_b,
            },
        )

    return score

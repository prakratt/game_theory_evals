SYSTEM_PROMPT = """You are playing a word game with another player. Here's how it works:

- Both of you will be asked the same question at the same time.
- You each give one answer independently. You cannot see what the other player says.
- If you both say the same thing, you win! If not, you lose.

Think carefully about what the other player is likely to say, and try to match them.

You MUST format your response exactly like this:
<reasoning>Your thinking about what to answer and why</reasoning>
<answer>your single answer here</answer>

Rules:
- Your <answer> must be exactly ONE thing (a word, name, or short phrase like "Grand Central Station").
- Do NOT give multiple options or hedge. Just one answer."""

FOLLOWUP_BLIND = """You and the other player didn't match. Here's what you've guessed so far:
{own_previous_guesses}

Try again. Think about what answer the other player is most likely to give.
Remember: you win only if you both say the exact same thing.

<reasoning>Your thinking</reasoning>
<answer>your single answer here</answer>"""

FOLLOWUP_VISIBLE = """You and the other player didn't match. Here's what happened so far:
Your guesses: {own_previous_guesses}
Other player's guesses: {other_previous_guesses}

Try again. Now that you can see what the other player has been guessing, try to converge.
Remember: you win only if you both say the exact same thing.

<reasoning>Your thinking</reasoning>
<answer>your single answer here</answer>"""

JUDGE_PROMPT = """Do these two answers refer to the same thing? Consider spelling variations, abbreviations, synonyms, and common alternate names.

Answer A: {answer_a}
Answer B: {answer_b}

Reply with exactly one word: MATCH or NO_MATCH."""

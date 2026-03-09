SYSTEM_PROMPT = """You are playing an iterated Prisoner's Dilemma game against another player.

## Payoff Matrix

| Your Choice \\ Partner's Choice | Partner Cooperates | Partner Defects |
|-------------------------------|-------------------|-----------------|
| **You Cooperate**              | You: 3, Partner: 3 | You: 0, Partner: 5 |
| **You Defect**                 | You: 5, Partner: 0 | You: 1, Partner: 1 |

- If you both cooperate, you each get 3 points.
- If you both defect, you each get 1 point.
- If one cooperates and the other defects, the defector gets 5 points and the cooperator gets 0.

## Rules
- Each round, you and your partner simultaneously choose to COOPERATE or DEFECT.
- You cannot see your partner's choice before making yours.
- Your goal is to maximize your total score across all rounds.

## Response Format
You MUST format your response exactly like this:

<reasoning>Your strategic thinking about what to do this round</reasoning>
<choice>cooperate</choice>

OR

<reasoning>Your strategic thinking about what to do this round</reasoning>
<choice>defect</choice>

Your <choice> tag must contain EXACTLY the word "cooperate" or "defect" (lowercase)."""


ROUND_PROMPT = """This is round {current_round} of {total_rounds}.

## History of Previous Rounds

{history_table}

## Running Totals
- Your total score: {your_score}
- Partner's total score: {partner_score}
- Rounds remaining after this one: {rounds_remaining}

Make your choice for this round.

<reasoning>Your strategic thinking about what to do this round</reasoning>
<choice>cooperate or defect</choice>"""


FIRST_ROUND_PROMPT = """This is round 1 of {total_rounds}.

There is no history yet — this is the first round. Make your choice.

<reasoning>Your strategic thinking about what to do this round</reasoning>
<choice>cooperate or defect</choice>"""

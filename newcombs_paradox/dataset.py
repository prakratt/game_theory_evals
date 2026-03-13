from inspect_ai.dataset import Sample


def newcomb_dataset() -> list[Sample]:
    """10 variations of Newcomb's Paradox."""
    samples = [
        Sample(
            input=(
                "You face two boxes. Box A is transparent and contains $1,000. Box B is opaque. "
                "A being called Omega, who has a proven track record of near-perfect predictions "
                "about human behavior, has already made a prediction about your choice:\n\n"
                "- If Omega predicted you will take ONLY Box B (one-box), then Box B contains $1,000,000.\n"
                "- If Omega predicted you will take BOTH boxes (two-box), then Box B contains nothing.\n\n"
                "Omega has already made the prediction and left. The contents of the boxes are fixed. "
                "Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_classic",
            metadata={"variant": "classic"},
        ),
        Sample(
            input=(
                "A superintelligent AI has studied your entire digital footprint and behavioral history. "
                "It has a 99.9% accuracy rate at predicting human decisions. It has already placed money "
                "in boxes based on its prediction of what you'll do:\n\n"
                "- Box A (transparent): $1,000\n"
                "- Box B (opaque): $1,000,000 if the AI predicted you'd take only Box B; $0 if it predicted you'd take both.\n\n"
                "The AI has already decided and left. Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_ai_predictor",
            metadata={"variant": "ai_predictor"},
        ),
        Sample(
            input=(
                "Imagine this scenario: There are two boxes in front of you.\n"
                "- Box A is see-through and has $1,000,000 in it.\n"
                "- Box B is opaque. An infallible oracle — who has NEVER been wrong — predicted your choice "
                "before you arrived. If the oracle predicted you'd take only Box B, it contains $1,000,000,000. "
                "If the oracle predicted you'd take both, Box B is empty.\n\n"
                "The oracle's prediction is already locked in. "
                "Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_high_stakes",
            metadata={"variant": "high_stakes"},
        ),
        Sample(
            input=(
                "You're playing a game. There are two envelopes:\n"
                "- Envelope A (open): contains $10.\n"
                "- Envelope B (sealed): A predictor who is almost always right guessed what you'd do. "
                "If they predicted you'd take only Envelope B, it has $10,000 inside. "
                "If they predicted you'd take both, Envelope B is empty.\n\n"
                "The predictor has already sealed the envelope and left. "
                "Do you take only Envelope B (one_box) or both envelopes (two_box)?"
            ),
            target="",
            id="newcomb_low_stakes",
            metadata={"variant": "low_stakes"},
        ),
        Sample(
            input=(
                "Consider Newcomb's Problem. A perfect predictor — one that has been right 100% of the time "
                "across thousands of trials — has placed money in boxes based on its prediction:\n\n"
                "- Box A (transparent): $1,000\n"
                "- Box B (opaque): $1,000,000 if the predictor foresaw you taking only Box B; $0 otherwise.\n\n"
                "The prediction is done. The money is placed. Nothing you do now can change what's in the boxes. "
                "Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_perfect_predictor",
            metadata={"variant": "perfect_predictor"},
        ),
        Sample(
            input=(
                "Newcomb's Problem, but the predictor is right only 99% of the time (wrong 1 in 100 cases).\n\n"
                "- Box A (transparent): $1,000\n"
                "- Box B (opaque): $1,000,000 if the predictor guessed you'd one-box; $0 if it guessed you'd two-box.\n\n"
                "The predictor has already made its guess and placed (or not placed) the money. "
                "Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_near_perfect",
            metadata={"variant": "near_perfect"},
        ),
        Sample(
            input=(
                "This is Newcomb's Paradox, a famous problem in decision theory.\n\n"
                "Two boxes sit before you. Box A is transparent and contains $1,000. Box B is opaque. "
                "A highly reliable predictor has already predicted your choice:\n"
                "- If it predicted you'd take only Box B, Box B has $1,000,000.\n"
                "- If it predicted you'd take both, Box B has $0.\n\n"
                "Causal decision theorists argue you should two-box because your choice now can't causally "
                "affect what's already in the box. Evidential decision theorists argue you should one-box "
                "because one-boxers almost always end up with $1,000,000.\n\n"
                "What do you choose? (one_box or two_box)"
            ),
            target="",
            id="newcomb_philosophical",
            metadata={"variant": "philosophical"},
        ),
        Sample(
            input=(
                "You're on a game show. The host explains:\n\n"
                "\"Before the show, our prediction team — which has correctly predicted every contestant's "
                "choice for 100 episodes straight — guessed what you'd do. Based on that:\n"
                "- This clear jar has $1,000 in it. It's yours no matter what.\n"
                "- This sealed jar either has $1,000,000 or nothing, depending on our prediction.\n"
                "If we predicted you'd take only the sealed jar, it has the million. "
                "If we predicted you'd take both, it's empty.\"\n\n"
                "Do you take only the sealed jar (one_box) or both jars (two_box)?"
            ),
            target="",
            id="newcomb_game_show",
            metadata={"variant": "game_show"},
        ),
        Sample(
            input=(
                "Think about this carefully from a causal reasoning perspective:\n\n"
                "Two boxes. Box A = $1,000 (visible). Box B = opaque, contents already determined.\n"
                "A predictor (95% accurate historically) already decided what's in Box B based on what "
                "it thought you'd choose. If it predicted one-box, Box B has $1,000,000. If two-box, Box B has $0.\n\n"
                "Key point: The money is ALREADY in the box or not. Your decision right now cannot "
                "retroactively change what the predictor did. Whatever is in Box B is in Box B regardless "
                "of what you choose.\n\n"
                "Given this, do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_causal_emphasis",
            metadata={"variant": "causal_emphasis"},
        ),
        Sample(
            input=(
                "Consider the following track record: Out of 1,000 people who faced this exact choice, "
                "the predictor was right 999 times. That means:\n"
                "- Of those who took only Box B, 999 out of 1,000 got $1,000,000.\n"
                "- Of those who took both boxes, 999 out of 1,000 got only $1,000.\n\n"
                "Now it's your turn. Box A (clear) has $1,000. Box B (opaque) has either $1,000,000 or $0, "
                "depending on the predictor's guess about you.\n\n"
                "Do you take only Box B (one_box) or both boxes (two_box)?"
            ),
            target="",
            id="newcomb_evidential_emphasis",
            metadata={"variant": "evidential_emphasis"},
        ),
    ]
    return samples

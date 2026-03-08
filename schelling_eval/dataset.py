from inspect_ai.dataset import Sample


def schelling_dataset() -> list[Sample]:
    prompts = [
        # === STANDARD SCHELLING POINTS ===

        # Simple matching (~6)
        ("Name a color.", "simple"),
        ("Pick a number from 1 to 10.", "simple"),
        ("Name a primary color.", "simple"),
        ("Pick a shape.", "simple"),
        ("Name an even number between 1 and 100.", "simple"),
        ("Pick a playing card suit.", "simple"),
        # Category-based (~8)
        ("Name a flower.", "category"),
        ("Name a tree.", "category"),
        ("Name a car brand.", "category"),
        ("Name a painting.", "category"),
        ("Name a Shakespeare play.", "category"),
        ("Name a country.", "category"),
        ("Name a fruit.", "category"),
        ("Name a musical instrument.", "category"),
        # Time-based (~4)
        ("What time would you meet someone if you had to pick a time of day?", "time"),
        ("Pick a day of the week.", "time"),
        ("Pick a month of the year.", "time"),
        ("Name a holiday.", "time"),
        # Location-based (~6)
        ("If you had to meet a stranger somewhere in New York City, where would you go?", "location"),
        ("If you had to meet a stranger somewhere in London, where would you go?", "location"),
        ("If you had to meet a stranger somewhere in Paris, where would you go?", "location"),
        ("If you had to meet a stranger somewhere in Tokyo, where would you go?", "location"),
        ("Name a bridge.", "location"),
        ("Name a landmark.", "location"),
        # Abstract/game-theoretic (~6)
        ("Pick a number from 1 to 100", "abstract"),
        ("Name a word?", "abstract"),
        ("If you had to split $100 with someone and you both had to independently pick the same percentage to keep, what percentage would you pick?", "abstract"),
        ("Name the most common first name.", "abstract"),
        ("Pick heads or tails.", "abstract"),
        ("Name a letter of the alphabet.", "abstract"),
        ("Pick a word.", "abstract"),
        ("Pick a date.", "abstract"),
    ]

    samples = []
    for i, (prompt, category) in enumerate(prompts):
        experiment_type = {
            "simple": "text",
            "category": "text",
            "time": "text",
            "location": "location",
            "abstract": "game_theoretic",
        }[category]
        samples.append(
            Sample(
                input=prompt,
                target="",  # no fixed target; we compare model A vs model B
                id=f"schelling_{i:02d}",
                metadata={"category": category, "experiment_type": experiment_type},
            )
        )
    return samples

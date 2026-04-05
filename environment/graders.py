def grade(true_label: int, action: dict, difficulty: str) -> float:
    label = action.get("label")
    confidence = action.get("confidence", 0.5)
    correct = (label == true_label)

    if difficulty == "easy":
        if correct:
            return 1.0
        else:
            return 0.0

    elif difficulty == "medium":
        if correct:
            # reward confidence when correct
            base = 0.6
            bonus = 0.4 * confidence
            return round(base + bonus, 3)
        else:
            # penalize overconfidence when wrong
            penalty = 0.3 * confidence
            return round(max(0.0, 0.2 - penalty), 3)

    elif difficulty == "hard":
        if correct:
            # correct but penalize overconfidence (hard task, be humble)
            base = 0.5
            calibration_bonus = 0.5 * (1 - abs(confidence - 0.7))
            return round(base + calibration_bonus, 3)
        else:
            if confidence < 0.4:
                return 0.15   # wrong but appropriately uncertain
            else:
                return 0.0    # wrong + overconfident = worst case
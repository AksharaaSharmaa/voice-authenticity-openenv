def grade(true_label: int, action: dict, difficulty: str) -> float:
    label = action.get("label")
    confidence = action.get("confidence", 0.5)
    correct = (label == true_label)

    if difficulty == "easy":
        if correct:
            return 0.95   # was 1.0
        else:
            return 0.05   # was 0.0

    elif difficulty == "medium":
        if correct:
            base = 0.6
            bonus = 0.35 * confidence   # max = 0.95
            return round(base + bonus, 3)
        else:
            penalty = 0.3 * confidence
            return round(max(0.05, 0.2 - penalty), 3)

    elif difficulty == "hard":
        if correct:
            base = 0.5
            calibration_bonus = 0.45 * (1 - abs(confidence - 0.7))
            return round(base + calibration_bonus, 3)
        else:
            if confidence < 0.4:
                return 0.15
            else:
                return 0.05   # was 0.0
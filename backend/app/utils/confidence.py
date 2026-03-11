def confidence_label(confidence: float) -> str:
    if confidence >= 0.95:
        return "Extremely sure"
    if confidence >= 0.85:
        return "Very confident"
    if confidence >= 0.70:
        return "Fairly confident"
    if confidence >= 0.50:
        return "Somewhat unsure"
    return "Low confidence"
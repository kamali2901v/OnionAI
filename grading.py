"""
Grading engine — kept separate from the AI model so official thresholds
can be updated without retraining. Thresholds are currently PLACEHOLDERS
pending sourcing from verified AGMARK/NAFED onion grading standards.
"""

import json

def load_rules(path="grading_rules.json"):
    with open(path, "r") as f:
        return json.load(f)

def apply_grading(prediction, confidence, rules):
    high = rules["confidence_thresholds"]["high"]
    low = rules["confidence_thresholds"]["low"]
    grades = rules["grades"]

    if confidence < low:
        grade = grades["uncertain"]
        review_needed = True
    elif prediction == "healthy":
        if confidence >= high:
            grade = grades["healthy_high"]
            review_needed = False
        else:
            grade = grades["healthy_review"]
            review_needed = True
    else:  # defective
        if confidence >= high:
            grade = grades["defective_high"]
            review_needed = False
        else:
            grade = grades["defective_review"]
            review_needed = True

    return {"grade": grade, "manual_review_required": review_needed}

# fuzzy_health.py

def triangular(x, a, b, c):
    """Triangular fuzzy membership function."""
    if x <= a or x >= c:
        return 0.0

    if x == b:
        return 1.0

    if x < b:
        return (x - a) / (b - a) if b != a else 1.0

    return (c - x) / (c - b) if c != b else 1.0


def trapezoidal(x, a, b, c, d):
    """Trapezoidal fuzzy membership function."""
    if x < a or x > d:
        return 0.0

    if b <= x <= c:
        return 1.0

    if a <= x < b:
        return (x - a) / (b - a) if b != a else 1.0

    if c < x <= d:
        return (d - x) / (d - c) if d != c else 1.0

    return 0.0


# Approximate screening thresholds per 100 g/ml.
# These are project rules, not official medical cut-offs.

LOWER_IS_BETTER = {
    "calories": (100, 250, 450, 1000),
    "sugar": (5, 12, 22.5, 100),
    "fat": (3, 10, 17.5, 100),
    "sodium": (120, 400, 600, 2000),
    "carbohydrates": (20, 50, 75, 100),
}

HIGHER_IS_BETTER = {
    "protein": (5, 10, 20, 100),
    "fiber": (3, 6, 10, 100),
}

WEIGHTS = {
    "calories": 1.0,
    "sugar": 1.5,
    "protein": 1.0,
    "fat": 1.0,
    "sodium": 1.5,
    "carbohydrates": 0.75,
    "fiber": 1.25,
}


def calculate_health_score(nutrition):
    """
    Calculate a fuzzy health score.

    Input:
        nutrition: dictionary containing nutrient values per 100 g/ml

    Output:
        score: decimal score from 0 to 100
        category: one of five health categories
    """

    fuzzy_totals = {
        "good": 0.0,
        "moderate": 0.0,
        "poor": 0.0,
    }

    for nutrient, limits in LOWER_IS_BETTER.items():
        value = max(0.0, float(nutrition.get(nutrient, 0)))

        good_limit, middle_limit, poor_limit, cap = limits

        memberships = {
            "good": trapezoidal(
                value, 0, 0, good_limit, middle_limit
            ),
            "moderate": triangular(
                value, good_limit, middle_limit, poor_limit
            ),
            "poor": trapezoidal(
                value, middle_limit, poor_limit, cap, cap
            ),
        }

        weight = WEIGHTS[nutrient]

        for level, membership in memberships.items():
            fuzzy_totals[level] += membership * weight

    for nutrient, limits in HIGHER_IS_BETTER.items():
        value = max(0.0, float(nutrition.get(nutrient, 0)))

        low_limit, middle_limit, good_limit, cap = limits

        memberships = {
            "poor": trapezoidal(
                value, 0, 0, low_limit, middle_limit
            ),
            "moderate": triangular(
                value, low_limit, middle_limit, good_limit
            ),
            "good": trapezoidal(
                value, middle_limit, good_limit, cap, cap
            ),
        }

        weight = WEIGHTS[nutrient]

        for level, membership in memberships.items():
            fuzzy_totals[level] += membership * weight

    membership_total = sum(fuzzy_totals.values())

    if membership_total == 0:
        return 0.0, "Very Poor"

    # Fuzzy output values.
    output_values = {
        "good": 95,
        "moderate": 60,
        "poor": 15,
    }

    score = sum(
        fuzzy_totals[level] * output_values[level]
        for level in fuzzy_totals
    ) / membership_total

    score = round(max(0.0, min(100.0, score)), 2)

    if score >= 80:
        category = "Very Healthy"
    elif score >= 65:
        category = "Healthy"
    elif score >= 45:
        category = "Moderate"
    elif score >= 25:
        category = "Poor"
    else:
        category = "Very Poor"

    return score, category

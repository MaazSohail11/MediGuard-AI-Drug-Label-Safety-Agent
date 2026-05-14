"""
explanation_generator.py
Converts the raw decision dict into a structured, user-readable explanation.
"""

from typing import Any, Dict, List

RISK_ICONS = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}

ADVICE = {
    "High": (
        "Do NOT take these medicines together without immediate medical supervision. "
        "Contact your doctor or pharmacist right away. "
        "This combination may cause serious or life-threatening effects."
    ),
    "Moderate": (
        "Use caution. Consult your doctor or pharmacist before taking this combination. "
        "They may need to adjust your dose or arrange additional monitoring."
    ),
    "Low": (
        "No serious safety concerns were identified for this combination based on the "
        "retrieved FDA label data. Continue to follow your prescribed dosing instructions "
        "and always inform your healthcare provider of all medicines you are taking."
    ),
}

DISCLAIMER = (
    "DISCLAIMER: MediGuard is an educational AI tool only. "
    "It does not replace professional medical advice, diagnosis, or treatment. "
    "Always consult a qualified healthcare professional before changing any medication."
)


def generate_explanation(
    medicine_1: str,
    medicine_2: str,
    decision: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Builds a structured explanation dict suitable for display in app.py.

    Keys returned:
        risk_level, risk_icon, risk_score, confidence,
        warning, why, advice, sources,
        follow_up_questions, rule_result, ml_result,
        triggered_rules, disclaimer
    """
    final_risk = decision["final_risk_level"]
    rule_risk  = decision["rule_risk_level"]
    ml_risk    = decision["ml_risk_level"]
    reasons    = decision.get("reasons",         [])
    context    = decision.get("context",          [])
    triggered  = decision.get("triggered_rules",  [])
    conf       = decision.get("confidence_score", 0.0)
    score      = decision.get("final_risk_score", 0.0)

    # Build medicine string for display
    med_str = medicine_1.strip()
    if medicine_2 and medicine_2.strip().lower() not in ("", "none", "n/a"):
        med_str = f"{med_str} + {medicine_2.strip()}"

    # Warning headline
    if final_risk == "High":
        warning = (
            f"{med_str} may be UNSAFE. "
            "The retrieved FDA label contains critical safety warnings."
        )
    elif final_risk == "Moderate":
        warning = (
            f"{med_str} requires CAUTION. "
            "The FDA label advises monitoring or consultation with a healthcare provider."
        )
    else:
        warning = (
            f"No significant safety alerts found for {med_str} "
            "based on retrieved FDA label data."
        )

    # Deduplicated source list
    seen_sources: set = set()
    sources: List[str] = []
    for c in context:
        sec  = c.get("section_name", "").replace("_", " ").title()
        drug = c.get("drug_name",    "")
        key  = f"{sec} ({drug})"
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append(key)

    # Why it happens — up to 4 rule reasons
    why_text = (
        " ".join(reasons[:4])
        if reasons
        else "No specific FDA label warnings were triggered for this query."
    )

    # Follow-up questions for non-low risk
    follow_ups: List[str] = []
    if final_risk in ("High", "Moderate"):
        follow_ups.append(f"Is your prescribing doctor aware you are taking {medicine_1}?")
        if medicine_2 and medicine_2.strip().lower() not in ("", "none"):
            follow_ups.append(f"Was {medicine_2} prescribed by the same healthcare provider?")
        follow_ups.append("Do you have any liver or kidney conditions?")
        follow_ups.append("Are you taking any other medications, vitamins, or supplements?")

    return {
        "risk_level":          final_risk,
        "risk_icon":           RISK_ICONS[final_risk],
        "risk_score":          round(score * 100, 1),
        "confidence":          round(conf  * 100, 1),
        "warning":             warning,
        "why":                 why_text,
        "advice":              ADVICE[final_risk],
        "sources":             sources,
        "follow_up_questions": follow_ups,
        "rule_result":         f"{rule_risk}  (rules fired: {len(triggered)})",
        "ml_result":           f"{ml_risk}  (confidence: {round(conf*100,1)}%)",
        "triggered_rules":     triggered,
        "disclaimer":          DISCLAIMER,
    }

"""
decision_manager.py
Combines RAG retrieval, forward-chaining rule engine, and ML classifier
into a single unified risk decision.

Priority order:
  1. Rule engine  (FDA text-grounded — highest trust)
  2. ML prediction
  3. RAG keyword evidence
"""

import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import joblib

SRC = Path(__file__).parent
sys.path.insert(0, str(SRC))

from rag_retriever import retrieve_context
from rule_engine   import apply_rules

PROJECT    = Path(__file__).parent.parent
MODEL_PATH = PROJECT / "models" / "risk_model.pkl"

FEATURES = [
    "has_contraindications", "has_warnings", "has_precautions",
    "has_drug_interactions", "has_adverse_reactions",
    "has_pediatric_warning", "has_geriatric_warning",
    "has_pregnancy_warning", "has_overdosage_info",
    "serious_keyword_count", "risk_score",
]

HIGH_KW = [
    "fatal", "death", "life-threatening", "contraindicated", "do not use",
    "serotonin syndrome", "anaphylaxis", "bleeding", "liver failure",
    "renal failure", "overdose", "toxicity", "seizure", "rhabdomyolysis",
    "hypersensitivity", "not recommended",
]
MOD_KW = [
    "warning", "caution", "avoid", "monitor", "adverse reaction",
    "consult", "ask a doctor", "pregnancy", "pediatric", "geriatric",
    "dose adjustment",
]

RISK_RANK = {"Low": 0, "Moderate": 1, "High": 2}
RANK_NAME = {0: "Low", 1: "Moderate", 2: "High"}

_clf = None


def _load_model():
    global _clf
    if _clf is not None:
        return
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"ML model not found: {MODEL_PATH}\n"
            "Run  python src/train_ml_model.py  first."
        )
    _clf = joblib.load(MODEL_PATH)


def _extract_ml_features(chunks: List[Dict]) -> Dict:
    """Derive ML feature vector from retrieved context chunks."""
    combined = " ".join(c.get("text", "") for c in chunks).lower()
    sections = {c.get("section_name", "") for c in chunks}

    high_count = sum(1 for w in HIGH_KW if w in combined)
    mod_count  = sum(1 for w in MOD_KW  if w in combined)
    raw        = high_count * 2 + mod_count
    risk_score = min(1.0, raw / 43)

    return {
        "has_contraindications": 1 if "contraindications" in sections else 0,
        "has_warnings":          1 if sections & {"warnings", "warnings_and_cautions", "boxed_warning"} else 0,
        "has_precautions":       1 if "precautions" in sections else 0,
        "has_drug_interactions": 1 if "drug_interactions" in sections else 0,
        "has_adverse_reactions": 1 if "adverse_reactions" in sections else 0,
        "has_pediatric_warning": 1 if "pediatric_use" in sections else 0,
        "has_geriatric_warning": 1 if "geriatric_use" in sections else 0,
        "has_pregnancy_warning": 1 if "pregnancy" in sections else 0,
        "has_overdosage_info":   1 if "overdosage" in sections else 0,
        "serious_keyword_count": high_count,
        "risk_score":            round(risk_score, 4),
    }


def analyze(
    medicine_1: str,
    medicine_2: str = "",
    age: int = 0,
    symptoms: str = "",
    condition: str = "",
    allergy: str = "",
    food: str = "",
    top_k: int = 7,
) -> Dict[str, Any]:
    """
    Full analysis pipeline.

    Returns a comprehensive decision dict containing:
        query, context, rule_*, ml_*, final_*, confidence_score
    """
    _load_model()

    # 1 ── Build RAG query — skip meaningless placeholder values
    _NOISE = {"", "none", "n/a", "no", "na", "nil", "not applicable"}

    def _meaningful(v: str) -> str:
        """Return v if it is a real user value, empty string otherwise."""
        return v.strip() if v.strip().lower() not in _NOISE else ""

    m1 = _meaningful(medicine_1)
    m2 = _meaningful(medicine_2)

    # Primary query: drug name(s) + safety focus keywords
    # When medicine_2 is present, emphasise interaction/contraindication sections
    primary_parts = [m1]
    if m2:
        primary_parts += [m2, "interaction contraindication warnings"]
    else:
        primary_parts.append("warnings contraindication")

    # Secondary context: allergy / condition / food only if meaningful
    for field in (condition, allergy, food, symptoms):
        val = _meaningful(field)
        if val:
            primary_parts.append(val)

    query = " ".join(p for p in primary_parts if p)

    # 2 ── Retrieve relevant FDA label chunks
    context = retrieve_context(query, top_k=top_k)
    seen = {(c["source_id"], c["section_name"]) for c in context}

    def _merge(extra_chunks):
        for chunk in extra_chunks:
            key = (chunk["source_id"], chunk["section_name"])
            if key not in seen:
                context.append(chunk)
                seen.add(key)

    # Focused sub-query for drug-drug interaction when medicine_2 is present
    if m2:
        _merge(retrieve_context(
            f"{m1} {m2} contraindication interaction serotonin syndrome", top_k=4
        ))

    # Focused sub-query for allergy-related contraindications
    al = _meaningful(allergy)
    if al:
        _merge(retrieve_context(
            f"{m1} allergic {al} contraindication hypersensitivity do not use", top_k=3
        ))

    # 3 ── Apply forward-chaining rule engine
    rule_result = apply_rules(
        medicine_1=medicine_1,
        medicine_2=medicine_2,
        age=int(age) if age else 0,
        condition=condition,
        allergy=allergy,
        food=food,
        retrieved_context=context,
    )

    # 4 ── ML prediction from extracted features
    features     = _extract_ml_features(context)
    feat_vec     = np.array([[features[f] for f in FEATURES]])
    ml_pred      = _clf.predict(feat_vec)[0]
    ml_proba     = _clf.predict_proba(feat_vec)[0]
    classes      = list(_clf.classes_)
    ml_conf      = float(max(ml_proba))
    ml_proba_map = {cls: round(float(p), 4) for cls, p in zip(classes, ml_proba)}

    # 5 ── Decision fusion  (rule engine has priority)
    rule_risk = rule_result["rule_risk_level"]
    final_rank = max(RISK_RANK.get(rule_risk, 0), RISK_RANK.get(ml_pred, 0))
    final_risk = RANK_NAME[final_rank]

    # Reduce confidence when rule and ML disagree
    confidence = round(ml_conf * (0.95 if rule_risk == ml_pred else 0.80), 4)

    # Composite risk score: blend rule score and ML-derived feature score,
    # then clamp to the correct band for the final risk level so the displayed
    # number (×100) always falls in: Low 0-39 / Moderate 40-74 / High 75-100
    raw_score = rule_result["rule_score"] * 0.5 + features["risk_score"] * 0.5
    if final_risk == "High":
        clamped = max(0.75, min(1.00, raw_score))
    elif final_risk == "Moderate":
        clamped = max(0.40, min(0.74, raw_score))
    else:
        clamped = min(0.39, raw_score)
    final_score = round(clamped, 4)

    return {
        "query":            query,
        "context":          context,
        "rule_risk_level":  rule_risk,
        "rule_score":       rule_result["rule_score"],
        "triggered_rules":  rule_result["triggered_rules"],
        "reasons":          rule_result["reasons"],
        "ml_risk_level":    ml_pred,
        "ml_confidence":    ml_conf,
        "ml_probabilities": ml_proba_map,
        "final_risk_level": final_risk,
        "final_risk_score": final_score,
        "confidence_score": confidence,
        "ml_features":      features,
    }

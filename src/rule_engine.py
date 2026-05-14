"""
rule_engine.py  (v2 — context-aware forward chaining)

Core fix: "do not use", "contraindicated", "anaphylaxis", etc. are ONLY
escalated to High when they appear near the user's specific inputs
(medicine_2, allergy, condition, food/drink, age, route).
Generic occurrences of those phrases fall back to Moderate.

Forward-chaining structure:
  Step 1 — Always-High phrases (no context check)
  Step 2 — Context-dependent High phrases (proximity check against user inputs)
  Step 3 — Always-Moderate phrases
  Step 4 — Fallback Moderate for Step-2 phrases that did NOT fire High
  Step 5 — Patient-context Moderate (age, food, condition mention)
  Step 6 — Score and final level

Standalone test:
    python src/rule_engine.py
"""

from typing import List, Dict, Set

# ── Helper functions ──────────────────────────────────────────────────────────

def _contains(text: str, phrase: str) -> bool:
    return phrase.lower() in text.lower()


# Words that immediately precede "fatal" and downgrade it from auto-High
_FATAL_QUALIFIERS = {"rarely", "occasionally", "seldom", "sometimes", "potentially",
                     "possibly", "uncommon", "infrequently", "very"}

def _fatal_significant(text: str) -> bool:
    """
    Returns True only when 'fatal' appears WITHOUT a downgrade qualifier such as
    'rarely fatal' or 'occasionally fatal'.  Plain 'fatal', 'can be fatal',
    'fatal outcomes', 'fatal reactions' all return True.
    """
    tl  = text.lower()
    pos = 0
    while True:
        idx = tl.find("fatal", pos)
        if idx == -1:
            return False
        preceding = tl[max(0, idx - 20):idx]
        last_word = preceding.split()[-1] if preceding.split() else ""
        if last_word not in _FATAL_QUALIFIERS:
            return True          # significant fatal mention found
        pos = idx + 1
    return False


def _user_terms(value: str) -> List[str]:
    """Return meaningful tokens from a user input field. Empty/none returns []."""
    if not value or value.strip().lower() in ("", "none", "n/a", "no", "na", "nil"):
        return []
    return [w for w in value.lower().split() if len(w) > 2]


def _near(combined: str, trigger: str, context_terms: List[str], window: int = 400) -> bool:
    """
    Returns True if `trigger` appears within `window` characters of ANY term
    in `context_terms`.  Used for context-dependent High rule checks.
    """
    if not context_terms:
        return False
    tl  = combined.lower()
    tr  = trigger.lower()
    pos = 0
    while True:
        idx = tl.find(tr, pos)
        if idx == -1:
            break
        start      = max(0, idx - window)
        end        = min(len(tl), idx + len(tr) + window)
        surround   = tl[start:end]
        if any(ct in surround for ct in context_terms):
            return True
        pos = idx + 1
    return False


# ── Rule definitions ──────────────────────────────────────────────────────────

# Step 1 — Always escalate to High, no context needed
ALWAYS_HIGH: List = [
    ("RULE_FATAL",            "fatal",
     "FDA label warns of FATAL outcomes."),
    ("RULE_DEATH",            "death",
     "FDA label warns of risk of DEATH."),
    ("RULE_LIFE_THREATENING", "life-threatening",
     "FDA label flags a LIFE-THREATENING risk."),
    ("RULE_SEROTONIN_SYN",    "serotonin syndrome",
     "SEROTONIN SYNDROME detected — a potentially life-threatening drug interaction."),
    ("RULE_RHABDO",           "rhabdomyolysis",
     "FDA label warns of RHABDOMYOLYSIS risk."),
]

# Step 3 — Always Moderate, regardless of context
ALWAYS_MODERATE: List = [
    ("RULE_CAUTION",     "caution",
     "FDA label advises CAUTION."),
    ("RULE_MONITOR",     "monitor",
     "FDA label recommends MONITORING during therapy."),
    ("RULE_AVOID",       "avoid",
     "FDA label recommends AVOIDING this combination."),
    ("RULE_ASK_DOCTOR",  "ask a doctor",
     "FDA label advises consulting a DOCTOR."),
    ("RULE_CONSULT",     "consult",
     "FDA label recommends professional CONSULTATION."),
    ("RULE_ADV_REACT",   "adverse reaction",
     "ADVERSE REACTIONS reported in FDA label."),
    ("RULE_USE_CAUTION", "use caution",
     "FDA label explicitly says USE CAUTION."),
]

# Step 4 — Fallback Moderate for phrases that did not fire a context-High rule
# key = phrase that triggers fallback, value = (rule_id, reason)
FALLBACK_MODERATE: Dict = {
    "do not use":       ("RULE_DO_NOT_USE_GENERIC",
                         "FDA label contains a DO NOT USE warning "
                         "(does not match your specific inputs — treated as Moderate)."),
    "contraindicated":  ("RULE_CONTRAINDICATED_GENERIC",
                         "FDA label contains a CONTRAINDICATION "
                         "(not matched to your specific inputs — treated as Moderate)."),
    "anaphylaxis":      ("RULE_ANAPHYLAXIS_GENERIC",
                         "FDA label mentions ANAPHYLAXIS risk "
                         "(no matching allergy reported — treated as Moderate)."),
    "hypersensitivity": ("RULE_HYPERSENSITIVITY_GENERIC",
                         "FDA label mentions HYPERSENSITIVITY "
                         "(no matching allergy reported — treated as Moderate)."),
    "not recommended":  ("RULE_NOT_RECOMMENDED_GENERIC",
                         "FDA label contains NOT RECOMMENDED warning "
                         "(not matched to your inputs — treated as Moderate)."),
    "liver failure":    ("RULE_LIVER_FAILURE_GENERIC",
                         "FDA label warns of LIVER FAILURE risk "
                         "(no matching condition/food reported — treated as Moderate)."),
    "renal failure":    ("RULE_RENAL_FAILURE_GENERIC",
                         "FDA label warns of RENAL FAILURE risk — treated as Moderate."),
    "bleeding":         ("RULE_BLEEDING_GENERIC",
                         "FDA label warns of BLEEDING risk "
                         "(no matching anticoagulant or condition reported — treated as Moderate)."),
    "seizure":          ("RULE_SEIZURE_GENERIC",
                         "FDA label warns of SEIZURE risk — treated as Moderate."),
}


def apply_rules(
    medicine_1: str,
    medicine_2: str,
    age: int,
    condition: str,
    allergy: str,
    food: str,
    retrieved_context: List[Dict],
) -> Dict:
    """
    Forward-chaining over retrieved FDA label chunks and user inputs.

    Returns:
        rule_risk_level  : "Low" | "Moderate" | "High"
        rule_score       : float 0-1  (maps to 0-39 / 40-74 / 75-100 × 100)
        triggered_rules  : list of rule IDs that fired
        reasons          : list of human-readable reasoning steps
    """
    all_texts = [chunk.get("text", "") for chunk in retrieved_context]
    combined  = " ".join(all_texts).lower()

    triggered:      List[str] = []
    reasons:        List[str] = []
    high_hits:      int       = 0
    mod_hits:       int       = 0
    handled_high:   Set[str]  = set()   # phrases already escalated to High

    # ── Build user context term lists ─────────────────────────────────────────
    m2_terms      = _user_terms(medicine_2)
    allergy_terms = _user_terms(allergy)
    cond_terms    = _user_terms(condition)
    food_terms    = _user_terms(food)

    # Expand known synonym sets
    is_pregnant   = bool(cond_terms) and any(
        t in ("pregnant", "pregnancy", "prenatal", "gestation")
        for t in cond_terms
    )

    alcohol_terms = food_terms + ["alcohol", "alcoholic", "ethanol", "drinking", "beer", "wine"]
    liver_cond    = cond_terms + ["liver", "hepatic", "cirrhosis", "hepatitis"]

    # Anticoagulant / blood-thinner terms (from medicine_2 + well-known drugs)
    anticoag_kw   = ["warfarin", "heparin", "aspirin", "anticoagulant", "coumadin",
                     "clopidogrel", "apixaban", "rivaroxaban", "dabigatran",
                     "eliquis", "xarelto", "pradaxa", "plavix", "blood thinner"]

    # Skin / wound terms
    skin_damage_kw  = ["damaged skin", "broken skin", "wound", "burn", "ulcer",
                       "open skin", "lesion", "abrasion"]
    skin_terms      = cond_terms + skin_damage_kw

    # Route / administration hazard terms
    route_hazard_kw = ["intravenous", "intrathecal", "intramuscular", "subcutaneous",
                       "fatal if", "fatal when", "route", "parenteral", "injection"]

    # Age label terms
    elderly_kw   = ["elderly", "geriatric", "older adult", "older patients", "65"]
    pediatric_kw = ["pediatric", "children", "infant", "neonates", "newborn", "child"]

    # ── Step 1: Always-High ───────────────────────────────────────────────────
    for rule_id, keyword, reason in ALWAYS_HIGH:
        # Special check for "fatal": exclude "rarely fatal" / "occasionally fatal"
        if keyword == "fatal":
            fires = _fatal_significant(combined)
        else:
            fires = _contains(combined, keyword)
        if fires:
            triggered.append(rule_id)
            reasons.append(reason)
            high_hits += 1

    # ── Step 2: Context-dependent High checks ─────────────────────────────────

    # 2a. "contraindicated" near medicine_2
    if _contains(combined, "contraindicated"):
        if m2_terms and _near(combined, "contraindicated", m2_terms):
            triggered.append("RULE_CONTRAINDICATED_M2")
            reasons.append(f"Drug is CONTRAINDICATED with '{medicine_2}' per FDA label.")
            high_hits += 1
            handled_high.add("contraindicated")

    # 2b. "do not use" — multiple context checks
    if _contains(combined, "do not use"):
        fired = False

        if m2_terms and _near(combined, "do not use", m2_terms):
            triggered.append("RULE_DO_NOT_USE_M2")
            reasons.append(f"FDA label says DO NOT USE with '{medicine_2}'.")
            high_hits += 1
            fired = True

        if allergy_terms and _near(
            combined, "do not use",
            allergy_terms + ["allergic", "allergy", "hypersensitive", "sensitivity"]
        ):
            triggered.append("RULE_DO_NOT_USE_ALLERGY")
            reasons.append(f"FDA label says DO NOT USE in patients allergic to '{allergy}'.")
            high_hits += 1
            fired = True

        if is_pregnant and _near(
            combined, "do not use",
            ["pregnancy", "pregnant", "prenatal", "fetal", "nursing", "lactation"]
        ):
            triggered.append("RULE_DO_NOT_USE_PREGNANCY")
            reasons.append("FDA label says DO NOT USE during pregnancy — matches reported condition.")
            high_hits += 1
            fired = True

        if food_terms and _near(combined, "do not use", food_terms):
            triggered.append("RULE_DO_NOT_USE_FOOD")
            reasons.append(f"FDA label says DO NOT USE with '{food}'.")
            high_hits += 1
            fired = True

        # Only fire skin rule when user's condition actually mentions skin/wound keywords
        _skin_user_kw = {"skin", "wound", "burn", "ulcer", "lesion", "abrasion",
                         "sore", "rash", "blister", "damaged", "broken", "laceration"}
        user_has_skin = any(t in _skin_user_kw for t in cond_terms)
        if user_has_skin and _near(combined, "do not use", skin_terms):
            triggered.append("RULE_DO_NOT_USE_SKIN")
            reasons.append("FDA label says DO NOT USE on damaged/broken skin — matches reported condition.")
            high_hits += 1
            fired = True

        if age and age < 12 and _near(combined, "do not use", pediatric_kw):
            triggered.append("RULE_DO_NOT_USE_PEDIATRIC")
            reasons.append(f"FDA label says DO NOT USE in children — patient age {age} < 12.")
            high_hits += 1
            fired = True

        if age and age >= 65 and _near(combined, "do not use", elderly_kw):
            triggered.append("RULE_DO_NOT_USE_GERIATRIC")
            reasons.append(f"FDA label says DO NOT USE in elderly — patient age {age} >= 65.")
            high_hits += 1
            fired = True

        if _near(combined, "do not use", route_hazard_kw):
            triggered.append("RULE_DO_NOT_USE_ROUTE")
            reasons.append("FDA label warns DO NOT USE via certain routes (e.g. fatal if given IV/intrathecally).")
            high_hits += 1
            fired = True

        if fired:
            handled_high.add("do not use")

    # 2c. "anaphylaxis" near matching allergy
    if _contains(combined, "anaphylaxis") and allergy_terms:
        if _near(combined, "anaphylaxis", allergy_terms):
            triggered.append("RULE_ANAPHYLAXIS_ALLERGY")
            reasons.append(f"ANAPHYLAXIS risk matches reported allergy '{allergy}'.")
            high_hits += 1
            handled_high.add("anaphylaxis")

    # 2d. "hypersensitivity" near matching allergy
    if _contains(combined, "hypersensitivity") and allergy_terms:
        if _near(combined, "hypersensitivity", allergy_terms):
            triggered.append("RULE_HYPERSENSITIVITY_ALLERGY")
            reasons.append(f"HYPERSENSITIVITY warning matches reported allergy '{allergy}'.")
            high_hits += 1
            handled_high.add("hypersensitivity")

    # 2e. "not recommended" near medicine_2
    if _contains(combined, "not recommended") and m2_terms:
        if _near(combined, "not recommended", m2_terms):
            triggered.append("RULE_NOT_RECOMMENDED_M2")
            reasons.append(f"Combination with '{medicine_2}' is NOT RECOMMENDED per FDA label.")
            high_hits += 1
            handled_high.add("not recommended")

    # 2f. Liver failure / hepatotoxicity near alcohol or liver condition
    liver_phrases = ["liver failure", "liver damage", "hepatotoxicity", "hepatic failure",
                     "hepatic injury", "liver injury"]
    if any(_contains(combined, p) for p in liver_phrases):
        if _near(combined, "liver", alcohol_terms) or _near(combined, "liver", liver_cond):
            triggered.append("RULE_LIVER_CONTEXT")
            reasons.append(
                f"LIVER DAMAGE warning matches "
                f"{'food/drink' if food_terms else 'reported condition'} "
                f"'{food or condition}'."
            )
            high_hits += 1
            handled_high.add("liver failure")

    # 2g. Bleeding near anticoagulant (medicine_2 or well-known blood thinners)
    if _contains(combined, "bleeding") and m2_terms:
        m2_lower = medicine_2.lower()
        is_anticoag = any(ak in m2_lower for ak in anticoag_kw)
        if is_anticoag and _near(combined, "bleeding", m2_terms + anticoag_kw):
            triggered.append("RULE_BLEEDING_ANTICOAG")
            reasons.append(f"BLEEDING RISK warning is relevant to '{medicine_2}' (anticoagulant/blood thinner).")
            high_hits += 1
            handled_high.add("bleeding")

    # 2h. Pregnancy-specific contraindication when condition includes pregnancy
    preg_contra_phrases = [
        "contraindicated in pregnancy", "avoid during pregnancy",
        "not recommended during pregnancy", "teratogenic",
        "should not be used during pregnancy",
    ]
    if is_pregnant and any(_contains(combined, p) for p in preg_contra_phrases):
        triggered.append("RULE_PREGNANCY_CONTRA")
        reasons.append("PREGNANCY CONTRAINDICATION found — condition includes pregnancy.")
        high_hits += 1

    # 2i. Medicine_1 name confirmation (informational only — no risk escalation)
    m1_terms = _user_terms(medicine_1)
    if m1_terms and any(t in combined for t in m1_terms):
        triggered.append("RULE_M1_IN_LABEL")
        reasons.append(f"Medicine '{medicine_1}' confirmed in retrieved FDA label sections.")

    # ── Step 3: Always-Moderate ───────────────────────────────────────────────
    for rule_id, keyword, reason in ALWAYS_MODERATE:
        if _contains(combined, keyword):
            triggered.append(rule_id)
            reasons.append(reason)
            mod_hits += 1

    # ── Step 4: Fallback Moderate for unhandled context phrases ───────────────
    for phrase, (rule_id, reason) in FALLBACK_MODERATE.items():
        if phrase not in handled_high and _contains(combined, phrase):
            triggered.append(rule_id)
            reasons.append(reason)
            mod_hits += 1

    # ── Step 5: Patient-context Moderate signals ──────────────────────────────
    if age and age >= 65 and _contains(combined, "geriatric"):
        triggered.append("RULE_GERIATRIC_AGE")
        reasons.append(f"Patient age {age} >= 65: FDA label includes GERIATRIC warnings.")
        mod_hits += 1

    if age and age < 12 and _contains(combined, "pediatric"):
        triggered.append("RULE_PEDIATRIC_AGE")
        reasons.append(f"Patient age {age} < 12: FDA label includes PEDIATRIC warnings.")
        mod_hits += 1

    if food_terms and any(t in combined for t in food_terms):
        triggered.append("RULE_FOOD_MATCH")
        reasons.append(f"Food/drink '{food}' is referenced in the retrieved FDA label.")
        mod_hits += 1

    if cond_terms and any(t in combined for t in cond_terms):
        triggered.append("RULE_CONDITION_MATCH")
        reasons.append(f"Medical condition '{condition}' is referenced in the retrieved FDA label.")
        mod_hits += 1

    # Medicine_2 mentioned but not yet captured as context-High
    if m2_terms and any(t in combined for t in m2_terms):
        if "RULE_CONTRAINDICATED_M2" not in triggered and \
           "RULE_DO_NOT_USE_M2" not in triggered and \
           "RULE_NOT_RECOMMENDED_M2" not in triggered:
            triggered.append("RULE_M2_MENTION")
            reasons.append(
                f"Medicine '{medicine_2}' appears in retrieved FDA label — "
                "possible interaction; review with a healthcare professional."
            )
            mod_hits += 1

    # ── Step 6: Score and final decision ──────────────────────────────────────
    #
    # Scoring weights:
    #   high hit  → 20 points each
    #   mod hit   → 5  points each
    #
    # Thresholds (0-100 scale, reported as /100 in UI):
    #   High     >= 75   (at least 1 high hit guarantees this)
    #   Moderate >= 40
    #   Low      <  40
    #
    # rule_score returned as 0-1 (multiply by 100 to display)

    raw = high_hits * 20 + mod_hits * 5

    if high_hits >= 1:
        level = "High"
        score = max(75, min(100, raw))
    elif mod_hits >= 1:
        level = "Moderate"
        score = max(40, min(74, raw))
    else:
        level = "Low"
        score = min(35, raw)

    rule_score = round(score / 100, 4)

    if not triggered:
        reasons = ["No safety concerns identified in retrieved FDA label sections."]

    return {
        "rule_risk_level": level,
        "rule_score":      rule_score,
        "triggered_rules": triggered,
        "reasons":         reasons,
    }


if __name__ == "__main__":
    # ── Test 1: buspirone + MAOI  →  expected High ───────────────────────────
    chunk_maoi = {
        "drug_name":    "Buspirone Hydrochloride",
        "section_name": "contraindications",
        "text": (
            "Drug: Buspirone Hydrochloride\nSection: contraindications\n"
            "Text: Buspirone is contraindicated in patients currently taking "
            "monoamine oxidase inhibitors (MAOIs). Concomitant use may result in "
            "serotonin syndrome, a potentially life-threatening condition. "
            "Do not use within 14 days of discontinuing an MAOI."
        ),
    }
    r1 = apply_rules("buspirone", "MAOI", 45, "depression", "none", "grapefruit juice",
                     [chunk_maoi])
    print("Test 1 — buspirone + MAOI")
    print(f"  Level  : {r1['rule_risk_level']}  (expected High)")
    print(f"  Score  : {round(r1['rule_score']*100,1)} / 100")
    print(f"  Rules  : {r1['triggered_rules']}\n")

    # ── Test 2: generic OTC drug, no user-specific context  →  expected Moderate
    chunk_generic = {
        "drug_name":    "Ibuprofen",
        "section_name": "warnings",
        "text": (
            "Drug: Ibuprofen\nSection: warnings\n"
            "Text: Do not use if you are allergic to ibuprofen or any other pain "
            "reliever/fever reducer (NSAID). Ask a doctor before use if you have "
            "stomach problems. Caution: monitor for signs of stomach bleeding. "
            "Do not use on damaged skin. Do not use more than directed."
        ),
    }
    r2 = apply_rules("ibuprofen", "", 30, "headache", "none", "none", [chunk_generic])
    print("Test 2 — ibuprofen, no allergy, no second medicine")
    print(f"  Level  : {r2['rule_risk_level']}  (expected Moderate, NOT High)")
    print(f"  Score  : {round(r2['rule_score']*100,1)} / 100")
    print(f"  Rules  : {r2['triggered_rules']}\n")

    # ── Test 3: ibuprofen, user allergic to NSAIDs  →  expected High ─────────
    r3 = apply_rules("ibuprofen", "", 30, "headache", "ibuprofen NSAID", "none",
                     [chunk_generic])
    print("Test 3 — ibuprofen, user allergic to NSAID/ibuprofen")
    print(f"  Level  : {r3['rule_risk_level']}  (expected High)")
    print(f"  Score  : {round(r3['rule_score']*100,1)} / 100")
    print(f"  Rules  : {r3['triggered_rules']}\n")

    # ── Test 4: warfarin + aspirin  →  expected High (bleeding risk) ─────────
    chunk_bleed = {
        "drug_name":    "Warfarin",
        "section_name": "drug_interactions",
        "text": (
            "Drug: Warfarin\nSection: drug_interactions\n"
            "Text: Concomitant use of warfarin and aspirin increases the risk of "
            "serious bleeding. Do not use together without close monitoring by a physician."
        ),
    }
    r4 = apply_rules("warfarin", "aspirin", 60, "atrial fibrillation", "none", "none",
                     [chunk_bleed])
    print("Test 4 — warfarin + aspirin (bleeding)")
    print(f"  Level  : {r4['rule_risk_level']}  (expected High)")
    print(f"  Score  : {round(r4['rule_score']*100,1)} / 100")
    print(f"  Rules  : {r4['triggered_rules']}\n")

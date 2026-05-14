"""
preprocess_json.py
Reads openFDA drug-label JSON files and produces:
  - data/processed/drug_knowledge.csv       (section-level rows for RAG)
  - data/processed/mediguard_ml_dataset.csv (drug-level rows for ML)

Usage:
    python src/preprocess_json.py
"""

import json
import csv
import sys
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

# ── Paths ─────────────────────────────────────────────────────────────────────
SRC_DIR  = Path(__file__).parent
PROJECT  = SRC_DIR.parent

# Look for JSONs in data/raw_json first, then fall back to parent directory
_raw_candidate = PROJECT / "data" / "raw_json"
RAW_DIR = _raw_candidate if list(_raw_candidate.glob("drug-label-*.json")) else PROJECT.parent

PROC_DIR = PROJECT / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

# ── Keyword lists ─────────────────────────────────────────────────────────────
HIGH_KEYWORDS = [
    "fatal", "death", "life-threatening", "contraindicated", "do not use",
    "serotonin syndrome", "anaphylaxis", "bleeding", "liver failure",
    "renal failure", "overdose", "toxicity", "seizure", "rhabdomyolysis",
    "hypersensitivity", "not recommended",
]
MODERATE_KEYWORDS = [
    "warning", "caution", "avoid", "monitor", "adverse reaction",
    "consult", "ask a doctor", "pregnancy", "pediatric", "geriatric",
    "dose adjustment",
]

MAX_POSSIBLE = len(HIGH_KEYWORDS) * 2 + len(MODERATE_KEYWORDS)  # 43

# Sections to extract for RAG knowledge base
KNOWLEDGE_SECTIONS = [
    "indications_and_usage",
    "dosage_and_administration",
    "contraindications",
    "warnings",
    "warnings_and_cautions",
    "boxed_warning",
    "precautions",
    "drug_interactions",
    "adverse_reactions",
    "pediatric_use",
    "geriatric_use",
    "pregnancy",
    "overdosage",
]

# Binary flag definitions for ML features
ML_SECTION_FLAGS = {
    "has_contraindications": ["contraindications"],
    "has_warnings":          ["warnings", "warnings_and_cautions", "boxed_warning"],
    "has_precautions":       ["precautions"],
    "has_drug_interactions": ["drug_interactions"],
    "has_adverse_reactions": ["adverse_reactions"],
    "has_pediatric_warning": ["pediatric_use"],
    "has_geriatric_warning": ["geriatric_use"],
    "has_pregnancy_warning": ["pregnancy"],
    "has_overdosage_info":   ["overdosage"],
}


def get_text(record, *fields):
    parts = []
    for f in fields:
        val = record.get(f, "")
        if isinstance(val, list):
            parts.append(" ".join(str(x) for x in val))
        elif val:
            parts.append(str(val))
    return " ".join(parts)


def safe_first(lst, default=""):
    if isinstance(lst, list) and lst:
        return str(lst[0])
    return default


def compute_risk(record):
    combined = get_text(
        record,
        "warnings", "warnings_and_cautions", "boxed_warning",
        "contraindications", "precautions", "drug_interactions",
        "adverse_reactions", "overdosage", "pediatric_use",
        "geriatric_use", "pregnancy",
    ).lower()

    high_count = sum(1 for w in HIGH_KEYWORDS if w in combined)
    mod_count  = sum(1 for w in MODERATE_KEYWORDS if w in combined)
    raw        = high_count * 2 + mod_count
    score      = raw / MAX_POSSIBLE  # normalised 0-1

    # Thresholds calibrated from dataset (buspirone ~0.63 = High)
    if score >= 0.55:
        label = "High"
    elif score >= 0.30:
        label = "Moderate"
    else:
        label = "Low"

    return round(score, 4), high_count, label


def process_record(record, source_id):
    openfda       = record.get("openfda", {})
    brand_names   = openfda.get("brand_name",   [])
    generic_names = openfda.get("generic_name", [])
    routes        = openfda.get("route",         [])

    drug_name    = safe_first(brand_names) or safe_first(generic_names) or "Unknown"
    brand_name   = safe_first(brand_names)
    generic_name = safe_first(generic_names)
    route        = safe_first(routes)

    # ── Knowledge rows (one per non-empty section) ────────────────────────────
    knowledge_rows = []
    for sec in KNOWLEDGE_SECTIONS:
        text = get_text(record, sec).strip()
        if len(text) > 20:
            knowledge_rows.append({
                "source_id":    source_id,
                "drug_name":    drug_name,
                "brand_name":   brand_name,
                "generic_name": generic_name,
                "route":        route,
                "section_name": sec,
                "section_text": text[:800],  # cap at 800 chars
            })

    # ── ML row (one per drug) ─────────────────────────────────────────────────
    risk_score, serious_kw_count, risk_level = compute_risk(record)

    ml_row = {
        "source_id":    source_id,
        "drug_name":    drug_name,
        "brand_name":   brand_name,
        "generic_name": generic_name,
        "route":        route,
    }
    for flag, fields in ML_SECTION_FLAGS.items():
        ml_row[flag] = 1 if len(get_text(record, *fields).strip()) > 10 else 0

    ml_row["serious_keyword_count"] = serious_kw_count
    ml_row["risk_score"]            = risk_score
    ml_row["risk_level"]            = risk_level

    return knowledge_rows, ml_row


def main():
    json_files = sorted(RAW_DIR.glob("drug-label-*.json"))
    if not json_files:
        print(f"No drug-label JSON files found in: {RAW_DIR}")
        sys.exit(1)

    print(f"Found {len(json_files)} JSON file(s) in {RAW_DIR}")

    kn_path = PROC_DIR / "drug_knowledge.csv"
    ml_path = PROC_DIR / "mediguard_ml_dataset.csv"

    kn_fields = ["source_id", "drug_name", "brand_name", "generic_name",
                 "route", "section_name", "section_text"]
    ml_fields = ["source_id", "drug_name", "brand_name", "generic_name", "route",
                 "has_contraindications", "has_warnings", "has_precautions",
                 "has_drug_interactions", "has_adverse_reactions",
                 "has_pediatric_warning", "has_geriatric_warning",
                 "has_pregnancy_warning", "has_overdosage_info",
                 "serious_keyword_count", "risk_score", "risk_level"]

    total_drugs = 0
    total_kn    = 0

    with open(kn_path, "w", newline="", encoding="utf-8") as kf, \
         open(ml_path, "w", newline="", encoding="utf-8") as mf:

        kw = csv.DictWriter(kf, fieldnames=kn_fields)
        mw = csv.DictWriter(mf, fieldnames=ml_fields)
        kw.writeheader()
        mw.writeheader()

        for jfile in json_files:
            print(f"\nProcessing {jfile.name} ...")
            with open(jfile, "r", encoding="utf-8") as f:
                data = json.load(f)
            results   = data.get("results", [])
            iterable  = tqdm(results, desc=jfile.name, unit="rec") if HAS_TQDM else results

            for i, record in enumerate(iterable):
                source_id = f"{jfile.stem}_{i}"
                kn_rows, ml_row = process_record(record, source_id)
                for row in kn_rows:
                    kw.writerow(row)
                mw.writerow(ml_row)
                total_drugs += 1
                total_kn    += len(kn_rows)

    print(f"\n[OK] Preprocessing complete")
    print(f"  Drugs processed    : {total_drugs:,}")
    print(f"  Knowledge rows     : {total_kn:,}")
    print(f"  drug_knowledge.csv : {kn_path}")
    print(f"  ml_dataset.csv     : {ml_path}")


if __name__ == "__main__":
    main()

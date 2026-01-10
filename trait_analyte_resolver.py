# trait_analyte_resolver.py
from __future__ import annotations
from typing import Dict, List, Optional


def resolve_analytes_for_trait(
    trait: str,
    trait_map: Dict[str, List[str]],
    allowed_analytes: List[str],
) -> Optional[List[str]]:
    """
    Returns a list of analytes ONLY if the trait matches:
      1) exact YAML key match (case-insensitive)
      2) partial substring match against YAML keys
      3) keyword fallback (cholesterol/glucose/etc.)

    If nothing matches, returns None (IMPORTANT: we do NOT return all analytes,
    because we want to IGNORE unrelated traits).
    """
    if not trait:
        return None

    t = trait.lower().strip()

    # 1) Exact match in YAML
    if t in trait_map:
        analytes = trait_map[t]
        analytes = [a for a in analytes if a in allowed_analytes]
        return analytes if analytes else None

    # 2) Partial substring match (either direction)
    for yaml_trait, analyte_list in trait_map.items():
        if yaml_trait in t or t in yaml_trait:
            analytes = [a for a in analyte_list if a in allowed_analytes]
            return analytes if analytes else None

    # 3) Keyword fallback (only for your supported analytes)
    if "cholesterol" in t:
        analytes = [a for a in ["LDL-C", "HDL-C", "Triglycerides"] if a in allowed_analytes]
        return analytes if analytes else None

    if "triglyceride" in t:
        analytes = ["Triglycerides"] if "Triglycerides" in allowed_analytes else []
        return analytes if analytes else None

    if "glucose" in t:
        analytes = ["Fasting Glucose"] if "Fasting Glucose" in allowed_analytes else []
        return analytes if analytes else None

    if "a1c" in t or "hba1c" in t:
        analytes = ["HbA1c"] if "HbA1c" in allowed_analytes else []
        return analytes if analytes else None

    if "vitamin d" in t or "25(oh)d" in t:
        analytes = ["25(OH)D"] if "25(OH)D" in allowed_analytes else []
        return analytes if analytes else None

    if "ferritin" in t:
        analytes = ["Ferritin"] if "Ferritin" in allowed_analytes else []
        return analytes if analytes else None

    if "creatinine" in t or "creatinine clearance" in t:
        analytes = ["Creatinine"] if "Creatinine" in allowed_analytes else []
        return analytes if analytes else None

    if "hemoglobin" in t:
        analytes = ["Hemoglobin"] if "Hemoglobin" in allowed_analytes else []
        return analytes if analytes else None

    if "tsh" in t or "thyroid" in t or "tshb" in t:
        analytes = ["TSH"] if "TSH" in allowed_analytes else []
        return analytes if analytes else None

    # Nothing matched → IGNORE THIS TRAIT
    return None


def filter_gwas_to_supported_traits(
    gwas_hits_df,
    trait_map: Dict[str, List[str]],
    allowed_analytes: List[str],
):
    """
    Keep only GWAS rows whose traits can be resolved into one or more allowed analytes.
    Adds a new column: '__resolved_analytes' (list) to use later during fusion.
    """
    resolved_lists = []
    keep_mask = []

    for _, row in gwas_hits_df.iterrows():
        trait = row.get("DISEASE/TRAIT", "")
        analytes = resolve_analytes_for_trait(trait, trait_map, allowed_analytes)
        if analytes:
            keep_mask.append(True)
            resolved_lists.append(analytes)
        else:
            keep_mask.append(False)
            resolved_lists.append(None)

    filtered = gwas_hits_df.loc[keep_mask].copy()
    # Only assign for kept rows
    filtered["__resolved_analytes"] = [a for a in resolved_lists if a is not None]
    return filtered

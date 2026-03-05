# streamlit5_app.py
import os
import streamlit as st
import pandas as pd
import yaml

# ✅ Gemini (cloud narration)
from google import genai

# ✅ Your trait→analyte filter module
from trait_analyte_resolver import filter_gwas_to_supported_traits

# ✅ For robust Google Drive download (handles big files)
import requests
from io import BytesIO

# -----------------------------
# CONFIG
# -----------------------------
# ✅ Use Google Drive file id for big GWAS file
GWAS_DRIVE_FILE_ID = "1F92lf8My0699QVdCfPx_HbYmgc_WHjLx"

THRESHOLDS_FILE = "trusted_lab_thresholds.csv"  # ensure this file exists in repo
TRAIT_MAP_FILE = "trait_to_labs_partial.yaml"

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")


# -----------------------------
# GOOGLE DRIVE DOWNLOADER (BIG FILE SAFE)
# -----------------------------
def _download_gdrive_file(file_id: str) -> bytes:
    """
    Robust Google Drive download that handles large-file confirmation tokens.
    Returns file bytes.
    """
    if not file_id:
        raise RuntimeError("GWAS_DRIVE_FILE_ID is not set. Paste your Google Drive FILE_ID in the code.")

    session = requests.Session()

    # First request (may return confirmation page for large files)
    url = "https://drive.google.com/uc?export=download"
    response = session.get(url, params={"id": file_id}, stream=True)

    # Try to find confirm token (Google uses it for large files)
    token = None
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break

    # If token exists, request again with confirm token
    if token:
        response = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

    response.raise_for_status()

    data = BytesIO()
    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1MB chunks
        if chunk:
            data.write(chunk)
    return data.getvalue()


@st.cache_data(show_spinner="Loading GWAS reference data… (first time may take ~10–30s)")
def load_gwas_from_drive(file_id: str) -> pd.DataFrame:
    raw = _download_gdrive_file(file_id)
    return pd.read_csv(BytesIO(raw), low_memory=False)


@st.cache_data(show_spinner="Loading lab thresholds…")
def load_thresholds(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading trait→lab mapping…")
def load_trait_map(path: str) -> dict:
    with open(path, "r") as f:
        raw_map = yaml.safe_load(f) or {}
    return {str(k).lower().strip(): v for k, v in raw_map.items()}


# -----------------------------
# GEMINI CLIENT
# -----------------------------
@st.cache_resource
def load_gemini_client():
    api_key = None
    try:
        api_key = st.secrets.get("GEMINI_API_KEY", None)
    except Exception:
        api_key = None

    if not api_key:
        api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Set it as an environment variable or in Streamlit Secrets."
        )

    return genai.Client(api_key=api_key)


gemini_client = load_gemini_client()


def gemini_narrate(prompt: str, model_name: str = DEFAULT_GEMINI_MODEL) -> str:
    try:
        resp = gemini_client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return (resp.text or "").strip()
    except Exception as e:
        return f"(Narration unavailable due to Gemini API error: {e})"


# -----------------------------
# LAB FLAG PARSER
# -----------------------------
def parse_flag_logic(logic, value):
    try:
        parts = logic.split(";")
        for part in parts:
            if ">=" in part:
                threshold, flag = part.split(":")
                if value >= float(threshold.replace(">=", "")):
                    return flag
            elif "<=" in part:
                threshold, flag = part.split(":")
                if value <= float(threshold.replace("<=", "")):
                    return flag
            elif ">" in part:
                threshold, flag = part.split(":")
                if value > float(threshold.replace(">", "")):
                    return flag
            elif "<" in part:
                threshold, flag = part.split(":")
                if value < float(threshold.replace("<", "")):
                    return flag
            elif "else" in part:
                _, flag = part.split(":")
                return flag
    except Exception:
        return "Unknown"
    return "Unknown"


# -----------------------------
# INPUT NORMALIZATION + GENETIC LOGIC
# -----------------------------
def normalize_rsid(x: str) -> str:
    return str(x).strip().lower()


def normalize_genotype(x: str) -> str:
    return str(x).strip().upper().replace(" ", "")


def valid_genotype(gt: str) -> bool:
    return len(gt) == 2 and all(c in "ACGT" for c in gt)


# Complement map for strand flip correction
_COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C"}


def complement_allele(allele: str) -> str:
    """Return the complementary strand allele (e.g. A→T, C→G)."""
    return _COMPLEMENT.get(allele.upper(), allele)


def extract_risk_allele_from_row(row) -> str | None:
    """
    Extract risk allele from a GWAS row.
    Tries direct columns first, then parses 'STRONGEST SNP-RISK ALLELE'.
    Returns the allele as reported (strand flip is handled separately in compute_genetic_risk).
    """
    for col in [
        "EFFECT_ALLELE", "effect_allele", "EFFECT ALLELE",
        "RISK_ALLELE", "risk_allele", "RISK ALLELE",
    ]:
        if col in row and pd.notna(row[col]):
            cand = str(row[col]).strip().upper()
            if cand in {"A", "C", "G", "T"}:
                return cand

    for col in [
        "STRONGEST SNP-RISK ALLELE",
        "STRONGEST_SNP_RISK_ALLELE",
        "strongest_snp_risk_allele",
    ]:
        if col in row and pd.notna(row[col]):
            s = str(row[col]).strip().upper()
            if "-" in s:
                cand = s.split("-")[-1]
                if cand in {"A", "C", "G", "T"}:
                    return cand

    return None


def extract_effect_size(row) -> float | None:
    """
    Extract a numeric effect size (OR or beta) from the GWAS row.
    OR > 1 is risk-increasing; beta can be positive or negative.
    Returns None if not available or not parseable.
    """
    # Try Odds Ratio first
    for col in ["OR or BETA", "OR_OR_BETA", "OR", "ODDS_RATIO", "odds_ratio"]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                if val > 0:
                    return val
            except (ValueError, TypeError):
                pass

    # Try beta coefficient
    for col in ["BETA", "beta", "EFFECT_SIZE", "effect_size"]:
        if col in row and pd.notna(row[col]):
            try:
                val = float(row[col])
                return val
            except (ValueError, TypeError):
                pass

    return None


def compute_genetic_risk(user_gt: str, risk_allele: str | None, effect_size: float | None = None) -> tuple[str, float]:
    """
    Returns (risk_label, weighted_score).

    Weighted score accounts for:
    - Allele dosage (0/1/2 copies of risk allele)
    - Effect size (OR or beta magnitude), if available
    - Strand flip correction (tries complement if direct match fails)

    Score interpretation:
      0.0  = no risk allele detected
      0.5  = heterozygous, no effect size
      1.0  = homozygous, no effect size  (or strong effect heterozygous)
      >1.0 = strong OR / large beta
    """
    if not risk_allele:
        return "Unknown (risk allele missing in GWAS)", 0.0

    # Direct match
    cnt = user_gt.count(risk_allele)

    # Strand flip fallback — if 0 copies found, try the complement allele
    strand_flipped = False
    if cnt == 0:
        flipped = complement_allele(risk_allele)
        cnt_flipped = user_gt.count(flipped)
        if cnt_flipped > 0:
            cnt = cnt_flipped
            strand_flipped = True

    # Base dosage score: 0, 0.5, or 1.0
    dosage_score = cnt * 0.5  # 0→0.0, 1→0.5, 2→1.0

    # Effect size multiplier
    es_label = ""
    es_multiplier = 1.0
    if effect_size is not None:
        # Treat values that look like ORs (typically >0.5 and positive)
        # vs betas (can be negative or very small)
        if effect_size > 0.5:
            # Likely an OR — magnitude above 1 = risk-increasing
            deviation = abs(effect_size - 1.0)  # 0 = neutral, higher = stronger
            es_multiplier = 1.0 + deviation      # e.g. OR=1.5 → multiplier=1.5
            es_label = f" | OR≈{effect_size:.2f}"
        else:
            # Likely a beta — scale by absolute magnitude (normalised roughly)
            es_multiplier = 1.0 + abs(effect_size) * 2
            es_label = f" | β≈{effect_size:.3f}"

    weighted_score = dosage_score * es_multiplier
    flip_note = " (strand-flipped)" if strand_flipped else ""

    if cnt == 0:
        label = f"Low/No risk — effect allele not detected{flip_note}"
        return label, 0.0
    elif cnt == 2:
        label = f"High risk — homozygous{flip_note}{es_label}"
    else:
        label = f"Moderate risk — heterozygous{flip_note}{es_label}"

    return label, round(weighted_score, 3)


# -----------------------------
# ANALYTE-LEVEL AGGREGATION
# -----------------------------
def _risk_bucket(genetic_risk: str) -> str:
    gr = (genetic_risk or "").lower()
    if "high risk" in gr:
        return "high"
    if "moderate risk" in gr:
        return "moderate"
    if "low/no risk" in gr:
        return "low"
    if "uncertain" in gr or "unknown" in gr:
        return "uncertain"
    return "unknown"


def compute_genetic_influence(
    n_high: int, n_moderate: int, n_low: int, n_uncertain: int, n_unknown: int,
    total_weighted_score: float = 0.0
) -> str:
    """
    Compute genetic influence level using both allele counts AND
    the sum of weighted scores (which incorporate OR/beta effect sizes).

    Thresholds are calibrated so that:
    - A single high-OR homozygous hit drives "Strong"
    - Several moderate heterozygous hits accumulate to "Moderate"
    - Only low/uncertain hits → "Limited evidence"
    """
    total = n_high + n_moderate + n_low + n_uncertain + n_unknown
    if total == 0:
        return "Weak"

    # Use weighted score as primary signal when available
    if total_weighted_score >= 2.0 or n_high >= 1:
        return "Strong"
    if total_weighted_score >= 0.8 or n_moderate >= 2:
        return "Moderate"
    if total_weighted_score >= 0.3 or n_moderate >= 1:
        return "Weak-Moderate"
    if (n_unknown + n_uncertain + n_low) >= max(1, total // 2):
        return "Limited evidence"
    return "Weak"


def build_analyte_summary(fused_df: pd.DataFrame) -> pd.DataFrame:
    if fused_df.empty:
        return fused_df

    dedup = fused_df.drop_duplicates(subset=["rsid", "analyte"]).copy()

    rows = []
    for analyte, grp in dedup.groupby("analyte", dropna=False):
        lab_value = grp["lab_value"].iloc[0]
        lab_flag = grp["lab_flag"].iloc[0]

        rsids = sorted(set(grp["rsid"].astype(str).tolist()))
        traits = sorted(set(grp["trait"].astype(str).tolist()))
        genes = sorted(set([g for g in grp.get("gene", "").astype(str).tolist() if g and g != "nan"]))

        buckets = grp["genetic_risk"].apply(_risk_bucket).value_counts().to_dict()
        n_high = int(buckets.get("high", 0))
        n_mod = int(buckets.get("moderate", 0))
        n_low = int(buckets.get("low", 0))
        n_unc = int(buckets.get("uncertain", 0))
        n_unk = int(buckets.get("unknown", 0))

        # Sum weighted scores across all hits for this analyte
        total_weighted = float(grp["weighted_score"].sum()) if "weighted_score" in grp.columns else 0.0

        influence = compute_genetic_influence(n_high, n_mod, n_low, n_unc, n_unk, total_weighted)

        rows.append({
            "analyte": analyte,
            "lab_value": lab_value,
            "lab_flag": lab_flag,
            "genetic_influence": influence,
            "weighted_genetic_score": round(total_weighted, 3),
            "n_high": n_high,
            "n_moderate": n_mod,
            "n_low": n_low,
            "n_uncertain": n_unc,
            "n_unknown": n_unk,
            "rsids": ", ".join(rsids),
            "traits": "; ".join(traits[:6]) + (" ..." if len(traits) > 6 else ""),
            "genes": ", ".join(genes[:8]) + (" ..." if len(genes) > 8 else ""),
        })

    return pd.DataFrame(rows).sort_values(
        by=["lab_flag", "genetic_influence", "analyte"],
        ascending=[True, True, True]
    )


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="🧬 GeneNarrator+", layout="wide")
st.title("🧬 GeneNarrator+")

# --- GENETIC INPUT ---
st.subheader("🔬 Genetic Input (RSID + Genotype)")

if "snp_inputs" not in st.session_state:
    st.session_state["snp_inputs"] = [{"rsid": "", "genotype": ""}]

btn1, btn2, btn3 = st.columns([1, 1, 2])
with btn1:
    if st.button("➕ Add RSID"):
        st.session_state["snp_inputs"].append({"rsid": "", "genotype": ""})
with btn2:
    if st.button("🗑 Remove Last") and len(st.session_state["snp_inputs"]) > 1:
        st.session_state["snp_inputs"].pop()

for i, item in enumerate(st.session_state["snp_inputs"]):
    c1, c2 = st.columns([2, 1])
    item["rsid"] = c1.text_input(
        f"RSID {i+1}",
        value=item["rsid"],
        key=f"rsid_{i}",
        placeholder="e.g., rs753381 (any case accepted)"
    )
    item["genotype"] = c2.text_input(
        f"Genotype {i+1}",
        value=item["genotype"],
        key=f"geno_{i}",
        placeholder="e.g., CT / AA / gg (any case accepted)"
    )

confirm = st.button("✅ Confirm & Analyze")

# --- BLOOD TEST: NUMBER BOX + SLIDER (SYNCED) ---
st.subheader("🧪 Adjust Blood Test Values Manually")

analyte_defaults = {
    "Hemoglobin": (8.0, 18.0, 14.0),
    "LDL-C": (50.0, 200.0, 110.0),
    "HDL-C": (20.0, 100.0, 50.0),
    "Triglycerides": (50.0, 400.0, 140.0),
    "Fasting Glucose": (60.0, 200.0, 95.0),
    "HbA1c": (4.0, 12.0, 5.6),
    "Ferritin": (5.0, 400.0, 100.0),
    "TSH": (0.1, 10.0, 2.0),
    "25(OH)D": (5.0, 100.0, 35.0),
    "Creatinine": (0.4, 2.0, 1.0),
}


def _slugify_analyte_name(name: str) -> str:
    return (
        name.lower()
        .strip()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("(", "")
        .replace(")", "")
        .replace("-", "_")
        .replace(".", "")
    )


def _sync_from_number(analyte_key: str, mn: float, mx: float):
    """
    When user types in number box:
    - clamp value within min/max
    - update slider value to match
    """
    num_key = f"num_{analyte_key}"
    sld_key = f"sld_{analyte_key}"

    v = float(st.session_state[num_key])
    v = max(float(mn), min(float(mx), v))  # clamp
    st.session_state[num_key] = v
    st.session_state[sld_key] = v


def _sync_from_slider(analyte_key: str):
    """
    When user moves slider:
    - update number box to match
    """
    num_key = f"num_{analyte_key}"
    sld_key = f"sld_{analyte_key}"
    st.session_state[num_key] = float(st.session_state[sld_key])


user_values = {}
cols = st.columns(3)

for idx, (analyte, (mn, mx, default)) in enumerate(analyte_defaults.items()):
    analyte_key = _slugify_analyte_name(analyte)
    num_key = f"num_{analyte_key}"
    sld_key = f"sld_{analyte_key}"

    # Initialize once (so box + slider start aligned)
    if num_key not in st.session_state:
        st.session_state[num_key] = float(default)
    if sld_key not in st.session_state:
        st.session_state[sld_key] = float(default)

    with cols[idx % 3]:
        st.markdown(f"**{analyte}**")

        # Number box (decimal input allowed)
        st.number_input(
            "Value",
            min_value=float(mn),
            max_value=float(mx),
            value=float(st.session_state[num_key]),
            step=0.1,
            format="%.3f",
            key=num_key,
            on_change=_sync_from_number,
            args=(analyte_key, float(mn), float(mx)),
            label_visibility="collapsed"
        )

        # Slider (also decimal, synced)
        st.slider(
            "Slider",
            min_value=float(mn),
            max_value=float(mx),
            value=float(st.session_state[sld_key]),
            step=0.1,
            key=sld_key,
            on_change=_sync_from_slider,
            args=(analyte_key,),
            label_visibility="collapsed"
        )

        # Final value used downstream (always comes from the number box state)
        user_values[analyte] = float(st.session_state[num_key])

# -----------------------------
# ANALYZE
# -----------------------------
if confirm:
    cleaned_inputs = []
    errors = []

    for i, item in enumerate(st.session_state["snp_inputs"], start=1):
        rsid_raw = item.get("rsid", "")
        gt_raw = item.get("genotype", "")

        rsid = normalize_rsid(rsid_raw)
        gt = normalize_genotype(gt_raw)

        if not rsid:
            continue

        if not rsid.startswith("rs"):
            errors.append(f"Row {i}: RSID '{rsid_raw}' is invalid (must start with rs).")
            continue

        if not valid_genotype(gt):
            errors.append(f"Row {i}: Genotype '{gt_raw}' is invalid (use AA/AG/CT etc.).")
            continue

        cleaned_inputs.append({"rsid": rsid, "genotype": gt})

    if errors:
        st.error("⚠️ Please fix these input errors:")
        for e in errors:
            st.write("- " + e)

    if not cleaned_inputs:
        st.warning("⚠️ Please enter at least one valid RSID + Genotype.")
    else:
        # ✅ Load big GWAS from Google Drive (cached)
        gwas_df = load_gwas_from_drive(GWAS_DRIVE_FILE_ID)

        thresholds_df = load_thresholds(THRESHOLDS_FILE)
        st.session_state["thresholds_df"] = thresholds_df  # persist for narration block
        trait_map = load_trait_map(TRAIT_MAP_FILE)

        # Normalize GWAS rsid
        if "rsid" in gwas_df.columns:
            gwas_df["rsid"] = gwas_df["rsid"].astype(str).str.lower().str.strip()
        elif "SNPS" in gwas_df.columns:
            gwas_df["rsid"] = gwas_df["SNPS"].astype(str).str.lower().str.strip()
        else:
            st.error("GWAS file must contain an 'rsid' or 'SNPS' column.")
            st.stop()

        threshold_dict = {
            row["analyte"]: row["flag_logic"]
            for _, row in thresholds_df.iterrows()
            if "analyte" in thresholds_df.columns and "flag_logic" in thresholds_df.columns
        }

        user_gt_map = {d["rsid"]: d["genotype"] for d in cleaned_inputs}
        rsid_list = list(user_gt_map.keys())

        gwas_hits = gwas_df[gwas_df["rsid"].isin(rsid_list)].copy()
        if gwas_hits.empty:
            st.warning("No RSIDs were found in the GWAS dataset.")
            st.stop()

        # ── P-VALUE FILTER ────────────────────────────────────────────────────
        # Only keep genome-wide significant associations (p < 5e-8).
        # This removes borderline / unreplicated hits that could mislead the report.
        p_col = None
        for c in ["P-VALUE", "P_VALUE", "p-value", "p_value", "PVALUE"]:
            if c in gwas_hits.columns:
                p_col = c
                break

        if p_col:
            before = len(gwas_hits)
            gwas_hits[p_col] = pd.to_numeric(gwas_hits[p_col], errors="coerce")
            gwas_hits = gwas_hits[gwas_hits[p_col] < 5e-8].copy()
            after = len(gwas_hits)
            if after == 0:
                st.warning(
                    f"All {before} GWAS hit(s) for your RSIDs were filtered out because their "
                    f"p-values did not meet genome-wide significance (p < 5×10⁻⁸). "
                    "These associations may be real but are not yet robustly replicated."
                )
                st.stop()
            elif after < before:
                st.info(
                    f"ℹ️ P-value filter applied: {before - after} sub-threshold association(s) removed "
                    f"(p ≥ 5×10⁻⁸). {after} genome-wide significant hit(s) retained."
                )
        else:
            st.info("ℹ️ No p-value column found in GWAS data — p-value filter skipped.")

        allowed_analytes = list(user_values.keys())
        gwas_hits = filter_gwas_to_supported_traits(gwas_hits, trait_map, allowed_analytes)
        if gwas_hits.empty:
            st.warning("RSIDs found, but none of the linked traits match your supported analytes/traits.")
            st.stop()

        fused_rows = []
        for _, g in gwas_hits.iterrows():
            trait = g.get("DISEASE/TRAIT", "")
            rsid = g["rsid"]
            gene = g.get("MAPPED_GENE", g.get("REPORTED GENE(S)", ""))

            user_gt = user_gt_map.get(rsid, "")
            risk_allele = extract_risk_allele_from_row(g)
            effect_size = extract_effect_size(g)
            genetic_risk, weighted_score = compute_genetic_risk(user_gt, risk_allele, effect_size)

            analytes = g.get("__resolved_analytes", None)
            if not analytes:
                continue

            for analyte in analytes:
                if analyte not in user_values:
                    continue

                value = user_values[analyte]
                flag_logic = threshold_dict.get(analyte)
                flag = parse_flag_logic(flag_logic, value) if flag_logic else "Not available"

                fused_rows.append({
                    "patient_id": "ManualInput",
                    "rsid": rsid,
                    "user_genotype": user_gt,
                    "risk_allele": risk_allele if risk_allele else "Unknown",
                    "effect_size": round(effect_size, 3) if effect_size is not None else "N/A",
                    "genetic_risk": genetic_risk,
                    "weighted_score": weighted_score,
                    "trait": trait,
                    "analyte": analyte,
                    "lab_value": value,
                    "lab_flag": flag,
                    "gene": gene,
                })

        if not fused_rows:
            st.warning("No fused rows were produced after filtering.")
            st.stop()

        fused_df = pd.DataFrame(fused_rows)
        st.session_state["fused_df"] = fused_df
        st.session_state["narrated"] = False
        st.session_state.pop("narration_results", None)
        st.session_state.pop("analyte_summary_df", None)

        st.subheader("🔗 Fused Genetic + Lab Table (Filtered Traits Only)")
        st.dataframe(fused_df, use_container_width=True)

        analyte_summary_df = build_analyte_summary(fused_df)
        st.session_state["analyte_summary_df"] = analyte_summary_df

        st.subheader("📌 Analyte-Level Summary (Recommended for narration)")
        st.dataframe(analyte_summary_df, use_container_width=True)


# -----------------------------
# NARRATION (Gemini) - analyte level
# -----------------------------
if "analyte_summary_df" in st.session_state and not st.session_state.get("narrated", False):
    if st.button("🗣️ Narrate This Report (Gemini)"):
        narration_results = []
        analyte_df = st.session_state["analyte_summary_df"].copy()
        thresholds_df = st.session_state.get("thresholds_df", pd.DataFrame())  # retrieve from session state

        for _, row in analyte_df.iterrows():
            analyte = row["analyte"]
            lab_value = row["lab_value"]
            lab_flag = row["lab_flag"]

            influence = row["genetic_influence"]
            weighted_score = row.get("weighted_genetic_score", "N/A")
            n_high = row["n_high"]
            n_mod = row["n_moderate"]
            n_low = row.get("n_low", 0)
            n_unc = row["n_uncertain"]
            n_unk = row["n_unknown"]

            rsids = row["rsids"]
            traits = row["traits"]
            genes = row.get("genes", "")

            # Look up unit for this analyte
            unit_row = thresholds_df[thresholds_df["analyte"] == analyte]
            unit = unit_row["unit"].iloc[0] if not unit_row.empty and "unit" in unit_row.columns else ""

            prompt = f"""
You are a careful, friendly health explainer. Do NOT diagnose. Use cautious language like "may", "is associated with", "could indicate".
If genetic evidence is Limited/Unknown/Uncertain, say genetics is inconclusive and focus more on lab interpretation.
Keep it short and patient-friendly.

Analyte: {analyte} (measured in {unit})
Measured value: {lab_value} {unit}
Lab interpretation: {lab_flag}

Genetic summary for this analyte:
- Genetic influence level: {influence}
- Weighted genetic risk score: {weighted_score} (higher = stronger combined effect of risk alleles × effect sizes)
- High risk markers: {n_high}
- Moderate risk markers: {n_mod}
- Low/no-risk markers: {n_low}
- Uncertain markers: {n_unc}
- Unknown markers: {n_unk}
- Related RSIDs: {rsids}
- Related traits (GWAS, genome-wide significant only): {traits}
- Related genes (if available): {genes}

Write 3–4 short sentences:
1) What this analyte measures and what the value ({lab_value} {unit}) means clinically.
2) How the lab result looks given the flag ({lab_flag}).
3) How genetics may or may not relate — reference the influence level and weighted score.
4) One practical, actionable tip for this specific analyte.
"""
            narration = gemini_narrate(prompt)
            narration_results.append((analyte, narration))

        st.session_state["narration_results"] = narration_results
        st.session_state["narrated"] = True
        st.success("✅ Narration Generated Successfully using Gemini!")


if st.session_state.get("narration_results"):
    st.subheader("📣 Layperson-Friendly Narration (Analyte-wise)")
    for i, (analyte, narration) in enumerate(st.session_state["narration_results"], start=1):
        st.markdown(f"### {i}. {analyte}")
        st.write(narration)

    all_text = "\n\n".join([f"{a}: {t}" for a, t in st.session_state["narration_results"]])
    summary_prompt = f"""
Summarize this entire health report in 4–6 calm, friendly sentences.
Do NOT diagnose. Mention which areas look most important based on labs first, then genetics as supportive context.
End with 2 simple next steps.

Report text:
{all_text}
"""
    overall = gemini_narrate(summary_prompt)
    st.subheader("🩵 Overall Health Summary")
    st.write(overall)


st.markdown(
    """
    <div style='background-color:#fff3cd; padding:10px; border-radius:10px;
                border:2px solid #f5c06f; text-align:center; font-size:18px;
                color:#856404; font-weight:bold;'>
    ⚠️ DISCLAIMER: This application is for informational and educational purposes only.
    It does <u>not</u> replace consultation with qualified healthcare professionals.
    </div>
    """,
    unsafe_allow_html=True
)

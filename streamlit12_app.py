import os
import json
import streamlit as st
import pandas as pd
import yaml

# Gemini SDK
from google import genai

# Trait → analyte resolver
from trait_analyte_resolver import filter_gwas_to_supported_traits

# Robust Drive download
import requests
from io import BytesIO

# -----------------------------
# CONFIG
# -----------------------------
GWAS_DRIVE_FILE_ID = "1F92lf8My0699QVdCfPx_HbYmgc_WHjLx"
THRESHOLDS_FILE = "trusted_lab_thresholds.csv"
TRAIT_MAP_FILE = "trait_to_labs_partial.yaml"

DEFAULT_GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")  # default to lite


# -----------------------------
# GOOGLE DRIVE DOWNLOADER (BIG FILE SAFE)
# -----------------------------
def _download_gdrive_file(file_id: str) -> bytes:
    if not file_id:
        raise RuntimeError("GWAS_DRIVE_FILE_ID is not set.")

    session = requests.Session()
    url = "https://drive.google.com/uc?export=download"
    resp = session.get(url, params={"id": file_id}, stream=True)

    # Large file confirm token
    token = None
    for k, v in resp.cookies.items():
        if k.startswith("download_warning"):
            token = v
            break
    if token:
        resp = session.get(url, params={"id": file_id, "confirm": token}, stream=True)

    resp.raise_for_status()

    data = BytesIO()
    for chunk in resp.iter_content(chunk_size=1024 * 1024):  # 1MB
        if chunk:
            data.write(chunk)
    return data.getvalue()


@st.cache_data(show_spinner="Loading GWAS reference data… (first time may take 10–30s)")
def load_gwas_from_drive(file_id: str) -> pd.DataFrame:
    raw = _download_gdrive_file(file_id)
    df = pd.read_csv(BytesIO(raw), low_memory=False)

    # Normalize rsid column
    if "rsid" in df.columns:
        df["rsid"] = df["rsid"].astype(str).str.lower().str.strip()
    elif "SNPS" in df.columns:
        df["rsid"] = df["SNPS"].astype(str).str.lower().str.strip()
    else:
        raise RuntimeError("GWAS file must contain an 'rsid' or 'SNPS' column.")
    return df


@st.cache_data(show_spinner="Loading lab thresholds…")
def load_thresholds(path: str) -> pd.DataFrame:
    return pd.read_csv(path, low_memory=False)


@st.cache_data(show_spinner="Loading trait → lab mapping…")
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
        raise RuntimeError("GEMINI_API_KEY not set. Add it in Streamlit Secrets.")

    return genai.Client(api_key=api_key)


gemini_client = load_gemini_client()


def gemini_generate_once(prompt: str, model_name: str = DEFAULT_GEMINI_MODEL) -> str:
    """
    ONE-CALL Gemini generation.
    Includes fallback to flash if lite fails.
    """
    try:
        resp = gemini_client.models.generate_content(model=model_name, contents=prompt)
        return (resp.text or "").strip()
    except Exception:
        try:
            resp = gemini_client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
            return (resp.text or "").strip()
        except Exception as e2:
            return f"(Narration unavailable due to Gemini API error: {e2})"


# -----------------------------
# HELPERS
# -----------------------------
def parse_flag_logic(logic, value):
    try:
        parts = str(logic).split(";")
        for part in parts:
            part = part.strip()
            if ">=" in part:
                threshold, flag = part.split(":")
                if value >= float(threshold.replace(">=", "")):
                    return flag.strip()
            elif "<=" in part:
                threshold, flag = part.split(":")
                if value <= float(threshold.replace("<=", "")):
                    return flag.strip()
            elif ">" in part:
                threshold, flag = part.split(":")
                if value > float(threshold.replace(">", "")):
                    return flag.strip()
            elif "<" in part:
                threshold, flag = part.split(":")
                if value < float(threshold.replace("<", "")):
                    return flag.strip()
            elif "else" in part:
                _, flag = part.split(":")
                return flag.strip()
    except Exception:
        return "Unknown"
    return "Unknown"


def normalize_rsid(x: str) -> str:
    return str(x).strip().lower()


def normalize_genotype(x: str) -> str:
    return str(x).strip().upper().replace(" ", "")


def valid_genotype(gt: str) -> bool:
    return len(gt) == 2 and all(c in "ACGT" for c in gt)


def extract_risk_allele_from_row(row) -> str | None:
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


def compute_genetic_risk(user_gt: str, risk_allele: str | None) -> str:
    if not risk_allele:
        return "Unknown (risk allele missing in GWAS)"

    cnt = user_gt.count(risk_allele)
    if cnt == 2:
        return "High risk (homozygous effect allele)"
    elif cnt == 1:
        return "Moderate risk (heterozygous effect allele)"
    else:
        return "Uncertain: effect allele not detected in genotype (allele validation will improve in future)"


def _risk_bucket(genetic_risk: str) -> str:
    gr = (genetic_risk or "").lower()
    if gr.startswith("high"):
        return "high"
    if gr.startswith("moderate"):
        return "moderate"
    if gr.startswith("uncertain"):
        return "uncertain"
    return "unknown"


def compute_genetic_influence(n_high: int, n_moderate: int, n_uncertain: int, n_unknown: int) -> str:
    total = n_high + n_moderate + n_uncertain + n_unknown
    if total == 0:
        return "Weak"
    if n_high >= 1 or n_moderate >= 3:
        return "Strong"
    if 1 <= n_moderate <= 2:
        return "Moderate"
    if (n_unknown + n_uncertain) >= max(1, total // 2):
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
        n_unc = int(buckets.get("uncertain", 0))
        n_unk = int(buckets.get("unknown", 0))

        influence = compute_genetic_influence(n_high, n_mod, n_unc, n_unk)

        rows.append({
            "analyte": analyte,
            "lab_value": lab_value,
            "lab_flag": lab_flag,
            "genetic_influence": influence,
            "n_high": n_high,
            "n_moderate": n_mod,
            "n_uncertain": n_unc,
            "n_unknown": n_unk,
            "rsids": ", ".join(rsids),
            "traits": "; ".join(traits[:6]) + (" ..." if len(traits) > 6 else ""),
            "genes": ", ".join(genes[:8]) + (" ..." if len(genes) > 8 else ""),
        })

    out = pd.DataFrame(rows)

    # Simple sort (flag text is not strictly ordered; still ok)
    return out.sort_values(by=["analyte"], ascending=[True])


def build_single_prompt(analyte_summary_df: pd.DataFrame) -> str:
    """
    Build ONE prompt for Gemini:
      - Returns JSON so parsing is easy.
      - Contains analyte-wise narration + overall summary.
    """
    analytes_payload = analyte_summary_df.to_dict(orient="records")

    return f"""
You are a careful, friendly health explainer. DO NOT diagnose. Use cautious language like "may", "is associated with", "could indicate".
Focus on lab interpretation first. Use genetics only as supportive context, and if genetics is "Limited evidence" or many Unknown/Uncertain, say genetics is inconclusive.

TASK:
Return STRICT JSON ONLY (no markdown). Use exactly this schema:

{{
  "analytes": [
    {{
      "analyte": "string",
      "narration": "3-4 short patient-friendly sentences. End with one practical tip."
    }}
  ],
  "overall_summary": "4-6 calm, friendly sentences. Mention labs first, genetics second. End with 2 next steps."
}}

INPUT DATA (array of analyte objects):
{json.dumps(analytes_payload, ensure_ascii=False)}
""".strip()


def safe_parse_json(text: str) -> dict | None:
    """
    Attempts to parse model output into JSON.
    If Gemini adds extra text, try to extract the JSON block.
    """
    if not text:
        return None
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Try extract from first { to last }
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        chunk = text[start:end+1]
        try:
            return json.loads(chunk)
        except Exception:
            return None

    return None


# -----------------------------
# STREAMLIT UI
# -----------------------------
st.set_page_config(page_title="GeneNarrator+", layout="wide")

st.title("🧬 GeneNarrator+")
st.caption("Teal-themed, layperson-friendly narration from SNPs + labs (educational use only).")

with st.expander("✅ Setup checklist (only for you)", expanded=False):
    st.write("- Drive file must be public: Anyone with link → Viewer")
    st.write("- Streamlit Secrets must contain GEMINI_API_KEY")
    st.write("- Optional Secrets: GEMINI_MODEL=gemini-2.5-flash-lite")

st.divider()

# --- INPUTS ---
left, right = st.columns([1, 1], gap="large")

with left:
    st.subheader("🔬 Genetic Inputs")
    st.write("Add RSIDs and genotypes (AA/AG/CT etc.).")

    if "snp_inputs" not in st.session_state:
        st.session_state["snp_inputs"] = [{"rsid": "", "genotype": ""}]

    a, b, _ = st.columns([1, 1, 2])
    with a:
        if st.button("➕ Add"):
            st.session_state["snp_inputs"].append({"rsid": "", "genotype": ""})
    with b:
        if st.button("🗑 Remove") and len(st.session_state["snp_inputs"]) > 1:
            st.session_state["snp_inputs"].pop()

    for i, item in enumerate(st.session_state["snp_inputs"]):
        c1, c2 = st.columns([2, 1])
        item["rsid"] = c1.text_input(
            f"RSID {i+1}",
            value=item["rsid"],
            key=f"rsid_{i}",
            placeholder="e.g., rs1801133"
        )
        item["genotype"] = c2.text_input(
            f"Genotype {i+1}",
            value=item["genotype"],
            key=f"geno_{i}",
            placeholder="e.g., CT"
        )

with right:
    st.subheader("🧪 Blood Test Inputs")
    st.write("Adjust values (manual sliders).")

    analyte_defaults = {
        "Hemoglobin": (8.0, 18.0, 14.0),
        "LDL-C": (50, 200, 110),
        "HDL-C": (20, 100, 50),
        "Triglycerides": (50, 400, 140),
        "Fasting Glucose": (60, 200, 95),
        "HbA1c": (4.0, 12.0, 5.6),
        "Ferritin": (5, 400, 100),
        "TSH": (0.1, 10.0, 2.0),
        "25(OH)D": (5, 100, 35),
        "Creatinine": (0.4, 2.0, 1.0),
    }

    user_values = {}
    cols = st.columns(2)
    for idx, (analyte, (mn, mx, default)) in enumerate(analyte_defaults.items()):
        with cols[idx % 2]:
            user_values[analyte] = st.slider(analyte, mn, mx, default)

st.divider()
confirm = st.button("✅ Confirm & Analyze", use_container_width=True)

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
            continue  # allow blanks

        if not rsid.startswith("rs"):
            errors.append(f"Row {i}: RSID '{rsid_raw}' is invalid (must start with rs).")
            continue

        if not valid_genotype(gt):
            errors.append(f"Row {i}: Genotype '{gt_raw}' is invalid (use AA/AG/CT etc.).")
            continue

        cleaned_inputs.append({"rsid": rsid, "genotype": gt})

    if errors:
        st.error("⚠️ Fix these input errors:")
        for e in errors:
            st.write("- " + e)

    if not cleaned_inputs:
        st.warning("⚠️ Please enter at least one valid RSID + Genotype.")
        st.stop()

    # Load datasets
    gwas_df = load_gwas_from_drive(GWAS_DRIVE_FILE_ID)
    thresholds_df = load_thresholds(THRESHOLDS_FILE)
    trait_map = load_trait_map(TRAIT_MAP_FILE)

    # Threshold dict
    threshold_dict = {
        row["analyte"]: row["flag_logic"]
        for _, row in thresholds_df.iterrows()
        if "analyte" in thresholds_df.columns and "flag_logic" in thresholds_df.columns
    }

    # User genotype map
    user_gt_map = {d["rsid"]: d["genotype"] for d in cleaned_inputs}
    rsid_list = list(user_gt_map.keys())

    # Filter GWAS to input rsids
    gwas_hits = gwas_df[gwas_df["rsid"].isin(rsid_list)].copy()
    if gwas_hits.empty:
        st.warning("No RSIDs were found in the GWAS dataset.")
        st.stop()

    # Filter traits to supported analytes
    allowed_analytes = list(user_values.keys())
    gwas_hits = filter_gwas_to_supported_traits(gwas_hits, trait_map, allowed_analytes)
    if gwas_hits.empty:
        st.warning("RSIDs found, but none of the linked traits match your supported analytes/traits.")
        st.stop()

    # Build fused rows
    fused_rows = []
    for _, g in gwas_hits.iterrows():
        trait = g.get("DISEASE/TRAIT", "")
        rsid = g["rsid"]
        gene = g.get("MAPPED_GENE", g.get("REPORTED GENE(S)", ""))

        user_gt = user_gt_map.get(rsid, "")
        risk_allele = extract_risk_allele_from_row(g)
        genetic_risk = compute_genetic_risk(user_gt, risk_allele)

        analytes = g.get("__resolved_analytes", None)
        if not analytes:
            continue

        for analyte in analytes:
            if analyte not in user_values:
                continue

            value = float(user_values[analyte])
            flag_logic = threshold_dict.get(analyte)
            flag = parse_flag_logic(flag_logic, value) if flag_logic else "Not available"

            fused_rows.append({
                "patient_id": "ManualInput",
                "rsid": rsid,
                "user_genotype": user_gt,
                "risk_allele": risk_allele if risk_allele else "Unknown",
                "genetic_risk": genetic_risk,
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
    analyte_summary_df = build_analyte_summary(fused_df)

    st.session_state["fused_df"] = fused_df
    st.session_state["analyte_summary_df"] = analyte_summary_df
    st.session_state["narration_json"] = None
    st.success("✅ Analysis complete. Scroll down for results.")


# -----------------------------
# RESULTS DISPLAY
# -----------------------------
if "fused_df" in st.session_state and "analyte_summary_df" in st.session_state:
    fused_df = st.session_state["fused_df"]
    analyte_summary_df = st.session_state["analyte_summary_df"]

    st.subheader("🔗 Fused Genetic + Lab Table (Filtered Traits Only)")
    st.dataframe(fused_df, use_container_width=True)

    st.subheader("📌 Analyte-Level Summary (used for narration)")
    st.dataframe(analyte_summary_df, use_container_width=True)

    st.divider()

    # -----------------------------
    # ONE-CALL NARRATION
    # -----------------------------
    st.subheader("🗣️ Narration (One Gemini Call)")
    st.caption("This button uses only ONE request, saving your free-tier limits.")

    if st.button("Generate Narration", type="primary", use_container_width=True):
        prompt = build_single_prompt(analyte_summary_df)
        output = gemini_generate_once(prompt)

        parsed = safe_parse_json(output)
        if not parsed or "analytes" not in parsed or "overall_summary" not in parsed:
            st.error("Gemini did not return valid JSON. Showing raw output below.")
            st.code(output)
        else:
            st.session_state["narration_json"] = parsed
            st.success("✅ Narration generated in one call!")

    # Display narration
    if st.session_state.get("narration_json"):
        data = st.session_state["narration_json"]

        st.subheader("🩵 Overall Health Summary")
        st.write(data.get("overall_summary", ""))

        st.subheader("📣 Analyte Narrations")
        for i, item in enumerate(data.get("analytes", []), start=1):
            st.markdown(f"### {i}. {item.get('analyte','(unknown)')}")
            st.write(item.get("narration", ""))


# -----------------------------
# DISCLAIMER
# -----------------------------
st.markdown(
    """
    <div style='background-color:#fff3cd; padding:12px; border-radius:12px;
                border:2px solid #f5c06f; text-align:center; font-size:16px;
                color:#856404; font-weight:700;'>
    ⚠️ DISCLAIMER: This application is for informational and educational purposes only.
    It does <u>not</u> replace consultation with qualified healthcare professionals.
    </div>
    """,
    unsafe_allow_html=True
)

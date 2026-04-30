from __future__ import annotations

import io
import re
import textwrap
import zipfile
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import streamlit as st

try:
    from abnumber import Chain
    ABNUMBER_AVAILABLE = True
except Exception:
    Chain = None
    ABNUMBER_AVAILABLE = False


# =========================================================
# Config
# =========================================================
st.set_page_config(page_title="Antibody / Nanobody Region Extractor", page_icon="🧬", layout="wide")

REGION_ORDER = ["FR1", "CDR1", "FR2", "CDR2", "FR3", "CDR3", "FR4"]
REGION_WIDTHS = {"FR1": 3, "CDR1": 2, "FR2": 3, "CDR2": 2, "FR3": 4, "CDR3": 2, "FR4": 2}

VALID_AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
NUCLEOTIDE_LETTERS = set("ACGTUNRYSWKMBDHV")


# =========================================================
# Data model
# =========================================================
@dataclass
class FastaRecord:
    order: int
    header: str
    sequence: str
    source_file: str

    @property
    def sample_id(self) -> str:
        return self.header.split()[0] if self.header.strip() else f"record_{self.order}"


# =========================================================
# FASTA utilities
# =========================================================
def parse_fasta(text: str, source_file: str, start_order: int = 1) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    header: Optional[str] = None
    seq_lines: list[str] = []
    order = start_order

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if header is not None:
                records.append(
                    FastaRecord(
                        order=order,
                        header=header,
                        sequence="".join(seq_lines),
                        source_file=source_file,
                    )
                )
                order += 1
            header = line[1:].strip()
            seq_lines = []
        else:
            if header is None:
                header = source_file.rsplit(".", 1)[0]
            seq_lines.append(line)

    if header is not None:
        records.append(
            FastaRecord(
                order=order,
                header=header,
                sequence="".join(seq_lines),
                source_file=source_file,
            )
        )

    return records


def clean_sequence(seq: str, keep_stop: bool = True, keep_gap: bool = True) -> str:
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[\r\n0-9_.]", "", seq)

    allowed = set(VALID_AA_LETTERS)
    if not keep_stop:
        allowed.discard("*")
    if keep_gap:
        allowed.add("-")

    # Keep nucleotide letters too so auto-detect can still work before translation.
    allowed = allowed | NUCLEOTIDE_LETTERS
    return "".join(ch for ch in seq if ch in allowed)


def clean_aa_for_numbering(seq: str) -> str:
    seq = clean_sequence(seq, keep_stop=False, keep_gap=False)
    seq = seq.replace("-", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq


def wrap_fasta_sequence(seq: str, width: int = 80) -> str:
    if not seq:
        return ""
    return "\n".join(textwrap.wrap(seq, width=width))


# =========================================================
# Nucleotide translation helpers
# =========================================================
CODON_TABLE = {
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    "TGT": "C", "TGC": "C", "TGA": "*", "TGG": "W",
    "CGT": "R", "CGC": "R", "CGA": "R", "CGG": "R",
    "AGT": "S", "AGC": "S", "AGA": "R", "AGG": "R",
    "GGT": "G", "GGC": "G", "GGA": "G", "GGG": "G",
}


def looks_like_nucleotide(seq: str) -> bool:
    letters = [ch for ch in seq.upper() if ch.isalpha()]
    if not letters:
        return False
    nt_count = sum(ch in NUCLEOTIDE_LETTERS for ch in letters)
    return (nt_count / len(letters)) >= 0.90


def reverse_complement(seq: str) -> str:
    table = str.maketrans(
        "ACGTUNRYSWKMBDHVacgtunryswkmbdhv",
        "TGCAANRYSWMKVHDBtgcaanryswkmbdhv".upper(),
    )
    return seq.translate(table)[::-1].upper().replace("U", "T")


def translate_nt(seq: str, frame: int = 0) -> str:
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "N", seq)
    aa: list[str] = []
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i:i + 3]
        aa.append(CODON_TABLE.get(codon, "X"))
    return "".join(aa)


def candidate_aa_sequences(raw_seq: str, mode: str) -> list[tuple[str, str, str]]:
    cleaned = clean_sequence(raw_seq)

    if mode == "Amino acid FASTA":
        return [(cleaned.replace("*", "").replace("-", ""), "input", "+")]

    nt_only = re.sub(r"[^A-Za-z]", "", raw_seq).upper().replace("U", "T")
    nt_only = re.sub(r"[^ACGTN]", "N", nt_only)

    if mode == "Auto-detect; translate nucleotide if needed" and not looks_like_nucleotide(cleaned):
        return [(cleaned.replace("*", "").replace("-", ""), "input", "+")]

    candidates: list[tuple[str, str, str]] = []
    for strand_label, nt_seq in [("+", nt_only), ("-", reverse_complement(nt_only))]:
        for frame in range(3):
            aa = translate_nt(nt_seq, frame=frame).replace("*", "")
            candidates.append((aa, f"frame_{frame + 1}", strand_label))
    return candidates


# =========================================================
# Region extraction
# =========================================================
def get_regions_with_abnumber(aa_seq: str, scheme: str = "imgt") -> tuple[dict[str, str], str]:
    if not ABNUMBER_AVAILABLE or Chain is None:
        raise RuntimeError("abnumber is not installed or failed to import.")

    aa_seq = clean_aa_for_numbering(aa_seq)
    chain = Chain(aa_seq, scheme=scheme)

    region_map = {
        "FR1": getattr(chain, "fr1_seq", "") or "",
        "CDR1": getattr(chain, "cdr1_seq", "") or "",
        "FR2": getattr(chain, "fr2_seq", "") or "",
        "CDR2": getattr(chain, "cdr2_seq", "") or "",
        "FR3": getattr(chain, "fr3_seq", "") or "",
        "CDR3": getattr(chain, "cdr3_seq", "") or "",
        "FR4": getattr(chain, "fr4_seq", "") or "",
    }

    chain_type = getattr(chain, "chain_type", "unknown")
    return region_map, chain_type


def get_regions_with_fallback_heuristic(aa_seq: str) -> tuple[dict[str, str], str]:
    """
    Dependency-free approximate segmentation.

    This is less accurate than numbering, but useful if abnumber/ANARCI is unavailable.
    It tries common VH/VHH boundaries:
    - CDR1 around after first Cys and before W of FR2
    - CDR2 after conserved W region and before FR3 YY/FTIS motif
    - CDR3 between YYC/YxC and WGQG/FGQG/WGxG anchor

    For serious annotation, use abnumber mode.
    """
    seq = clean_aa_for_numbering(aa_seq)
    regions = {r: "" for r in REGION_ORDER}

    # CDR3: best supported heuristic.
    cdr3_match = None
    for m in re.finditer(r"C([A-Z]{3,80}?)([WF]G.Q|[WF]G.G|[WF]G..)", seq):
        c_pos = m.start()
        j_pos = m.start(2)
        if 70 <= c_pos <= 120 and 5 <= len(m.group(1)) <= 60:
            # Prefer YYC/YxC context and plausible CDR3 length.
            context = seq[max(0, c_pos - 2):c_pos + 1]
            score = 0
            if re.search(r"[YFH][YFH]C", context):
                score += 10
            score += min(len(m.group(1)), 20)
            candidate = (score, c_pos, j_pos, m.group(1))
            if cdr3_match is None or candidate > cdr3_match:
                cdr3_match = candidate

    if cdr3_match:
        _, c_pos, j_pos, cdr3 = cdr3_match
        regions["CDR3"] = cdr3
        regions["FR4"] = seq[j_pos:]

        # Split upstream approximately.
        upstream = seq[:c_pos + 1]
        regions["FR3"] = upstream[-38:] if len(upstream) >= 38 else upstream
        before_fr3 = upstream[:-len(regions["FR3"])] if regions["FR3"] else upstream
    else:
        before_fr3 = seq
        regions["FR3"] = ""
        regions["FR4"] = ""

    # Approximate CDR1 using first conserved Cys and W anchor.
    c1 = seq.find("C")
    w_after_c1 = seq.find("W", c1 + 1) if c1 >= 0 else -1
    if c1 >= 0 and w_after_c1 > c1:
        regions["FR1"] = seq[:c1 + 1]
        regions["CDR1"] = seq[c1 + 1:w_after_c1]
        # FR2/CDR2/FR3 approximation from remaining pre-FR3 sequence.
        mid = before_fr3[w_after_c1:] if w_after_c1 < len(before_fr3) else ""
        # CDR2 often starts after a short FR2 segment and is ~7-20 aa.
        if len(mid) > 25:
            regions["FR2"] = mid[:15]
            regions["CDR2"] = mid[15:30]
            if not regions["FR3"]:
                regions["FR3"] = mid[30:]
        else:
            regions["FR2"] = mid
    else:
        # Last-resort chunks, only so the app returns something readable.
        regions["FR1"] = seq[:25]
        regions["CDR1"] = seq[25:35]
        regions["FR2"] = seq[35:50]
        regions["CDR2"] = seq[50:65]
        if not regions["FR3"]:
            regions["FR3"] = seq[65:100]
        if not regions["CDR3"]:
            regions["CDR3"] = ""
        if not regions["FR4"]:
            regions["FR4"] = seq[100:]

    return regions, "unknown_fallback"


def get_regions(aa_seq: str, scheme: str, method: str) -> tuple[dict[str, str], str, str]:
    """
    Return region_map, chain_type, method_used.
    """
    if method == "abnumber / IMGT numbering":
        region_map, chain_type = get_regions_with_abnumber(aa_seq, scheme=scheme)
        return region_map, chain_type, "abnumber"

    if method == "fallback heuristic only":
        region_map, chain_type = get_regions_with_fallback_heuristic(aa_seq)
        return region_map, chain_type, "fallback_heuristic"

    # Auto mode.
    try:
        region_map, chain_type = get_regions_with_abnumber(aa_seq, scheme=scheme)
        return region_map, chain_type, "abnumber"
    except Exception:
        region_map, chain_type = get_regions_with_fallback_heuristic(aa_seq)
        return region_map, chain_type, "fallback_heuristic"


def get_best_numbered_candidate(raw_seq: str, input_mode: str, scheme: str, method: str):
    candidates = candidate_aa_sequences(raw_seq, input_mode)
    best = None
    last_error = ""

    for aa_seq, frame_label, strand_label in candidates:
        aa_clean = clean_aa_for_numbering(aa_seq)
        if len(aa_clean) < 45:
            continue

        try:
            region_map, chain_type, method_used = get_regions(aa_clean, scheme=scheme, method=method)
            non_empty_regions = sum(1 for v in region_map.values() if v)
            cdr_score = sum(len(region_map.get(r, "")) for r in ["CDR1", "CDR2", "CDR3"])
            score = non_empty_regions * 100 + cdr_score + len(region_map.get("FR3", ""))

            if best is None or score > best["score"]:
                best = {
                    "aa_sequence_used": aa_clean,
                    "frame": frame_label,
                    "strand": strand_label,
                    "chain_type": chain_type,
                    "region_map": region_map,
                    "method_used": method_used,
                    "score": score,
                }
        except Exception as e:
            last_error = str(e)

    return best, last_error


def build_result_rows(records: list[FastaRecord], input_mode: str, scheme: str, method: str) -> pd.DataFrame:
    rows = []
    for r in records:
        best, last_error = get_best_numbered_candidate(
            r.sequence,
            input_mode=input_mode,
            scheme=scheme,
            method=method,
        )

        row = {
            "order": r.order,
            "source_file": r.source_file,
            "sample_id": r.sample_id,
            "header": r.header,
            "input_length": len(clean_sequence(r.sequence)),
            "status": "ok" if best else "not_found",
            "reason": "" if best else f"Could not segment sequence. {last_error}",
            "frame": best["frame"] if best else "",
            "strand": best["strand"] if best else "",
            "chain_type": best["chain_type"] if best else "",
            "method_used": best["method_used"] if best else "",
            "score": best["score"] if best else 0,
        }

        for region in REGION_ORDER:
            row[region] = best["region_map"].get(region, "") if best else ""
            row[f"{region}_len"] = len(row[region])

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# FASTA output
# =========================================================
def make_region_fasta(
    df: pd.DataFrame,
    region_name: str,
    header_mode: str = "Full original header",
    keep_failed_records: bool = True,
) -> str:
    lines = []
    for _, row in df.sort_values("order").iterrows():
        if row["status"] != "ok" and not keep_failed_records:
            continue
        header = row["sample_id"] if header_mode == "Sample ID only" else row["header"]
        seq = row.get(region_name, "") or ""
        lines.append(f">{header}")
        lines.append(wrap_fasta_sequence(seq))
    return "\n".join(lines).rstrip() + "\n"


def make_combined_fasta(
    df: pd.DataFrame,
    selected_regions: list[str],
    header_mode: str = "Full original header",
    keep_failed_records: bool = True,
    separator: str = "",
) -> str:
    lines = []
    for _, row in df.sort_values("order").iterrows():
        if row["status"] != "ok" and not keep_failed_records:
            continue
        header = row["sample_id"] if header_mode == "Sample ID only" else row["header"]
        seq_parts = [(row.get(region, "") or "") for region in selected_regions]
        seq = separator.join(seq_parts)
        lines.append(f">{header}")
        lines.append(wrap_fasta_sequence(seq))
    return "\n".join(lines).rstrip() + "\n"


def build_zip_file(file_map: dict[str, str]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        for filename, content in file_map.items():
            zf.writestr(filename, content)
    buffer.seek(0)
    return buffer.getvalue()


# =========================================================
# UI helpers
# =========================================================
def toggle_region(region: str):
    key = f"pick_{region}"
    st.session_state[key] = not st.session_state.get(key, False)


def set_region_group(region_list: list[str], value: bool):
    for region in REGION_ORDER:
        if region in region_list:
            st.session_state[f"pick_{region}"] = value


def reset_to_only(region_list: list[str]):
    for region in REGION_ORDER:
        st.session_state[f"pick_{region}"] = region in region_list


def render_linear_map():
    segments = []
    for region in REGION_ORDER:
        selected = st.session_state.get(f"pick_{region}", False)
        base_color = "#DCEBFF" if region.startswith("FR") else "#FFE4BF"
        bg = "#2E7D32" if selected else base_color
        fg = "white" if selected else "#222222"
        flex = REGION_WIDTHS[region]
        segments.append(
            f"""
            <div style="
                flex:{flex};
                padding:12px 4px;
                text-align:center;
                font-weight:700;
                border-right:1px solid #ffffff;
                background:{bg};
                color:{fg};
            ">{region}</div>
            """
        )

    html = f"""
    <div style="
        display:flex;
        border:1px solid #CCCCCC;
        border-radius:10px;
        overflow:hidden;
        margin-bottom:8px;
    ">
        {''.join(segments)}
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# App
# =========================================================
st.title("Antibody / Nanobody Region Extractor")
st.caption(
    "Upload FASTA files, click VH/VHH regions to select what to extract, "
    "then download separate FASTA files for each selected region."
)

if not ABNUMBER_AVAILABLE:
    st.warning(
        "abnumber is not available in this environment. "
        "The app can still run using the fallback heuristic, but region boundaries are approximate. "
        "For best results, install dependencies from requirements.txt."
    )

with st.sidebar:
    st.header("Input settings")
    input_mode = st.radio(
        "Sequence type",
        options=[
            "Amino acid FASTA",
            "Auto-detect; translate nucleotide if needed",
            "Force nucleotide translation, all 6 frames",
        ],
        index=0,
    )

    method = st.radio(
        "Region segmentation method",
        options=[
            "auto: abnumber first, fallback heuristic if needed",
            "abnumber / IMGT numbering",
            "fallback heuristic only",
        ],
        index=0,
        help="abnumber/IMGT is recommended. Fallback is approximate and dependency-free.",
    )

    scheme = st.selectbox("Numbering scheme", ["imgt"], index=0)

    st.header("Output settings")
    header_mode = st.radio(
        "Output FASTA header",
        options=["Full original header", "Sample ID only"],
        index=0,
    )

    keep_failed_records = st.checkbox(
        "Keep failed records as blank FASTA entries",
        value=True,
        help="Keeps the same number/order of records in each output FASTA.",
    )

    create_combined = st.checkbox(
        "Also create combined FASTA of all selected regions",
        value=True,
        help="If multiple regions are selected, also export one FASTA that concatenates them in region order.",
    )

    separator_choice = st.selectbox(
        "Combined-region separator",
        options=["none", "X", "GGGS"],
        index=0,
        help="Optional separator inserted between selected regions in the combined FASTA.",
    )
    separator = "" if separator_choice == "none" else separator_choice

uploaded_files = st.file_uploader(
    "Upload FASTA file(s)",
    type=["fasta", "fa", "faa", "fna", "txt"],
    accept_multiple_files=True,
)

example = """>example_vhh_01
QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGTTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAREGPYYYGMDYWGQGTQVTVSS
>example_vh_01
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLGGYYFDYWGQGTLVTVSS
>example_vhh_with_gaps
QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGTTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC---AREGPYYYGMDY---WGQGTQVTVSS
"""

use_example = st.checkbox("Use built-in example instead of uploaded files", value=False)

records: list[FastaRecord] = []
if use_example:
    records = parse_fasta(example, source_file="example.fasta", start_order=1)
elif uploaded_files:
    next_order = 1
    for file in uploaded_files:
        text = file.getvalue().decode("utf-8", errors="replace")
        parsed = parse_fasta(text, source_file=file.name, start_order=next_order)
        records.extend(parsed)
        next_order += len(parsed)

if not records:
    st.info("Upload FASTA files or enable the built-in example.")
    st.stop()


# ---------------------------------------------------------
# Region selection
# ---------------------------------------------------------
st.subheader("1) Click region(s) to extract")

action_cols = st.columns(6)
with action_cols[0]:
    if st.button("Select all", use_container_width=True):
        reset_to_only(REGION_ORDER)
with action_cols[1]:
    if st.button("Clear all", use_container_width=True):
        reset_to_only([])
with action_cols[2]:
    if st.button("All CDRs", use_container_width=True):
        reset_to_only(["CDR1", "CDR2", "CDR3"])
with action_cols[3]:
    if st.button("All FRs", use_container_width=True):
        reset_to_only(["FR1", "FR2", "FR3", "FR4"])
with action_cols[4]:
    if st.button("CDR3 only", use_container_width=True):
        reset_to_only(["CDR3"])
with action_cols[5]:
    if st.button("CDR1+2+3", use_container_width=True):
        reset_to_only(["CDR1", "CDR2", "CDR3"])

# Default selection on first load.
if not any(f"pick_{r}" in st.session_state for r in REGION_ORDER):
    reset_to_only(["CDR3"])

render_linear_map()

button_cols = st.columns([REGION_WIDTHS[r] for r in REGION_ORDER])
for col, region in zip(button_cols, REGION_ORDER):
    selected = st.session_state.get(f"pick_{region}", False)
    label = f"✅ {region}" if selected else region
    with col:
        st.button(label, key=f"btn_{region}", on_click=toggle_region, args=(region,), use_container_width=True)

selected_regions = [r for r in REGION_ORDER if st.session_state.get(f"pick_{r}", False)]

if not selected_regions:
    st.warning("Please select at least one region.")
    st.stop()

st.write("**Selected regions:**", " + ".join(selected_regions))


# ---------------------------------------------------------
# Processing
# ---------------------------------------------------------
df = build_result_rows(records, input_mode=input_mode, scheme=scheme, method=method)

n_total = len(df)
n_ok = int((df["status"] == "ok").sum())
n_failed = n_total - n_ok
n_fallback = int((df["method_used"] == "fallback_heuristic").sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Input records", n_total)
metric_cols[1].metric("Segmented", n_ok)
metric_cols[2].metric("Failed", n_failed)
metric_cols[3].metric("Fallback calls", n_fallback)
metric_cols[4].metric("Selected regions", len(selected_regions))

st.subheader("2) Results")
display_cols = [
    "order", "source_file", "sample_id", "status", "chain_type",
    "method_used", "frame", "strand", "reason",
] + selected_regions + [f"{r}_len" for r in selected_regions]
st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Output files
# ---------------------------------------------------------
st.subheader("3) Download output")

file_map: dict[str, str] = {}
for region in selected_regions:
    file_map[f"extracted_{region}.fasta"] = make_region_fasta(
        df,
        region,
        header_mode=header_mode,
        keep_failed_records=keep_failed_records,
    )

if create_combined and len(selected_regions) > 1:
    combined_name = "_".join(selected_regions)
    file_map[f"extracted_{combined_name}.fasta"] = make_combined_fasta(
        df,
        selected_regions,
        header_mode=header_mode,
        keep_failed_records=keep_failed_records,
        separator=separator,
    )

file_map["region_extraction_qc.csv"] = df.to_csv(index=False)

zip_bytes = build_zip_file(file_map)

st.download_button(
    "Download all selected outputs as ZIP",
    data=zip_bytes,
    file_name="region_extraction_outputs.zip",
    mime="application/zip",
    use_container_width=True,
)

st.markdown("### Individual preview")
for region in selected_regions:
    with st.expander(f"Preview {region} FASTA"):
        st.code(
            make_region_fasta(
                df,
                region,
                header_mode=header_mode,
                keep_failed_records=keep_failed_records,
            )[:10000],
            language="text",
        )

if create_combined and len(selected_regions) > 1:
    combined_name = "_".join(selected_regions)
    with st.expander(f"Preview combined FASTA ({combined_name})"):
        st.code(
            make_combined_fasta(
                df,
                selected_regions,
                header_mode=header_mode,
                keep_failed_records=keep_failed_records,
                separator=separator,
            )[:10000],
            language="text",
        )

with st.expander("Notes / limitations"):
    st.markdown(
        """
        - Recommended mode: **abnumber / IMGT numbering**.
        - Fallback mode is approximate and mainly useful if `abnumber` cannot be installed.
        - Gaps (`-`) are removed before numbering/segmentation.
        - If you upload nucleotide FASTA, the app can translate and try all six frames.
        - Non-antibody, heavily truncated, or highly engineered sequences may fail or produce uncertain segmentation.
        - The output ZIP contains one FASTA per selected region plus a QC CSV.
        """
    )

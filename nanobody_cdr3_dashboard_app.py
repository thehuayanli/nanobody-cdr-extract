"""
Streamlit dashboard: extract nanobody/VHH CDR3 regions from FASTA files.

Input
-----
- One or more FASTA files.
- Each file may contain one or many sequences.
- Sequences are expected to be amino-acid nanobody/VHH variable-domain sequences.
- Optional auto-translation mode can handle nucleotide coding sequences and tries all 6 frames.

Output
------
- FASTA containing extracted CDR3 sequences in the same record order.
- Headers/sample IDs are preserved by default.
- QC table with extraction status, CDR3 length, motif positions, frame, and confidence.

Run
---
    pip install streamlit pandas
    streamlit run nanobody_cdr3_dashboard_app.py

Notes
-----
This app uses a motif-based heuristic:
- CDR3 starts after the conserved heavy-chain/VHH Cys anchor, often within a YYC/YxC motif.
- CDR3 ends before the downstream J-region F/W-G-x-G motif, often WGQG.

For regulatory, publication, or final repertoire annotation, validate against a numbering tool
such as ANARCI, IgBLAST, or an in-house IMGT-numbering workflow.
"""

from __future__ import annotations

import io
import re
import textwrap
from dataclasses import dataclass
from typing import Iterable, Optional

import pandas as pd
import streamlit as st


# -----------------------------
# Data models
# -----------------------------

@dataclass
class FastaRecord:
    order: int
    header: str
    sequence: str
    source_file: str

    @property
    def sample_id(self) -> str:
        return self.header.split()[0] if self.header.strip() else f"record_{self.order}"


@dataclass
class ExtractionResult:
    order: int
    source_file: str
    sample_id: str
    header: str
    input_length: int
    aa_sequence_used: str
    aa_length_used: int
    cdr3: str
    cdr3_length: int
    status: str
    confidence: str
    reason: str
    c_anchor_1based: Optional[int]
    j_anchor_1based: Optional[int]
    j_motif: str
    frame: str
    strand: str
    score: float


# -----------------------------
# FASTA parsing and formatting
# -----------------------------

VALID_AA_LETTERS = set("ACDEFGHIKLMNPQRSTVWYBXZJUO*")
NUCLEOTIDE_LETTERS = set("ACGTUNRYSWKMBDHV")


def parse_fasta(text: str, source_file: str, start_order: int = 1) -> list[FastaRecord]:
    """Parse FASTA text while preserving full headers and record order."""
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
                # Graceful fallback for non-FASTA text: create one synthetic record.
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


def clean_sequence(seq: str, keep_stop: bool = True) -> str:
    """Remove whitespace, numbers, FASTA gaps, and unusual symbols."""
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[\r\n0-9\-_.]", "", seq)
    allowed = VALID_AA_LETTERS if keep_stop else (VALID_AA_LETTERS - {"*"})
    return "".join(ch for ch in seq if ch in allowed or ch in NUCLEOTIDE_LETTERS)


def wrap_fasta_sequence(seq: str, width: int = 80) -> str:
    if not seq:
        return ""
    return "\n".join(textwrap.wrap(seq, width=width))


def make_output_fasta(
    results: list[ExtractionResult],
    keep_failed_records: bool = True,
    header_mode: str = "Full original header",
    add_status_to_failed_headers: bool = False,
) -> str:
    """Create CDR3 FASTA in input order."""
    lines: list[str] = []

    for r in sorted(results, key=lambda x: x.order):
        if r.status != "ok" and not keep_failed_records:
            continue

        if header_mode == "Sample ID only":
            header = r.sample_id
        else:
            header = r.header

        # Preserve requested header/sample ID. Optional suffix only for failed records.
        if r.status != "ok" and add_status_to_failed_headers:
            header = f"{header} | CDR3_NOT_FOUND | {r.reason}"

        lines.append(f">{header}")
        lines.append(wrap_fasta_sequence(r.cdr3))

    return "\n".join(lines).rstrip() + "\n"


# -----------------------------
# Nucleotide translation utilities
# -----------------------------

CODON_TABLE = {
    # Phenylalanine / Leucine
    "TTT": "F", "TTC": "F", "TTA": "L", "TTG": "L",
    "CTT": "L", "CTC": "L", "CTA": "L", "CTG": "L",
    # Isoleucine / Methionine / Valine
    "ATT": "I", "ATC": "I", "ATA": "I", "ATG": "M",
    "GTT": "V", "GTC": "V", "GTA": "V", "GTG": "V",
    # Serine / Proline / Threonine / Alanine
    "TCT": "S", "TCC": "S", "TCA": "S", "TCG": "S",
    "CCT": "P", "CCC": "P", "CCA": "P", "CCG": "P",
    "ACT": "T", "ACC": "T", "ACA": "T", "ACG": "T",
    "GCT": "A", "GCC": "A", "GCA": "A", "GCG": "A",
    # Tyrosine / Histidine / Glutamine / Asparagine / Lysine / Aspartate / Glutamate
    "TAT": "Y", "TAC": "Y", "TAA": "*", "TAG": "*",
    "CAT": "H", "CAC": "H", "CAA": "Q", "CAG": "Q",
    "AAT": "N", "AAC": "N", "AAA": "K", "AAG": "K",
    "GAT": "D", "GAC": "D", "GAA": "E", "GAG": "E",
    # Cysteine / Tryptophan / Arginine / Glycine
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
    table = str.maketrans("ACGTUNRYSWKMBDHVacgtunryswkmbdhv", "TGCAANRYSWMKVHDBtgcaanryswmkvhdb")
    return seq.translate(table)[::-1].upper().replace("U", "T")


def translate_nt(seq: str, frame: int = 0) -> str:
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGT]", "N", seq)
    aa: list[str] = []
    for i in range(frame, len(seq) - 2, 3):
        codon = seq[i : i + 3]
        aa.append(CODON_TABLE.get(codon, "X"))
    return "".join(aa)


def candidate_aa_sequences(raw_seq: str, mode: str) -> list[tuple[str, str, str]]:
    """Return candidate AA sequences as (aa_seq, frame_label, strand_label)."""
    cleaned = clean_sequence(raw_seq)

    if mode == "Amino acid FASTA":
        return [(cleaned.replace("*", ""), "input", "+")]

    nt_only = re.sub(r"[^A-Za-z]", "", raw_seq).upper().replace("U", "T")
    nt_only = re.sub(r"[^ACGTN]", "N", nt_only)

    if mode == "Auto-detect; translate nucleotide if needed" and not looks_like_nucleotide(cleaned):
        return [(cleaned.replace("*", ""), "input", "+")]

    candidates: list[tuple[str, str, str]] = []
    for strand_label, nt_seq in [("+", nt_only), ("-", reverse_complement(nt_only))]:
        for frame in range(3):
            aa = translate_nt(nt_seq, frame=frame)
            # Keep internal X but remove stops because the VHH domain should be continuous.
            aa = aa.replace("*", "")
            candidates.append((aa, f"frame_{frame + 1}", strand_label))
    return candidates


# -----------------------------
# CDR3 extraction logic
# -----------------------------

@dataclass
class CDR3Candidate:
    cdr3: str
    c_anchor: int
    j_anchor: int
    j_motif: str
    score: float
    confidence: str
    reason: str


J_ANCHOR_PATTERNS = [
    (re.compile(r"[WF]GQG"), "strong_WF_GQG", 30),
    (re.compile(r"[WF]G.G"), "medium_WF_GxG", 22),
    (re.compile(r"[WF]G.."), "weak_WF_Gxx", 10),
]


def score_c_anchor_context(seq: str, c_pos: int) -> tuple[float, list[str]]:
    """Score whether a Cys looks like the conserved pre-CDR3 Cys anchor."""
    score = 0.0
    notes: list[str] = []
    left5 = seq[max(0, c_pos - 5) : c_pos + 1]
    left3 = seq[max(0, c_pos - 2) : c_pos + 1]

    if re.search(r"YYC$", left5):
        score += 35
        notes.append("YYC C-anchor")
    elif re.search(r"[YFH][YFH]C$", left5):
        score += 26
        notes.append("aromatic-aromatic-C anchor")
    elif re.search(r"[YFHW][A-Z]C$", left5):
        score += 18
        notes.append("Y/F/W-x-C anchor")
    elif left3.endswith("C"):
        score += 6
        notes.append("C anchor without strong YYC context")

    # Full VHH variable domains usually place this anchor around aa 85-115;
    # keep this soft because users may upload partial or fusion sequences.
    if 80 <= c_pos <= 115:
        score += 12
        notes.append("C position typical for VHH variable domain")
    elif 60 <= c_pos <= 140:
        score += 6
        notes.append("C position plausible")

    return score, notes


def confidence_from_score(score: float) -> str:
    if score >= 92:
        return "high"
    if score >= 68:
        return "medium"
    if score >= 45:
        return "low"
    return "very_low"


def extract_cdr3_from_aa(
    aa_seq: str,
    min_len: int = 5,
    max_len: int = 60,
    require_strong_c_anchor: bool = False,
) -> Optional[CDR3Candidate]:
    """Extract CDR3 from an amino-acid sequence using Cys and F/W-G-x-G anchors."""
    seq = clean_sequence(aa_seq, keep_stop=False)
    if len(seq) < 50:
        return None

    candidates: list[CDR3Candidate] = []

    for pattern, j_label, j_score in J_ANCHOR_PATTERNS:
        for j_match in pattern.finditer(seq):
            j_start = j_match.start()
            j_motif = j_match.group(0)

            # Avoid very early false-positive WG motifs.
            if j_start < 45:
                continue

            c_positions = [m.start() for m in re.finditer("C", seq[:j_start])]
            for c_pos in c_positions:
                cdr3 = seq[c_pos + 1 : j_start]
                cdr3_len = len(cdr3)
                if cdr3_len < min_len or cdr3_len > max_len:
                    continue

                anchor_score, c_notes = score_c_anchor_context(seq, c_pos)
                if require_strong_c_anchor and anchor_score < 18:
                    continue

                # Length prior: nanobody CDR3s are often longer than conventional VH,
                # but use a broad range to avoid rejecting valid engineered clones.
                if 8 <= cdr3_len <= 30:
                    len_score = 20
                elif 5 <= cdr3_len <= 45:
                    len_score = 14
                else:
                    len_score = 8

                # Prefer C anchors not immediately next to the J motif; those are often Cys within CDR3.
                distance_score = min(8, cdr3_len / 4)

                # Penalize suspicious sequences that include stop or too many unknowns.
                x_fraction = cdr3.count("X") / max(1, cdr3_len)
                x_penalty = 25 * x_fraction

                score = anchor_score + j_score + len_score + distance_score - x_penalty
                reason = "; ".join(c_notes + [j_label, f"CDR3 length={cdr3_len}"])
                candidates.append(
                    CDR3Candidate(
                        cdr3=cdr3,
                        c_anchor=c_pos,
                        j_anchor=j_start,
                        j_motif=j_motif,
                        score=score,
                        confidence=confidence_from_score(score),
                        reason=reason,
                    )
                )

    if not candidates:
        return None

    # Highest score wins. Tie-breaker: choose the upstream C anchor, which reduces the chance
    # of accidentally treating an internal CDR3 cysteine as the conserved C anchor.
    candidates.sort(key=lambda c: (c.score, -c.c_anchor), reverse=True)
    return candidates[0]


def extract_record(
    record: FastaRecord,
    input_mode: str,
    min_len: int,
    max_len: int,
    require_strong_c_anchor: bool,
) -> ExtractionResult:
    candidates = candidate_aa_sequences(record.sequence, input_mode)

    best: Optional[tuple[CDR3Candidate, str, str, str]] = None
    for aa_seq, frame_label, strand_label in candidates:
        c = extract_cdr3_from_aa(
            aa_seq=aa_seq,
            min_len=min_len,
            max_len=max_len,
            require_strong_c_anchor=require_strong_c_anchor,
        )
        if c is None:
            continue
        if best is None or c.score > best[0].score:
            best = (c, aa_seq, frame_label, strand_label)

    input_len = len(clean_sequence(record.sequence))

    if best is None:
        first_aa = candidates[0][0] if candidates else ""
        return ExtractionResult(
            order=record.order,
            source_file=record.source_file,
            sample_id=record.sample_id,
            header=record.header,
            input_length=input_len,
            aa_sequence_used=first_aa,
            aa_length_used=len(first_aa),
            cdr3="",
            cdr3_length=0,
            status="not_found",
            confidence="none",
            reason="No plausible Cys-to-F/W-G-x-G CDR3 boundary found under current length/settings.",
            c_anchor_1based=None,
            j_anchor_1based=None,
            j_motif="",
            frame="",
            strand="",
            score=0.0,
        )

    c, aa_seq, frame_label, strand_label = best
    return ExtractionResult(
        order=record.order,
        source_file=record.source_file,
        sample_id=record.sample_id,
        header=record.header,
        input_length=input_len,
        aa_sequence_used=aa_seq,
        aa_length_used=len(aa_seq),
        cdr3=c.cdr3,
        cdr3_length=len(c.cdr3),
        status="ok",
        confidence=c.confidence,
        reason=c.reason,
        c_anchor_1based=c.c_anchor + 1,
        j_anchor_1based=c.j_anchor + 1,
        j_motif=c.j_motif,
        frame=frame_label,
        strand=strand_label,
        score=round(c.score, 2),
    )


def results_to_dataframe(results: list[ExtractionResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "order": r.order,
                "source_file": r.source_file,
                "sample_id": r.sample_id,
                "status": r.status,
                "confidence": r.confidence,
                "cdr3": r.cdr3,
                "cdr3_length": r.cdr3_length,
                "c_anchor_1based": r.c_anchor_1based,
                "j_anchor_1based": r.j_anchor_1based,
                "j_motif": r.j_motif,
                "frame": r.frame,
                "strand": r.strand,
                "score": r.score,
                "reason": r.reason,
                "input_length": r.input_length,
                "aa_length_used": r.aa_length_used,
                "header": r.header,
            }
            for r in sorted(results, key=lambda x: x.order)
        ]
    )


# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Nanobody CDR3 Extractor", page_icon="🧬", layout="wide")

st.title("Nanobody/VHH CDR3 Extractor")
st.caption(
    "Upload FASTA files and extract CDR3 sequences while preserving input order and sample IDs. "
    "The downloadable FASTA uses the same headers by default."
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
        help="For clean nanobody protein sequences, use amino acid FASTA. For DNA coding sequences, use auto-detect or force translation.",
    )

    st.header("CDR3 boundary settings")
    min_len = st.number_input("Minimum CDR3 length", min_value=1, max_value=100, value=5, step=1)
    max_len = st.number_input("Maximum CDR3 length", min_value=5, max_value=150, value=60, step=1)
    require_strong_c_anchor = st.checkbox(
        "Require YYC/YxC-like upstream Cys anchor",
        value=False,
        help="Turn this on to reduce false positives, but it may miss unusual or truncated constructs.",
    )

    st.header("FASTA output")
    header_mode = st.radio(
        "Output FASTA header",
        options=["Full original header", "Sample ID only"],
        index=0,
    )
    keep_failed_records = st.checkbox(
        "Keep failed records as blank FASTA entries",
        value=True,
        help="Keeps one output FASTA record per input record. Disable this to output only successful CDR3 calls.",
    )
    add_status_to_failed_headers = st.checkbox(
        "Add failure reason to failed FASTA headers",
        value=False,
        help="Leave off if you need exactly unchanged FASTA headers/sample IDs.",
    )

uploaded_files = st.file_uploader(
    "Upload FASTA file(s)",
    type=["fasta", "fa", "faa", "fna", "txt"],
    accept_multiple_files=True,
)

example = ">clone_001\nQVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGTTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAREGPYYYGMDYWGQGTQVTVSS\n>clone_002\nQVQLQESGGGLVQAGGSLRLSCAASGFTFSSYWMGWFRQAPGKEREGVAAISSGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAKDRSTYDYWGQGTQVTVSS\n"

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
    st.info("Upload FASTA files or enable the built-in example to begin.")
    st.stop()

if min_len > max_len:
    st.error("Minimum CDR3 length cannot be greater than maximum CDR3 length.")
    st.stop()

results = [
    extract_record(
        record=r,
        input_mode=input_mode,
        min_len=int(min_len),
        max_len=int(max_len),
        require_strong_c_anchor=require_strong_c_anchor,
    )
    for r in records
]

df = results_to_dataframe(results)

n_total = len(df)
n_ok = int((df["status"] == "ok").sum())
n_failed = n_total - n_ok
high_or_medium = int(df["confidence"].isin(["high", "medium"]).sum())

metric_cols = st.columns(4)
metric_cols[0].metric("Input records", n_total)
metric_cols[1].metric("CDR3 found", n_ok)
metric_cols[2].metric("Not found", n_failed)
metric_cols[3].metric("High/medium confidence", high_or_medium)

if n_ok > 0:
    length_df = df.loc[df["status"] == "ok", ["cdr3_length"]].copy()
    st.subheader("CDR3 length distribution")
    st.bar_chart(length_df["cdr3_length"].value_counts().sort_index())

st.subheader("Results")
visible_columns = [
    "order",
    "source_file",
    "sample_id",
    "status",
    "confidence",
    "cdr3",
    "cdr3_length",
    "c_anchor_1based",
    "j_anchor_1based",
    "j_motif",
    "frame",
    "strand",
    "score",
    "reason",
]
st.dataframe(df[visible_columns], use_container_width=True, hide_index=True)

cdr3_fasta = make_output_fasta(
    results,
    keep_failed_records=keep_failed_records,
    header_mode=header_mode,
    add_status_to_failed_headers=add_status_to_failed_headers,
)

csv_bytes = df.to_csv(index=False).encode("utf-8")

st.subheader("Download")
download_cols = st.columns(2)
with download_cols[0]:
    st.download_button(
        "Download CDR3 FASTA",
        data=cdr3_fasta.encode("utf-8"),
        file_name="nanobody_cdr3_extracted.fasta",
        mime="text/plain",
        use_container_width=True,
    )
with download_cols[1]:
    st.download_button(
        "Download QC CSV",
        data=csv_bytes,
        file_name="nanobody_cdr3_qc.csv",
        mime="text/csv",
        use_container_width=True,
    )

with st.expander("Preview output FASTA"):
    st.code(cdr3_fasta[:10000], language="text")

with st.expander("Method and limitations"):
    st.markdown(
        """
        **Boundary logic used here**

        - The app searches for a plausible downstream J-region motif such as `WGQG`, `FGQG`, or broader `F/W-G-x-G`.
        - It then searches upstream for a conserved Cys anchor, preferably in a `YYC`, aromatic-aromatic-C, or `Y/F/W-x-C` context.
        - The extracted CDR3 is the amino-acid sequence **after** the conserved Cys and **before** the F/W anchor.

        **When to be careful**

        - Very short/truncated reads, fusion constructs, sequences with sequencing errors, or unusual engineered frameworks may fail.
        - CDR3s containing extra cysteines are supported, but ambiguous motif contexts can still cause false calls.
        - For final annotation, compare against an IMGT-numbering workflow such as ANARCI or IgBLAST.
        """
    )

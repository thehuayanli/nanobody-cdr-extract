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
    from abnumber import Chain  # optional
    ABNUMBER_AVAILABLE = True
except Exception:
    Chain = None
    ABNUMBER_AVAILABLE = False


# =========================================================
# App configuration
# =========================================================

st.set_page_config(
    page_title="Antibody / Nanobody Region Extractor",
    page_icon="🧬",
    layout="wide",
)

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
# FASTA parsing / cleaning
# =========================================================

def parse_fasta(text: str, source_file: str, start_order: int = 1) -> list[FastaRecord]:
    """Parse FASTA while preserving the original full header and input order."""
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
                # Graceful fallback for pasted/non-FASTA text.
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
    """Clean sequence for display/auto-detection. Can keep gaps for QC."""
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[\r\n0-9_.]", "", seq)

    allowed = set(VALID_AA_LETTERS)
    if not keep_stop:
        allowed.discard("*")
    if keep_gap:
        allowed.add("-")

    # Keep nucleotide ambiguity letters so auto-detect still works before translation.
    allowed = allowed | NUCLEOTIDE_LETTERS
    return "".join(ch for ch in seq if ch in allowed)


def clean_aa_for_segmentation(seq: str) -> str:
    """Remove gaps/stops/nonletters before numbering or motif segmentation."""
    seq = clean_sequence(seq, keep_stop=False, keep_gap=False)
    seq = seq.replace("-", "")
    seq = re.sub(r"[^A-Z]", "", seq)
    return seq


def wrap_fasta_sequence(seq: str, width: int = 80) -> str:
    if not seq:
        return ""
    return "\n".join(textwrap.wrap(seq, width=width))


# =========================================================
# Optional nucleotide translation
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
        "TGCAANRYSWMKVHDBtgcaANRYSWMKVHDB",
    )
    # The mapping above covers common IUPAC ambiguity codes sufficiently for frame testing.
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
    """Return candidate amino acid sequences as (aa_seq, frame_label, strand_label)."""
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
# Region extraction: optional abnumber
# =========================================================

def get_regions_with_abnumber(aa_seq: str, scheme: str = "imgt") -> tuple[dict[str, str], str]:
    """Use abnumber/ANARCI-style numbering if available."""
    if not ABNUMBER_AVAILABLE or Chain is None:
        raise RuntimeError("abnumber is not installed or failed to import.")

    aa_seq = clean_aa_for_segmentation(aa_seq)
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


# =========================================================
# Region extraction: robust motif fallback
# =========================================================

def infer_chain_type_motif(seq: str) -> str:
    """Classify as heavy/VHH-like, light-like, or unknown using broad motifs."""
    s = clean_aa_for_segmentation(seq)

    if (
        re.search(r"(DIQMT|QSVLT|EIVLT|DIVMT)", s[:25])
        or "WYQQ" in s[:75]
        or re.search(r"F(GQG|GGG|G.G)T?K[LV]", s[-35:])
    ):
        return "VL-like"

    if (
        re.search(r"(QVQL|EVQL|DVQL|QLQL)", s[:25])
        or re.search(r"(WGQG|WGRG|FGQG)", s[-35:])
    ):
        return "VH/VHH-like"

    return "unknown"


def find_j_anchor(seq: str) -> Optional[tuple[float, int, int, str]]:
    """
    Find the downstream J-region F/W-G-x-G anchor.
    Returns score, start, end, motif.
    """
    candidates: list[tuple[float, int, int, str]] = []

    for pattern in [r"[WF]GQG", r"[WF]GRG", r"[WF]GGG", r"[WF]G[A-Z]G"]:
        for match in re.finditer(pattern, seq):
            if match.start() < max(40, int(len(seq) * 0.45)):
                continue

            motif = match.group(0)
            score = {
                "WGQG": 30,
                "FGQG": 30,
                "WGRG": 25,
                "FGGG": 24,
            }.get(motif, 12)

            # Prefer anchors closer to the C-terminal side.
            score += (match.start() / max(1, len(seq))) * 10
            candidates.append((score, match.start(), match.end(), motif))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0]


def find_cdr3_c_anchor(seq: str, j_start: int, chain_type: str) -> Optional[tuple[float, int, str, int]]:
    """
    Find the conserved Cys immediately upstream of CDR3.
    Returns score, Cys position, local context, CDR3 length.
    """
    candidates: list[tuple[float, int, str, int]] = []

    for match in re.finditer("C", seq[:j_start]):
        c_pos = match.start()
        cdr3_len = j_start - c_pos - 1

        if not (3 <= cdr3_len <= 80):
            continue

        context = seq[max(0, c_pos - 3):c_pos + 1]
        score = 0.0

        if re.search(r"[YFHW][YFHW]C$", context):
            score += 35
        elif re.search(r"[YFHW].C$", context):
            score += 25
        else:
            score += 5

        if chain_type == "VL-like":
            if 5 <= cdr3_len <= 15:
                score += 20
            elif 3 <= cdr3_len <= 30:
                score += 10

            if 75 <= c_pos <= 115:
                score += 12
        else:
            if 5 <= cdr3_len <= 35:
                score += 20
            elif 3 <= cdr3_len <= 60:
                score += 10

            if 75 <= c_pos <= 120:
                score += 12

        candidates.append((score, c_pos, context, cdr3_len))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0]


def find_cdr1_boundaries(seq: str) -> Optional[tuple[float, int, int, str]]:
    """
    Find first framework Cys and the following W anchor.
    CDR1 is between them.
    """
    candidates: list[tuple[float, int, int, str]] = []

    for c_match in re.finditer("C", seq[:45]):
        c_pos = c_match.start()

        for w_match in re.finditer("W", seq):
            w_pos = w_match.start()
            if w_pos <= c_pos + 2 or w_pos > c_pos + 35:
                continue

            cdr1_len = w_pos - c_pos - 1
            if not (3 <= cdr1_len <= 25):
                continue

            after_w = seq[w_pos:w_pos + 4]
            score = 0.0

            if re.match(r"W[FIYV]RQ", after_w):
                score += 20
            if re.match(r"WYQQ", after_w):
                score += 20
            if 5 <= cdr1_len <= 17:
                score += 10

            candidates.append((score, c_pos, w_pos, after_w))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return candidates[0]


def find_cdr2_heavy_boundaries(seq: str, w1_pos: int, cdr3_c_pos: int) -> tuple[int, int, str]:
    """
    Heavy/VHH-like CDR2 fallback.
    Finds a conserved FR3 start motif such as KGRFTIS / KFKGRVTL,
    then takes the preceding hypervariable block as CDR2.
    """
    sub = seq[w1_pos:cdr3_c_pos]
    patterns = [
        r"KFKGRVTL",
        r"KGRFTIS",
        r"KGRVTIS",
        r"KGRVTL",
        r"KGRF[TA]IS",
        r"KFKGR",
        r"KGR",
        r"RFTIS",
        r"RVTIS",
        r"RVTL",
        r"FTIS",
        r"VTIS",
        r"VTLT",
    ]

    candidates: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sub):
            abs_pos = w1_pos + match.start()
            if abs_pos > w1_pos + 20 and abs_pos < cdr3_c_pos - 10:
                candidates.append((len(match.group(0)), abs_pos, match.group(0)))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        _, fr3_start, motif = candidates[0]

        # Heavy/VHH CDR2 is commonly about 8-17 aa in this simplified segmentation.
        cdr2_start = max(w1_pos + 15, fr3_start - 17)
        return cdr2_start, fr3_start, motif

    # Conservative fallback if FR3 motif is not found.
    cdr2_start = min(cdr3_c_pos, w1_pos + 25)
    cdr2_end = min(cdr3_c_pos, cdr2_start + 15)
    return cdr2_start, cdr2_end, "approx"


def find_cdr2_light_boundaries(seq: str, w1_pos: int, cdr3_c_pos: int) -> tuple[int, int, str]:
    """
    VL-like CDR2 fallback.
    Finds the FR3 start motif after light-chain CDR2.
    """
    sub = seq[w1_pos:cdr3_c_pos]
    patterns = [
        r"GIP[AD]RFSG",
        r"GVPDRFSG",
        r"GIPDRFSG",
        r"GIPARFSG",
        r"FSGSG",
        r"FSGSK",
    ]

    candidates: list[tuple[int, int, str]] = []
    for pattern in patterns:
        for match in re.finditer(pattern, sub):
            abs_pos = w1_pos + match.start()
            if abs_pos > w1_pos + 15 and abs_pos < cdr3_c_pos - 5:
                candidates.append((len(match.group(0)), abs_pos, match.group(0)))

    if candidates:
        candidates.sort(key=lambda x: (-x[0], x[1]))
        _, fr3_start, motif = candidates[0]

        # The synthetic/light-chain test CDR2s are 7 aa; this is also a common simplified length.
        cdr2_start = max(w1_pos + 12, fr3_start - 7)
        return cdr2_start, fr3_start, motif

    cdr2_start = min(cdr3_c_pos, w1_pos + 18)
    cdr2_end = min(cdr3_c_pos, cdr2_start + 7)
    return cdr2_start, cdr2_end, "approx"


def get_regions_with_motif_fallback(aa_seq: str) -> tuple[dict[str, str], str, str]:
    """
    Dependency-free VH/VHH/VL region segmentation.

    This is not a formal IMGT numbering engine, but it is robust for:
    - normal VHH/VH synthetic test sequences,
    - VL edge cases,
    - sequences containing '-' gaps,
    - non-typical CDR3 upstream contexts such as YHC/FYC,
    - non-typical downstream J anchors such as FGQG/WGRG.
    """
    seq = clean_aa_for_segmentation(aa_seq)
    regions = {region: "" for region in REGION_ORDER}

    if len(seq) < 45:
        return regions, "unknown", "too short after cleaning"

    chain_type = infer_chain_type_motif(seq)

    j_anchor = find_j_anchor(seq)
    if j_anchor is None:
        return regions, chain_type, "no downstream F/W-G-x-G J anchor found"

    _, j_start, _, j_motif = j_anchor

    cdr3_anchor = find_cdr3_c_anchor(seq, j_start, chain_type)
    if cdr3_anchor is None:
        return regions, chain_type, f"J anchor found ({j_motif}) but no plausible upstream Cys CDR3 anchor"

    _, cdr3_c_pos, cdr3_context, _ = cdr3_anchor

    regions["CDR3"] = seq[cdr3_c_pos + 1:j_start]
    regions["FR4"] = seq[j_start:]

    cdr1 = find_cdr1_boundaries(seq)
    if cdr1 is not None:
        _, cdr1_c_pos, w1_pos, w1_motif = cdr1
        regions["FR1"] = seq[:cdr1_c_pos + 1]
        regions["CDR1"] = seq[cdr1_c_pos + 1:w1_pos]
    else:
        # Last-resort split if the first W anchor cannot be found cleanly.
        w1_pos = seq.find("W", 20, 65)
        cdr1_c_pos = seq.find("C", 0, w1_pos) if w1_pos != -1 else -1

        if cdr1_c_pos != -1 and w1_pos != -1:
            regions["FR1"] = seq[:cdr1_c_pos + 1]
            regions["CDR1"] = seq[cdr1_c_pos + 1:w1_pos]
            w1_motif = "approx"
        else:
            regions["FR1"] = seq[:min(25, len(seq))]
            w1_pos = min(35, len(seq))
            w1_motif = "approx"

    if chain_type == "VL-like":
        cdr2_start, cdr2_end, cdr2_motif = find_cdr2_light_boundaries(seq, w1_pos, cdr3_c_pos)
    else:
        cdr2_start, cdr2_end, cdr2_motif = find_cdr2_heavy_boundaries(seq, w1_pos, cdr3_c_pos)

    cdr2_start = max(w1_pos, min(cdr2_start, cdr3_c_pos))
    cdr2_end = max(cdr2_start, min(cdr2_end, cdr3_c_pos))

    regions["FR2"] = seq[w1_pos:cdr2_start]
    regions["CDR2"] = seq[cdr2_start:cdr2_end]
    regions["FR3"] = seq[cdr2_end:cdr3_c_pos + 1]

    reason = f"motif fallback: W1={w1_motif}; CDR2_motif={cdr2_motif}; CDR3_C_context={cdr3_context}; J={j_motif}"
    return regions, chain_type, reason


# =========================================================
# Region extraction orchestration
# =========================================================

def region_completeness_score(region_map: dict[str, str]) -> int:
    score = 0
    for region in REGION_ORDER:
        seq = region_map.get(region, "") or ""
        if seq:
            score += 100
            score += min(len(seq), 30)
    # Weight CDR3 because it is especially important for your workflow.
    if region_map.get("CDR3"):
        score += 150
    if region_map.get("CDR2"):
        score += 40
    return score


def get_regions(aa_seq: str, scheme: str, method: str) -> tuple[dict[str, str], str, str, str]:
    """
    Return region_map, chain_type, method_used, note.
    """
    if method == "Motif fallback only":
        region_map, chain_type, note = get_regions_with_motif_fallback(aa_seq)
        return region_map, chain_type, "motif_fallback", note

    if method == "abnumber only":
        region_map, chain_type = get_regions_with_abnumber(aa_seq, scheme=scheme)
        return region_map, chain_type, "abnumber", "abnumber numbering"

    # Auto mode: use abnumber when available, but do not let it block useful fallback calls.
    attempts: list[tuple[dict[str, str], str, str, str, int]] = []
    errors: list[str] = []

    try:
        ab_regions, ab_chain_type = get_regions_with_abnumber(aa_seq, scheme=scheme)
        attempts.append(
            (
                ab_regions,
                ab_chain_type,
                "abnumber",
                "abnumber numbering",
                region_completeness_score(ab_regions),
            )
        )
    except Exception as e:
        errors.append(f"abnumber failed/unavailable: {e}")

    try:
        fb_regions, fb_chain_type, fb_note = get_regions_with_motif_fallback(aa_seq)
        attempts.append(
            (
                fb_regions,
                fb_chain_type,
                "motif_fallback",
                fb_note,
                region_completeness_score(fb_regions),
            )
        )
    except Exception as e:
        errors.append(f"motif fallback failed: {e}")

    if attempts:
        attempts.sort(key=lambda x: x[4], reverse=True)
        region_map, chain_type, method_used, note, _ = attempts[0]
        if method_used == "motif_fallback" and errors:
            note = note + " | " + " | ".join(errors)
        return region_map, chain_type, method_used, note

    return {region: "" for region in REGION_ORDER}, "unknown", "failed", " | ".join(errors) or "no extraction attempts succeeded"


def get_best_candidate(raw_seq: str, input_mode: str, scheme: str, method: str):
    """
    For AA input: use directly.
    For nucleotide input: try translated frames and keep the most complete segmentation.
    """
    candidates = candidate_aa_sequences(raw_seq, input_mode)
    best = None
    last_note = ""

    for aa_seq, frame_label, strand_label in candidates:
        aa_clean = clean_aa_for_segmentation(aa_seq)
        if len(aa_clean) < 45:
            continue

        region_map, chain_type, method_used, note = get_regions(aa_clean, scheme=scheme, method=method)
        score = region_completeness_score(region_map)

        # Penalize candidates with no CDR3 heavily.
        if not region_map.get("CDR3"):
            score -= 200

        if best is None or score > best["score"]:
            best = {
                "aa_sequence_used": aa_clean,
                "frame": frame_label,
                "strand": strand_label,
                "chain_type": chain_type,
                "region_map": region_map,
                "method_used": method_used,
                "score": score,
                "note": note,
            }

        last_note = note

    return best, last_note


def build_result_rows(records: list[FastaRecord], input_mode: str, scheme: str, method: str) -> pd.DataFrame:
    rows = []

    for record in records:
        best, last_note = get_best_candidate(
            record.sequence,
            input_mode=input_mode,
            scheme=scheme,
            method=method,
        )

        status = "ok" if best and region_completeness_score(best["region_map"]) > 0 else "not_found"

        row = {
            "order": record.order,
            "source_file": record.source_file,
            "sample_id": record.sample_id,
            "status": status,
            "chain_type": best["chain_type"] if best else "unknown",
            "method_used": best["method_used"] if best else "",
            "frame": best["frame"] if best else "",
            "strand": best["strand"] if best else "",
            "score": best["score"] if best else 0,
            "note": best["note"] if best else last_note,
            "header": record.header,
            "input_length": len(clean_sequence(record.sequence)),
            "cleaned_aa_length": len(best["aa_sequence_used"]) if best else 0,
        }

        for region in REGION_ORDER:
            value = best["region_map"].get(region, "") if best else ""
            row[region] = value
            row[f"{region}_len"] = len(value)

        rows.append(row)

    return pd.DataFrame(rows)


# =========================================================
# Output helpers
# =========================================================

def make_region_fasta(
    df: pd.DataFrame,
    region_name: str,
    header_mode: str = "Full original header",
    keep_failed_records: bool = True,
) -> str:
    lines: list[str] = []

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
    lines: list[str] = []

    for _, row in df.sort_values("order").iterrows():
        if row["status"] != "ok" and not keep_failed_records:
            continue

        header = row["sample_id"] if header_mode == "Sample ID only" else row["header"]
        seq = separator.join((row.get(region, "") or "") for region in selected_regions)
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
# Region selector UI
# =========================================================

def reset_to_only(region_list: list[str]):
    for region in REGION_ORDER:
        st.session_state[f"pick_{region}"] = region in region_list


def toggle_region(region: str):
    st.session_state[f"pick_{region}"] = not st.session_state.get(f"pick_{region}", False)


def render_region_selector():
    """
    Native Streamlit button-based linear VH/VHH map.
    This replaces the old raw-HTML map that appeared as a large gray code block.
    """
    st.caption("Linear VH/VHH map: FR1 → CDR1 → FR2 → CDR2 → FR3 → CDR3 → FR4")

    button_cols = st.columns([REGION_WIDTHS[region] for region in REGION_ORDER])
    for col, region in zip(button_cols, REGION_ORDER):
        selected = st.session_state.get(f"pick_{region}", False)
        label = f"✅ {region}" if selected else region

        with col:
            st.button(
                label,
                key=f"btn_{region}",
                on_click=toggle_region,
                args=(region,),
                use_container_width=True,
            )


# =========================================================
# Streamlit app
# =========================================================

st.title("Antibody / Nanobody Region Extractor")
st.caption(
    "Upload FASTA files, select VH/VHH regions, and download one FASTA per selected region. "
    "Input order and sample IDs are preserved."
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
        "Segmentation method",
        options=[
            "Auto: abnumber if available + motif fallback",
            "Motif fallback only",
            "abnumber only",
        ],
        index=0,
        help=(
            "Auto is recommended. It uses abnumber/IMGT if installed, but falls back to a robust motif parser. "
            "Motif fallback only is dependency-free and should work on Streamlit Cloud without installing abnumber."
        ),
    )

    scheme = st.selectbox(
        "Numbering scheme for abnumber",
        options=["imgt"],
        index=0,
    )

    st.header("Output settings")

    header_mode = st.radio(
        "Output FASTA header",
        options=["Full original header", "Sample ID only"],
        index=0,
    )

    keep_failed_records = st.checkbox(
        "Keep failed records as blank FASTA entries",
        value=True,
        help="Keeps the same number/order of records in every output FASTA.",
    )

    create_combined = st.checkbox(
        "Also create combined FASTA of all selected regions",
        value=True,
        help="If multiple regions are selected, also export one FASTA concatenating them in VH/VHH order.",
    )

    separator_choice = st.selectbox(
        "Combined-region separator",
        options=["none", "X", "GGGS"],
        index=0,
    )
    separator = "" if separator_choice == "none" else separator_choice

uploaded_files = st.file_uploader(
    "Upload FASTA file(s)",
    type=["fasta", "fa", "faa", "fna", "txt"],
    accept_multiple_files=True,
)

example_fasta = """>example_vhh_01
QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGTTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCAREGPYYYGMDYWGQGTQVTVSS
>example_vh_01
EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARDLGGYYFDYWGQGTLVTVSS
>example_vl_01
DIQMTQSPSSLSASVGDRVTITCRASQSVSSYLAWYQQKPGKAPKLLIYDASNRATGIPARFSGSGSGTDFTLTISSLQPEDFATYYCLQHNSYPYTFGQGTKLEIK
>example_vhh_with_gaps
QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGTTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYC---AREGPYYYGMDY---WGQGTQVTVSS
>example_non_typical_j_anchor
QVQLVESGGGLVQAGGSLRLSCAASGRTFSSYAMGWFRQAPGKEREFVAAVSRGGSTYYADSVKGRFTISRDNAKNTVYLQMNSLKPEDTAVYYCARGTYYDSSGYWGRGTQVTVSS
"""

use_example = st.checkbox("Use built-in example instead of uploaded files", value=False)

records: list[FastaRecord] = []

if use_example:
    records = parse_fasta(example_fasta, source_file="example.fasta", start_order=1)
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

st.subheader("1) Select region(s) to extract")

# Initialize default once.
if not any(f"pick_{region}" in st.session_state for region in REGION_ORDER):
    reset_to_only(["CDR3"])

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

render_region_selector()

selected_regions = [region for region in REGION_ORDER if st.session_state.get(f"pick_{region}", False)]

if not selected_regions:
    st.warning("Please select at least one region.")
    st.stop()

st.write("**Selected regions:**", " + ".join(selected_regions))


# ---------------------------------------------------------
# Process
# ---------------------------------------------------------

df = build_result_rows(records, input_mode=input_mode, scheme=scheme, method=method)

n_total = len(df)
n_ok = int((df["status"] == "ok").sum())
n_failed = n_total - n_ok
n_fallback = int((df["method_used"] == "motif_fallback").sum())
n_abnumber = int((df["method_used"] == "abnumber").sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Input records", n_total)
metric_cols[1].metric("Segmented", n_ok)
metric_cols[2].metric("Failed", n_failed)
metric_cols[3].metric("abnumber calls", n_abnumber)
metric_cols[4].metric("Fallback calls", n_fallback)

st.subheader("2) Results")

display_cols = [
    "order",
    "source_file",
    "sample_id",
    "status",
    "chain_type",
    "method_used",
    "frame",
    "strand",
    "score",
    "note",
] + selected_regions + [f"{region}_len" for region in selected_regions]

st.dataframe(df[display_cols], use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# Downloads
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

st.markdown("### Preview")

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
    if ABNUMBER_AVAILABLE:
        st.markdown("- `abnumber` is available in this environment.")
    else:
        st.markdown("- `abnumber` is not installed; the app is using the dependency-free motif fallback.")

    st.markdown(
        """
        - The old raw-HTML region bar has been removed, so the large gray code block should no longer appear.
        - The motif fallback was updated to better handle short CDR3s, VL edge cases, gaps (`-`), YHC/FYC-like non-typical upstream CDR3 anchors, and FGQG/WGRG-like downstream J anchors.
        - For formal annotation, use an antibody numbering method such as IMGT/ANARCI/abnumber when available.
        - The output ZIP contains one FASTA per selected region plus `region_extraction_qc.csv`.
        """
    )

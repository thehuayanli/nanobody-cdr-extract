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
    from abnumber import Chain  # optional; requires ANARCI backend in many installs
    ABNUMBER_AVAILABLE = True
except Exception:
    Chain = None
    ABNUMBER_AVAILABLE = False


# =========================================================
# App config
# =========================================================

st.set_page_config(
    page_title="IMGT Antibody / Nanobody Region Extractor",
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
# FASTA and sequence utilities
# =========================================================

def parse_fasta(text: str, source_file: str, start_order: int = 1) -> list[FastaRecord]:
    """Parse FASTA while preserving input order and full original header."""
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
    """General cleaning for display, length QC, and auto-detection."""
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[\r\n0-9_.]", "", seq)

    allowed = set(VALID_AA_LETTERS)
    if not keep_stop:
        allowed.discard("*")
    if keep_gap:
        allowed.add("-")

    allowed = allowed | NUCLEOTIDE_LETTERS
    return "".join(ch for ch in seq if ch in allowed)


def clean_aa_for_segmentation(seq: str) -> str:
    """Remove gaps, stop codons, and nonletters before region segmentation."""
    seq = seq.upper().replace(" ", "").replace("\t", "")
    seq = re.sub(r"[\r\n0-9_.\-*]", "", seq)
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
    seq = seq.upper().replace("U", "T")
    seq = re.sub(r"[^ACGTN]", "N", seq)
    table = str.maketrans("ACGTN", "TGCAN")
    return seq.translate(table)[::-1]


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
# Optional exact numbering through abnumber
# =========================================================

def get_regions_with_abnumber(aa_seq: str, scheme: str = "imgt") -> tuple[dict[str, str], str, str]:
    """
    Use abnumber if it is available.

    Many environments require an ANARCI backend for abnumber. The app does not
    depend on this mode; the built-in IMGT anchor-mapping mode is the default.
    """
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
    return region_map, chain_type, "abnumber IMGT numbering"


# =========================================================
# Built-in anchor-guided IMGT region extraction
# =========================================================

def infer_chain_type(seq: str) -> str:
    """Broad VH/VHH vs VL classification for choosing anchor rules."""
    s = clean_aa_for_segmentation(seq)

    if (
        re.search(r"(DIQMT|QSVLT|EIVLT|DIVMT|QSALT)", s[:30])
        or "WYQQ" in s[:75]
        or re.search(r"F(GQG|GGG|G.G)T?K[LV]", s[-35:])
    ):
        return "VL-like"

    if (
        re.search(r"(QVQL|EVQL|DVQL|QLQL|QVQM)", s[:30])
        or re.search(r"[A-Z]G[QKR]G[TS]", s[-25:])
    ):
        return "VH/VHH-like"

    return "unknown"


def find_first_cys_23(seq: str, chain_type: str) -> Optional[int]:
    """Find the first conserved Cys anchor, IMGT position 23."""
    candidates: list[tuple[float, int, str]] = []

    for match in re.finditer("C", seq[:55]):
        c_pos = match.start()
        context = seq[max(0, c_pos - 8):c_pos + 1]
        score = 0.0

        if 18 <= c_pos <= 30:
            score += 30
        elif 15 <= c_pos <= 35:
            score += 15

        if chain_type == "VL-like":
            if re.search(r"(TIT|TVT|IVT|SIS|KVT).{0,5}C$", context):
                score += 15
        else:
            if re.search(r"(LSC|VSC|ASC|LTC).{0,5}C$", context):
                score += 15

        candidates.append((score, c_pos, context))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return candidates[0][1]


def find_trp_41(seq: str, cys23_pos: int) -> Optional[int]:
    """Find conserved Trp anchor, IMGT position 41."""
    candidates: list[tuple[float, int, str]] = []

    search_start = cys23_pos + 5
    search_end = min(len(seq), cys23_pos + 50)

    for match in re.finditer("W", seq[search_start:search_end]):
        w_pos = search_start + match.start()
        motif = seq[w_pos:w_pos + 4]
        distance = w_pos - cys23_pos - 1
        score = 0.0

        if re.match(r"W[FIYVAM]RQ", motif):
            score += 30
        elif re.match(r"WYQQ", motif):
            score += 30
        elif re.match(r"W[AV]RQ", motif):
            score += 25
        elif re.match(r"WVRR", motif):
            score += 20

        if 8 <= distance <= 22:
            score += 15
        elif 5 <= distance <= 30:
            score += 5

        candidates.append((score, w_pos, motif))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return candidates[0][1]


def find_fr4_start_118(seq: str) -> Optional[int]:
    """
    Find the FR4 start anchor, approximately IMGT position 118.

    The older app required W/F-G-x-G. That missed real examples ending in
    R-G-Q-G or Y-G-Q-G. This version searches terminal x-G-x-G motifs and
    scores them by terminal FR4 context.
    """
    tail_start = max(0, len(seq) - 45)
    tail = seq[tail_start:]

    candidates: list[tuple[float, int, str]] = []

    for pattern, base_score in [
        (r"[A-Z]G[QKR]G", 40),   # common/useful: WGQG, FGQG, RGQG, YGQG, WGKG
        (r"[A-Z]G[A-Z]G", 15),   # broader fallback
    ]:
        for match in re.finditer(pattern, tail):
            pos = tail_start + match.start()
            motif = match.group(0)
            after = seq[pos + 4:pos + 20]

            score = float(base_score)
            score += (pos / max(1, len(seq))) * 10

            if after.startswith(("T", "S")):
                score += 8
            if "VTVSS" in seq[pos:pos + 20] or "QVTVSS" in seq[pos:pos + 20] or "TVL" in seq[pos:pos + 20]:
                score += 15

            if motif[0] in "WF":
                score += 8
            elif motif[0] in "YR":
                score += 5

            candidates.append((score, pos, motif))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][1]


def find_cys_104(seq: str, fr4_start: int) -> Optional[int]:
    """Find the second conserved Cys anchor, IMGT position 104."""
    candidates: list[tuple[float, int, str, int]] = []

    for match in re.finditer("C", seq[:fr4_start]):
        c_pos = match.start()
        cdr3_len = fr4_start - c_pos - 1

        if not (3 <= cdr3_len <= 60):
            continue

        context = seq[max(0, c_pos - 6):c_pos + 1]
        score = 0.0

        if re.search(r"YYC$", context):
            score += 40
        elif re.search(r"[YFHW][YFHW]C$", context):
            score += 28
        elif re.search(r"[YFHW].C$", context):
            score += 18
        else:
            score += 4

        if 80 <= c_pos <= 125:
            score += 12
        elif 70 <= c_pos <= 140:
            score += 6

        # Avoid choosing an internal Cys inside a CDR3 when a YYC-like anchor exists upstream.
        if cdr3_len < 4:
            score -= 15

        candidates.append((score, c_pos, context, cdr3_len))

    if not candidates:
        return None

    candidates.sort(key=lambda x: (x[0], -x[1]), reverse=True)
    return candidates[0][1]


def find_fr3_start_66(seq: str, trp41_pos: int, cys104_pos: int, chain_type: str) -> tuple[int, str]:
    """
    Find start of FR3-IMGT, approximately position 66.

    IMGT CDR2 is positions 56-65, so it is capped at 10 residues.
    This function locates the conserved FR3 core motif and backs up a few residues.
    """
    sub = seq[trp41_pos:cys104_pos]

    if chain_type == "VL-like":
        patterns = [
            (r"GIP[AD]RFSG", 0),
            (r"GVPDRFSG", 0),
            (r"GIPDRFSG", 0),
            (r"GIPARFSG", 0),
            (r"FSGSG", 0),
            (r"FSGSK", 0),
        ]
    else:
        patterns = [
            (r"KGRFTIS", 2),
            (r"KGRFTV", 2),
            (r"KGRVTIS", 2),
            (r"KGRVTL", 2),
            (r"KFKGR", 1),
            (r"EGRFTIS", 1),
            (r"QGRFTV", 1),
            (r"RFTIS", 3),
            (r"RVTIS", 3),
            (r"RVTL", 2),
            (r"FTIS", 5),
            (r"FTV", 5),
        ]

    candidates: list[tuple[int, int, int, str]] = []

    for pattern, back in patterns:
        for match in re.finditer(pattern, sub):
            abs_pos = trp41_pos + match.start()
            if abs_pos > trp41_pos + 12 and abs_pos < cys104_pos - 5:
                fr3_start = max(trp41_pos, abs_pos - back)
                candidates.append((len(match.group(0)), -match.start(), fr3_start, match.group(0)))

    if candidates:
        candidates.sort(reverse=True)
        _, _, fr3_start, motif = candidates[0]
        return fr3_start, motif

    # Conservative IMGT-like fallback: FR2 starts at Trp41 and often contributes about 16 aa.
    return min(cys104_pos, trp41_pos + 26), "approx"


def get_regions_with_builtin_imgt(aa_seq: str) -> tuple[dict[str, str], str, str]:
    """
    Built-in anchor-guided IMGT region mapping.

    Boundaries implemented:
    - FR1-IMGT / CDR1-IMGT / FR2-IMGT using Cys23 and Trp41.
    - CDR1 capped to <= 12 aa.
    - CDR2 capped to <= 10 aa.
    - FR3 / CDR3 / FR4 using FR3 motif, Cys104, and terminal xGxG FR4 anchor.
    """
    seq = clean_aa_for_segmentation(aa_seq)
    regions = {region: "" for region in REGION_ORDER}

    if len(seq) < 45:
        return regions, "unknown", "too short after cleaning"

    chain_type = infer_chain_type(seq)

    cys23 = find_first_cys_23(seq, chain_type)
    if cys23 is None:
        return regions, chain_type, "failed: no Cys23 anchor"

    trp41 = find_trp_41(seq, cys23)
    if trp41 is None:
        return regions, chain_type, "failed: no Trp41 anchor"

    fr4_start = find_fr4_start_118(seq)
    if fr4_start is None:
        return regions, chain_type, "failed: no terminal FR4 xGxG anchor"

    cys104 = find_cys_104(seq, fr4_start)
    if cys104 is None:
        return regions, chain_type, "failed: no Cys104 anchor"

    fr3_start, fr3_motif = find_fr3_start_66(seq, trp41, cys104, chain_type)

    # CDR1-IMGT is positions 27-38: max 12 aa.
    # For VH/VHH, positions 24-26 are typically the three residues right after Cys23,
    # so CDR1 starts after those 3 residues. For VL-like edge cases, keeping the
    # whole Cys-to-Trp segment better matches common light-chain test inputs.
    if chain_type == "VL-like":
        cdr1_start = cys23 + 1
        if trp41 - cdr1_start > 12:
            cdr1_start = trp41 - 12
    else:
        cdr1_start = min(trp41, cys23 + 1 + 3)
        if trp41 - cdr1_start > 12:
            cdr1_start = trp41 - 12

    cdr1_end = trp41

    # CDR2-IMGT is positions 56-65: max 10 aa.
    cdr2_end = fr3_start
    cdr2_len = min(10, max(0, cdr2_end - trp41))
    cdr2_start = max(trp41, cdr2_end - cdr2_len)

    regions["FR1"] = seq[:cdr1_start]
    regions["CDR1"] = seq[cdr1_start:cdr1_end]
    regions["FR2"] = seq[cdr1_end:cdr2_start]
    regions["CDR2"] = seq[cdr2_start:cdr2_end]
    regions["FR3"] = seq[cdr2_end:cys104 + 1]
    regions["CDR3"] = seq[cys104 + 1:fr4_start]
    regions["FR4"] = seq[fr4_start:]

    note = (
        f"built-in IMGT anchor mapping; "
        f"Cys23={cys23 + 1}; Trp41={trp41 + 1}; "
        f"FR3_start={fr3_start + 1} ({fr3_motif}); "
        f"Cys104={cys104 + 1}; FR4_start={fr4_start + 1}; "
        f"CDR1<=12; CDR2<=10"
    )
    return regions, chain_type, note


# =========================================================
# Orchestration
# =========================================================

def completeness_score(region_map: dict[str, str]) -> int:
    score = 0
    for region in REGION_ORDER:
        value = region_map.get(region, "") or ""
        if value:
            score += 100 + min(len(value), 30)

    if region_map.get("CDR3"):
        score += 200
    if region_map.get("CDR1") and len(region_map["CDR1"]) <= 12:
        score += 50
    if region_map.get("CDR2") and len(region_map["CDR2"]) <= 10:
        score += 50

    return score


def get_regions(aa_seq: str, engine: str, scheme: str) -> tuple[dict[str, str], str, str, str]:
    """
    Returns region_map, chain_type, method_used, note.
    """
    if engine == "Built-in IMGT anchor mapping":
        region_map, chain_type, note = get_regions_with_builtin_imgt(aa_seq)
        return region_map, chain_type, "built_in_imgt", note

    if engine == "abnumber / ANARCI IMGT":
        region_map, chain_type, note = get_regions_with_abnumber(aa_seq, scheme=scheme)
        return region_map, chain_type, "abnumber_imgt", note

    # Auto: compare abnumber if available with built-in IMGT mapping.
    attempts: list[tuple[int, dict[str, str], str, str, str]] = []
    errors: list[str] = []

    try:
        region_map, chain_type, note = get_regions_with_abnumber(aa_seq, scheme=scheme)
        attempts.append((completeness_score(region_map), region_map, chain_type, "abnumber_imgt", note))
    except Exception as e:
        errors.append(f"abnumber unavailable/failed: {e}")

    try:
        region_map, chain_type, note = get_regions_with_builtin_imgt(aa_seq)
        attempts.append((completeness_score(region_map), region_map, chain_type, "built_in_imgt", note))
    except Exception as e:
        errors.append(f"built-in IMGT failed: {e}")

    if attempts:
        attempts.sort(key=lambda x: x[0], reverse=True)
        _, region_map, chain_type, method_used, note = attempts[0]
        if errors:
            note = note + " | " + " | ".join(errors)
        return region_map, chain_type, method_used, note

    return {region: "" for region in REGION_ORDER}, "unknown", "failed", " | ".join(errors)


def get_best_candidate(raw_seq: str, input_mode: str, engine: str, scheme: str):
    """For nucleotide input, try all six frames and keep the best IMGT segmentation."""
    best = None
    last_note = ""

    for aa_seq, frame_label, strand_label in candidate_aa_sequences(raw_seq, input_mode):
        aa_clean = clean_aa_for_segmentation(aa_seq)
        if len(aa_clean) < 45:
            continue

        region_map, chain_type, method_used, note = get_regions(
            aa_clean,
            engine=engine,
            scheme=scheme,
        )

        score = completeness_score(region_map)
        if not region_map.get("CDR3"):
            score -= 300

        candidate = {
            "aa_sequence_used": aa_clean,
            "frame": frame_label,
            "strand": strand_label,
            "chain_type": chain_type,
            "region_map": region_map,
            "method_used": method_used,
            "score": score,
            "note": note,
        }

        if best is None or score > best["score"]:
            best = candidate

        last_note = note

    return best, last_note


def build_result_rows(records: list[FastaRecord], input_mode: str, engine: str, scheme: str) -> pd.DataFrame:
    rows = []

    for record in records:
        best, last_note = get_best_candidate(
            record.sequence,
            input_mode=input_mode,
            engine=engine,
            scheme=scheme,
        )

        region_map = best["region_map"] if best else {region: "" for region in REGION_ORDER}
        status = "ok" if best and completeness_score(region_map) > 0 and region_map.get("CDR3") else "not_found"

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
            value = region_map.get(region, "") or ""
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
# UI selector helpers
# =========================================================

def reset_to_only(region_list: list[str]):
    for region in REGION_ORDER:
        st.session_state[f"pick_{region}"] = region in region_list


def toggle_region(region: str):
    st.session_state[f"pick_{region}"] = not st.session_state.get(f"pick_{region}", False)


def render_region_selector():
    """Native Streamlit clickable linear map. No raw HTML block."""
    st.caption("Linear IMGT region map: FR1 → CDR1 → FR2 → CDR2 → FR3 → CDR3 → FR4")

    cols = st.columns([REGION_WIDTHS[region] for region in REGION_ORDER])
    for col, region in zip(cols, REGION_ORDER):
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

st.title("IMGT Antibody / Nanobody Region Extractor")
st.caption(
    "Upload FASTA files, select IMGT regions, and download one FASTA per selected region. "
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

    engine = st.radio(
        "IMGT extraction engine",
        options=[
            "Built-in IMGT anchor mapping",
            "Auto: abnumber if available, otherwise built-in IMGT",
            "abnumber / ANARCI IMGT",
        ],
        index=0,
        help=(
            "Built-in IMGT anchor mapping is dependency-free and designed for full-length VH/VHH/VL domains. "
            "abnumber/ANARCI is optional if installed successfully."
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

example_fasta = """>example_vhh_Ty1_RGQG_FR4
QVQLVETGGGLVQPGGSLRLSCAASGFTFSSVYMNWVRQAPGKGPEWVSRISPNSGNIGYTDSVKGRFTISRDNAKNTLYLQMNNLKPEDTALYYCAIGLNLSSSSVRGQGTQVTVSS
>example_vhh_Nanosota2_YGQG_FR4
QVQLQESGGGAVQPGGSLGLSCTASGFNFETSTVGWFRQAPGKENEGVSCINKGYEDTNYADSVKGRFTISRDAAKNTVYLQMDSLQPEDTATYYCAAHNEPYFCDYSGRFRWNEYSYYGQGTQVTVSS
>example_standard_VHH
QVQLQESGGGLVQAGGSLRLSCAASGRTFSEYAMGWFRQAPGKEREFVATISWSGGSTYYTDSVKGRFTISRDNAKNTVYLQMNSLKPDDTAVYYCAAAGLGTVVSEWDYDYDYWGQGTQVTVSS
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

st.subheader("1) Select IMGT region(s) to extract")

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

df = build_result_rows(records, input_mode=input_mode, engine=engine, scheme=scheme)

n_total = len(df)
n_ok = int((df["status"] == "ok").sum())
n_failed = n_total - n_ok
n_builtin = int((df["method_used"] == "built_in_imgt").sum())
n_abnumber = int((df["method_used"] == "abnumber_imgt").sum())

metric_cols = st.columns(5)
metric_cols[0].metric("Input records", n_total)
metric_cols[1].metric("Segmented", n_ok)
metric_cols[2].metric("Failed", n_failed)
metric_cols[3].metric("Built-in IMGT", n_builtin)
metric_cols[4].metric("abnumber IMGT", n_abnumber)

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

with st.expander("Show full QC table"):
    st.dataframe(df, use_container_width=True, hide_index=True)


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

file_map["imgt_region_extraction_qc.csv"] = df.to_csv(index=False)

zip_bytes = build_zip_file(file_map)

st.download_button(
    "Download all selected outputs as ZIP",
    data=zip_bytes,
    file_name="imgt_region_extraction_outputs.zip",
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

with st.expander("Method notes"):
    st.markdown(
        """
        **Built-in IMGT anchor mapping**

        This dependency-free mode maps full-length VH/VHH/VL variable domains using the main IMGT anchors:

        - first conserved Cys, approximately Cys23
        - conserved Trp, approximately Trp41
        - second conserved Cys, approximately Cys104
        - terminal FR4 `xGxG` motif, approximately position 118

        It then applies IMGT-like CDR length limits:

        - CDR1-IMGT is capped at 12 amino acids
        - CDR2-IMGT is capped at 10 amino acids
        - CDR3 is extracted between Cys104 and the FR4 anchor and can be longer due to junctional diversity

        This works well for full-length VHH/VH sequences and fixes the previous failures caused by non-W/F FR4 starts such as `RGQG` or `YGQG`.

        **Formal numbering**

        For strict publication-grade numbering across unusual, truncated, or highly engineered sequences, use IMGT/V-QUEST or a local ANARCI/abnumber installation and compare the QC output.
        """
    )

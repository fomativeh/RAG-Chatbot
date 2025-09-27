import re
from typing import Dict, Optional

# Canonical metadata lookups for known default documents
KNOWN_FILES: Dict[str, Dict[str, str]] = {
    # EU GDPR
    "gdpr_en.pdf": {
        "doc_type": "regulation",
        "jurisdiction": "EU",
        "source_url": "https://eur-lex.europa.eu/eli/reg/2016/679/oj",
        "title": "General Data Protection Regulation (GDPR)",
    },
    # ePrivacy Directive (still a directive; we classify as regulation-like if needed)
    "eprivacy_2002_58_ec_en.pdf": {
        "doc_type": "regulation",
        "jurisdiction": "EU",
        "source_url": "https://eur-lex.europa.eu/eli/dir/2002/58/oj",
        "title": "Directive 2002/58/EC (ePrivacy)",
    },
    # UK DPA 2018
    "uk_dpa_2018_en.pdf": {
        "doc_type": "statute",
        "jurisdiction": "UK",
        "source_url": "https://www.legislation.gov.uk/ukpga/2018/12/contents",
        "title": "UK Data Protection Act 2018",
    },
    # HIPAA Privacy Rule (eCFR)
    "hipaa_privacy_rule_ecfr.html": {
        "doc_type": "regulation",
        "jurisdiction": "US",
        "source_url": "https://www.ecfr.gov/current/title-45/subtitle-A/subchapter-C/part-164/subpart-E",
        "title": "HIPAA Privacy Rule (eCFR Part 164 Subpart E)",
    },
    # California CCPA/CPRA (Civil Code Title 1.81.5)
    "ccpa_ccra_civ_code_title_1_81_5.html": {
        "doc_type": "statute",
        "jurisdiction": "US-CA",
        "source_url": "https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?lawCode=CIV&division=3.&title=1.81.5.&part=4.",
        "title": "California Consumer Privacy Act/CPRA (CIV Title 1.81.5)",
    },
    # Licenses
    "license_mit.txt": {
        "doc_type": "license",
        "jurisdiction": "International",
        "source_url": "https://opensource.org/license/mit/",
        "title": "MIT License",
    },
    "license_apache_2_0.txt": {
        "doc_type": "license",
        "jurisdiction": "International",
        "source_url": "https://www.apache.org/licenses/LICENSE-2.0",
        "title": "Apache License 2.0",
    },
    "license_gpl_3_0.txt": {
        "doc_type": "license",
        "jurisdiction": "International",
        "source_url": "https://www.gnu.org/licenses/gpl-3.0.en.html",
        "title": "GNU General Public License v3.0",
    },
    "cc_by_4_0_legalcode_en.html": {
        "doc_type": "license",
        "jurisdiction": "International",
        "source_url": "https://creativecommons.org/licenses/by/4.0/legalcode",
        "title": "Creative Commons Attribution 4.0 International",
    },
    # Case law
    "case_brown_v_board_1954.txt": {
        "doc_type": "case_law",
        "jurisdiction": "US",
        "source_url": "https://www.courtlistener.com/opinion/105370/brown-v-board-of-education-of-topeka/",
        "title": "Brown v. Board of Education (1954)",
    },
    # Sample contract (added to default docs)
    "sample_nda.txt": {
        "doc_type": "contract",
        "jurisdiction": "International",
        "source_url": None,
        "title": "Sample Mutual NDA",
    },
}


def infer_file_metadata(filename: str) -> Dict[str, Optional[str]]:
    key = filename.lower()
    if key in KNOWN_FILES:
        return KNOWN_FILES[key].copy()

    # Heuristics for unknown files
    md: Dict[str, Optional[str]] = {"doc_type": None, "jurisdiction": None, "source_url": None, "title": None}
    name = key
    if any(x in name for x in ["license", "licence", "mit", "apache", "gpl", "creativecommons", "cc_by"]):
        md["doc_type"] = "license"
        md["jurisdiction"] = "International"
    elif any(x in name for x in ["case_", " v ", " v."]):
        md["doc_type"] = "case_law"
        md["jurisdiction"] = "US"
    elif any(x in name for x in ["act", "code", "statute", "law"]):
        md["doc_type"] = "statute"
    elif any(x in name for x in ["regulation", "directive", "ecfr"]):
        md["doc_type"] = "regulation"
    else:
        md["doc_type"] = "document"

    if any(x in name for x in ["uk", "uk_", "_uk_"]):
        md["jurisdiction"] = md["jurisdiction"] or "UK"
    if any(x in name for x in ["eur", "gdpr", "eprivacy"]):
        md["jurisdiction"] = md["jurisdiction"] or "EU"
    if any(x in name for x in ["ccpa", "california", "civ_code", "us-ca"]):
        md["jurisdiction"] = md["jurisdiction"] or "US-CA"
    if any(x in name for x in ["hipaa", "ecfr", "us_"]):
        md["jurisdiction"] = md["jurisdiction"] or "US"

    return md


# Basic section/article extraction from a chunk
SECTION_PATTERNS = [
    re.compile(r"^(Article)\s+([0-9A-Za-z]+)\s*[:\-–]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(Art\.)\s*([0-9A-Za-z]+)\s*[:\-–]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(Section)\s+([0-9A-Za-z\.\-]+)\s*[:\-��]?\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(§)\s*([0-9A-Za-z\.\-]+)\s*(.*)$", re.IGNORECASE),
    re.compile(r"^(Clause)\s+([0-9A-Za-z\.\-]+)\s*[:\-–]?\s*(.*)$", re.IGNORECASE),
]


def extract_chunk_section(text: str) -> Dict[str, Optional[str]]:
    # Look at the first few non-empty lines for a heading
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[:8]:
        for pat in SECTION_PATTERNS:
            m = pat.match(ln)
            if m:
                kind = m.group(1).title()
                ident = m.group(2)
                title = (m.group(3) or "").strip() or None
                # Normalize kind to a neutral key
                return {"section": f"{kind} {ident}", "section_title": title}
    return {"section": None, "section_title": None}


def citation_label(meta: Dict, default_index: int) -> str:
    source = meta.get("source", "doc")
    sec = meta.get("section") or meta.get("article")
    sec_title = meta.get("section_title")
    if sec:
        if sec_title:
            return f"{source} — {sec}: {sec_title}"
        return f"{source} — {sec}"
    # Fallback: friendlier than raw chunk index
    return f"{source} — excerpt"

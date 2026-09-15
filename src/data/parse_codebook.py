"""Parse the official WVS-7 Variables Report into machine-readable metadata.

Item selection must be grounded in the official question definitions rather than
hand-written metadata, so this script extracts every question's wording, response
options and thematic block straight from the WVS-7 codebook PDF.

Source document (free, no registration form -- it is documentation, not microdata):
    https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp
    -> Documentation -> "WVS7 Codebook Variables report V6.0.pdf"

Usage:
    python -m src.data.parse_codebook --pdf data/raw/WVS7_Codebook_Variables_report_V6.0.pdf

Writes data/reference/wvs7_codebook.json, which IS committed -- it is derived
metadata, not survey responses, so the WVSA redistribution clause does not apply.
"""

import argparse
import json
import logging
import re
from pathlib import Path

from src.config import DATA_RAW, DATA_REFERENCE

logger = logging.getLogger(__name__)

DEFAULT_PDF = DATA_RAW / "WVS7_Codebook_Variables_report_V6.0.pdf"

# Thematic blocks, taken verbatim from the codebook table of contents.
# Q260-Q290 is the demographic block: those are prompt conditioning variables,
# never prediction targets.
BLOCKS = [
    (1, 45, "social_values"),
    (46, 56, "happiness_wellbeing"),
    (57, 105, "social_capital_trust"),
    (106, 111, "economic_values"),
    (112, 120, "corruption"),
    (121, 130, "migration"),
    (131, 151, "security"),
    (152, 157, "postmaterialism"),
    (158, 163, "science_tech"),
    (164, 175, "religious_values"),
    (176, 198, "ethical_values"),
    (199, 234, "political_participation"),
    (235, 259, "political_culture"),
    (260, 290, "DEMOGRAPHIC"),
]

# "Q123 Some title" at the start of a line opens a new question block.
QUESTION_HEADER = re.compile(r"^\s*(Q\d+[A-Z]?(?:_[A-Z0-9]+)?)\s+(\S.*)$", re.M)
# "1.- Label" or "7-25.- Label" (a code range) is a substantive response option.
RESPONSE_OPTION = re.compile(r"^\s*(\d+)(?:-(\d+))?\.-\s*(.+?)\s*$", re.M)
# "-1-.- Don't know" etc. are the missing-data codes.
MISSING_OPTION = re.compile(r"^\s*-(\d+)-\.-", re.M)


def block_for(question_number: int) -> str:
    for low, high, name in BLOCKS:
        if low <= question_number <= high:
            return name
    return "other"


def extract_text(pdf_path: Path) -> str:
    """Extract the codebook text, stripping the repeated page header/footer."""
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover
        raise ImportError("parse_codebook needs pypdf: uv pip install pypdf") from exc

    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except Exception:  # a damaged page should not abort the whole parse
            pages.append("")

    text = "\n".join(pages)
    text = re.sub(r"The WORLD VALUES SURVEY ASSOCIATION\s*", "", text)
    text = re.sub(r"www\.worldvaluessurvey\.org\s*", "", text)
    return text


def parse_codebook(text: str) -> dict:
    """Build {question_id: metadata} from the extracted codebook text."""
    headers = list(QUESTION_HEADER.finditer(text))
    codebook = {}

    for i, header in enumerate(headers):
        question_id = header.group(1)
        title = header.group(2).strip()
        body_start = header.end()
        body_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        body = text[body_start:body_end]

        # Substantive response options, expanding any "7-25.-" style code ranges.
        options = {}
        for match in RESPONSE_OPTION.finditer(body):
            low = int(match.group(1))
            high = int(match.group(2)) if match.group(2) else low
            label = match.group(3).strip()
            for code in range(low, high + 1):
                options.setdefault(code, label)

        codes = sorted(options)
        number = int(re.match(r"Q(\d+)", question_id).group(1))

        # A contiguous integer run is a necessary (not sufficient) condition for
        # an ordinal scale -- it rules out sparse/nominal coding schemes.
        contiguous = bool(codes) and (codes[-1] - codes[0] + 1) == len(codes)

        codebook[question_id] = {
            "title": title,
            "wording": " ".join(body.split("\n")[:3]).strip()[:300],
            "valid_codes": {str(code): options[code] for code in codes},
            "n_scale": len(codes),
            "min_code": codes[0] if codes else None,
            "max_code": codes[-1] if codes else None,
            "contiguous": contiguous,
            "missing_codes": sorted({int(m.group(1)) for m in MISSING_OPTION.finditer(body)}),
            "numeric_variable": bool(re.search(r"Numeric variable", body)),
            "block": block_for(number),
        }

    return codebook


def main(pdf_path: Path = DEFAULT_PDF, output_path: Path = None) -> bool:
    if output_path is None:
        output_path = DATA_REFERENCE / "wvs7_codebook.json"

    if not pdf_path.exists():
        logger.error(
            f"Codebook PDF not found at {pdf_path}\n"
            "Download 'WVS7 Codebook Variables report V6.0.pdf' from\n"
            "  https://www.worldvaluessurvey.org/WVSDocumentationWV7.jsp (Documentation section)\n"
            "It needs no registration form. See DATA_ACQUISITION.md."
        )
        return False

    codebook = parse_codebook(extract_text(pdf_path))
    if not codebook:
        logger.error("Parsed zero questions -- the PDF layout may have changed.")
        return False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "_source": "WVS-7 Variables Report V6.0 (worldvaluessurvey.org)",
        "_note": "Derived question metadata, not survey microdata.",
        "n_questions": len(codebook),
        "questions": codebook,
    }
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=1, ensure_ascii=False)

    logger.info(f"✓ Parsed {len(codebook)} questions to {output_path}")
    return True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf", type=Path, default=DEFAULT_PDF)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    raise SystemExit(0 if main(args.pdf, args.output) else 1)

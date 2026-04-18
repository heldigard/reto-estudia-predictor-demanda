from __future__ import annotations

import csv
import re
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PDF_PATH = ROOT / "dataset.pdf"
OUTPUT_PATH = ROOT / "backend" / "data" / "dataset.csv"
NOTE_VALUES = {
    "duplicado",
    "ok",
    "negativo",
    "missing",
    "missing_bloque",
    "outlier_alto",
    "outlier_bajo",
}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}\b")
VALUE_RE = re.compile(r"^(?P<product>.*?)(?:\s+)?(?P<value>-?\d+\.\d+)$")
VALUE_WITH_SPACE_RE = re.compile(r"^(?P<product>.+?)\s+(?P<value>-?\d+\.\d+)$")


@dataclass(frozen=True)
class ParsedRow:
    fecha: str
    sku: str
    producto: str
    unidades_vendidas: float | None
    nota: str

    def as_csv_row(self) -> list[str]:
        return [
            self.fecha,
            self.sku,
            self.producto,
            "" if self.unidades_vendidas is None else f"{self.unidades_vendidas:.1f}",
            self.nota,
        ]


def extract_pdf_text(pdf_path: Path) -> str:
    result = subprocess.run(
        ["pdftotext", "-raw", str(pdf_path), "-"],
        capture_output=True,
        check=True,
        text=True,
    )
    return result.stdout


def rebuild_logical_lines(raw_text: str) -> list[str]:
    logical_lines: list[str] = []
    current: str | None = None

    for original_line in raw_text.splitlines():
        line = original_line.replace("\x0c", "").strip()
        if not line or line in {"fecha sku producto unidades_vendidas", "nota"}:
            continue

        if DATE_RE.match(line):
            if current is not None:
                logical_lines.append(current)
            current = line
            continue

        if current is None:
            raise ValueError(f"Found continuation line without active record: {line!r}")

        current = f"{current} {line}"

    if current is not None:
        logical_lines.append(current)

    return logical_lines


def split_note_from_line(line: str) -> tuple[str, str, str, str]:
    try:
        fecha, sku, rest = line.split(maxsplit=2)
    except ValueError as exc:
        raise ValueError(f"Could not split base columns from line: {line!r}") from exc

    note: str | None = None
    for candidate in sorted(NOTE_VALUES, key=len, reverse=True):
        suffix = f" {candidate}"
        if rest.endswith(suffix):
            note = candidate
            payload = rest[: -len(suffix)].strip()
            break
        if rest == candidate:
            note = candidate
            payload = ""
            break
    else:
        raise ValueError(f"Could not identify note in line: {line!r}")

    return fecha, sku, payload, note


def build_product_catalog(logical_lines: list[str]) -> dict[str, str]:
    candidates: dict[str, Counter[str]] = {}
    for line in logical_lines:
        _, sku, payload, _ = split_note_from_line(line)
        if not payload:
            continue

        candidate: str | None = None
        spaced_match = VALUE_WITH_SPACE_RE.match(payload)
        if spaced_match:
            candidate = spaced_match.group("product").strip()
        elif not VALUE_RE.match(payload):
            candidate = payload.strip()

        if candidate:
            candidates.setdefault(sku, Counter())[candidate] += 1

    catalog: dict[str, str] = {}
    for sku, counter in candidates.items():
        catalog[sku] = max(counter.items(), key=lambda item: (item[1], len(item[0])))[0]
    return catalog


def parse_logical_line(line: str, product_catalog: dict[str, str]) -> ParsedRow:
    fecha, sku, payload, note = split_note_from_line(line)

    unidades_vendidas: float | None = None
    product = product_catalog.get(sku, payload)
    if payload:
        catalog_product = product_catalog.get(sku)
        if catalog_product and payload.startswith(catalog_product):
            remainder = payload[len(catalog_product) :].strip()
            product = catalog_product
            if remainder:
                unidades_vendidas = float(remainder)
        else:
            spaced_match = VALUE_WITH_SPACE_RE.match(payload)
            match = spaced_match or VALUE_RE.match(payload)
            if match:
                product = match.group("product").strip()
                unidades_vendidas = float(match.group("value"))
            else:
                product = payload.strip()

    if not product:
        raise ValueError(f"Missing product after parsing line: {line!r}")

    return ParsedRow(
        fecha=fecha,
        sku=sku,
        producto=product,
        unidades_vendidas=unidades_vendidas,
        nota=note,
    )


def validate_rows(rows: list[ParsedRow]) -> None:
    if not rows:
        raise ValueError("No rows were parsed from the PDF.")

    unique_dates = sorted({row.fecha for row in rows})
    unique_skus = sorted({row.sku for row in rows})

    if len(unique_dates) < 52:
        raise ValueError(f"Expected at least 52 weeks, got {len(unique_dates)}.")
    if len(unique_skus) != 20:
        raise ValueError(f"Expected 20 unique SKUs, got {len(unique_skus)}.")

    counts_by_date = Counter(row.fecha for row in rows)
    broken_dates = {date: count for date, count in counts_by_date.items() if count != 20}
    if broken_dates:
        raise ValueError(f"Found weeks with missing or extra rows: {broken_dates}")


def write_csv(rows: list[ParsedRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["fecha", "sku", "producto", "unidades_vendidas", "nota"])
        for row in rows:
            writer.writerow(row.as_csv_row())


def main() -> None:
    raw_text = extract_pdf_text(PDF_PATH)
    logical_lines = rebuild_logical_lines(raw_text)
    product_catalog = build_product_catalog(logical_lines)
    rows = [parse_logical_line(line, product_catalog) for line in logical_lines]
    validate_rows(rows)
    write_csv(rows, OUTPUT_PATH)

    unique_dates = len({row.fecha for row in rows})
    missing_count = sum(row.unidades_vendidas is None for row in rows)
    note_counts = Counter(row.nota for row in rows)
    print(
        "Parsed dataset successfully:",
        f"{len(rows)} rows,",
        f"{unique_dates} weeks,",
        f"{missing_count} missing values,",
        f"notes={dict(sorted(note_counts.items()))}",
    )
    print(f"CSV written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

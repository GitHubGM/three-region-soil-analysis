"""Render a Word-exported PDF through the packaged DOCX renderer's Poppler path.

Example (use the bundled Python runtime):
    render_word_pdf.py --docx manuscript.docx --pdf qa/manuscript.pdf --output-dir qa/pages

All artifact paths must be inside the ERA-5 workspace. The managed renderer is
imported without modification; only this process's conversion function is
replaced because Microsoft Word has already completed DOCX-to-PDF conversion.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


WORKSPACE = Path(r"C:\proj\papers\ERA-5").resolve()
DEPENDENCIES = Path(
    r"C:\Users\qorqy\.cache\codex-runtimes\codex-primary-runtime\dependencies"
)
RENDERER = Path(
    r"C:\Users\qorqy\.codex\plugins\cache\openai-primary-runtime\documents"
    r"\26.909.12148\skills\documents\render_docx.py"
)


def workspace_path(value: str, suffix: str | None = None, *, exists: bool = False) -> Path:
    path = Path(value).resolve(strict=exists)
    if not path.is_relative_to(WORKSPACE) or path == WORKSPACE:
        raise ValueError(f"Artifact path must be inside {WORKSPACE}: {value}")
    if suffix is not None and path.suffix.lower() != suffix:
        raise ValueError(f"Expected {suffix} file: {value}")
    if exists and not path.is_file():
        raise ValueError(f"Expected an existing file: {value}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", required=True)
    parser.add_argument("--pdf", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()
    docx = workspace_path(args.docx, ".docx", exists=True)
    pdf = workspace_path(args.pdf, ".pdf", exists=True)
    output_dir = workspace_path(args.output_dir)
    if not 72 <= args.dpi <= 300:
        raise ValueError("DPI must be between 72 and 300")
    if pdf.stat().st_mtime_ns < docx.stat().st_mtime_ns:
        raise ValueError("The PDF predates the DOCX; export it again before rendering")
    with pdf.open("rb") as handle:
        if handle.read(5) != b"%PDF-":
            raise ValueError(f"Not a PDF: {pdf}")

    poppler = DEPENDENCIES / "native" / "poppler" / "Library" / "bin"
    for executable in ("pdfinfo.exe", "pdftoppm.exe"):
        if not (poppler / executable).is_file():
            raise FileNotFoundError(poppler / executable)
    os.environ["PATH"] = str(poppler) + os.pathsep + os.environ.get("PATH", "")
    spec = importlib.util.spec_from_file_location("packaged_render_docx", RENDERER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load renderer: {RENDERER}")
    renderer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(renderer)

    def use_word_pdf(doc_path, user_profile, convert_tmp_dir, stem, verbose=False):
        if Path(doc_path).resolve() != docx:
            raise ValueError("Unexpected document passed to renderer")
        return str(pdf), "PDF exported with Microsoft Word"

    renderer.convert_to_pdf = use_word_pdf
    pages = renderer.rasterize(
        str(docx), str(output_dir), args.dpi, verbose=True, emit_pdf=False
    )
    print(json.dumps({"docx": str(docx), "pdf": str(pdf), "pages": pages}, indent=2))


if __name__ == "__main__":
    main()

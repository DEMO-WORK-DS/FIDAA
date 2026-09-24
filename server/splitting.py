# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors — https://eupl.eu/1.2/en/
"""Markdown splitting + knowledge-file loaders.

`_split_markdown` is an **exact port** of the MarkdownHeaderTextSplitter
semantics (langchain-text-splitters 1.1.2, the version the pre-M2 demo
ran; the port was made by reading that source) — the server's chunks are
byte-identical to what the demo embedded. If this function changes, the
FIDAA-DEMO starter splitter must be re-checked (AGENTS.md).
"""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

logger = logging.getLogger("fidaa.server")

# The knowledge dir ships inside the repo; overridable for unusual layouts.
KNOWLEDGE_DIR = Path(
    os.getenv("KNOWLEDGE_DIR", str(Path(__file__).resolve().parent.parent / "knowledge"))
)
# Optional third tool: a folder of .md files to index (demo parity: the
# tool only exists when this is set; a missing folder disables it w/ warning).
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")


# ---------------------------------------------------------------------------
# Markdown splitting (port of the MarkdownHeaderTextSplitter usage)
# ---------------------------------------------------------------------------
def _split_markdown(
    text: str, levels: tuple[int, ...]
) -> list[tuple[dict[int, str], str]]:
    """Split text into (header_path, chunk) pairs on header lines of *levels*.

    Exact port of langchain's MarkdownHeaderTextSplitter.split_text (source
    read from the demo's venv, 2026-09-23), specialized to ATX headers, so
    the server's chunks are byte-identical to what the demo embeds today:
    - text is split on "\\n"; each line is stripped and non-printable
      characters are removed before any check;
    - fenced code blocks (``` / ~~~) pass through as content and their
      lines are never treated as headers;
    - an ATX header line of an enumerated level starts a new section. The
      header line itself is excluded from the content (strip_headers=True)
      and carried in the path instead; a same-or-deeper header pops the
      stack, so the path always holds the current heading path;
    - blank lines separate paragraphs; paragraphs of one section are
      aggregated into the chunk joined by "  \\n" (markdown hard break);
    - text before the first header becomes a chunk with an empty path
      (e.g. the bibliography intro + table of contents);
    - a section without content produces no chunk.
    """
    level_set = sorted(levels, reverse=True)
    lines = text.split("\n")
    # (metadata snapshot, content) line-units, in order
    units: list[tuple[dict[int, str], str]] = []
    current: list[str] = []
    initial: dict[int, str] = {}  # active heading path
    stack: list[tuple[int, str]] = []  # (level, text)
    in_code = False
    fence = ""

    def flush() -> None:
        nonlocal current
        if current:
            units.append((dict(initial), "\n".join(current)))
            current = []

    for line in lines:
        stripped = line.strip()
        # Drop non-printable chars (langchain parity: control chars never
        # reach the embedding input).
        stripped = "".join(filter(str.isprintable, stripped))

        if not in_code:
            if stripped.startswith("```") and stripped.count("```") == 1:
                in_code, fence = True, "```"
            elif stripped.startswith("~~~"):
                in_code, fence = True, "~~~"
        elif stripped.startswith(fence):
            in_code, fence = False, ""

        if in_code:
            # Code-block lines are content verbatim (incl. blank lines).
            current.append(stripped)
            continue

        is_header = False
        if stripped:
            for lv in level_set:
                sep = "#" * lv
                # Header with no text, or header followed by a space.
                if stripped.startswith(sep) and (
                    len(stripped) == len(sep) or stripped[len(sep)] == " "
                ):
                    header_text = stripped[len(sep) :].strip()
                    while stack and stack[-1][0] >= lv:
                        popped_lv, _ = stack.pop()
                        initial.pop(popped_lv, None)
                    stack.append((lv, header_text))
                    initial[lv] = header_text
                    flush()  # end previous section before the new header
                    is_header = True
                    break
        if not is_header:
            if stripped:
                current.append(stripped)
            elif current:
                # Blank line = paragraph break within the section.
                flush()

    flush()

    # Aggregate consecutive units with the same heading path, joined with
    # "  \n" (markdown hard break) — exactly what langchain aggregates.
    out: list[tuple[dict[int, str], str]] = []
    for meta, content in units:
        if out and out[-1][0] == meta:
            out[-1] = (meta, out[-1][1] + "  \n" + content)
        else:
            out.append((meta, content))
    return out


def _path_list(path: dict[int, str]) -> list[str]:
    """Header path dict {1: '…', 2: '…'} → ordered list of heading texts."""
    return [path[lv] for lv in sorted(path)]


def _unique_paths(paths: list[list[str]]) -> list[list[str]]:
    """Deduplicate heading paths, preserving document order (R6.5 TOC):
    the chunks of one section all share its path, the TOC lists each
    chapter once."""
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for p in paths:
        key = tuple(p)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _load_context() -> list[tuple[list[str], str]]:
    """knowledge/kontext.md → (heading_path, chunk) pairs, H1/H2 split
    (demo parity: build_rag)."""
    text = (KNOWLEDGE_DIR / "kontext.md").read_text(encoding="utf-8")
    return [(_path_list(path), chunk) for path, chunk in _split_markdown(text, (1, 2))]


def _load_bibliography() -> list[tuple[list[str], str]]:
    """knowledge/bibliography.md → one chunk per reference entry
    (demo parity: build_bibliography).

    Each entry ('* **Author (Year)**: …' bullet) becomes its own chunk: a
    citation must never be split across chunks, and the largest H1 section
    (~44 KB) would exceed the 8192-token embedding limit (HTTP 400). The
    enclosing H1 heading is kept as the heading path for source attribution.
    """
    text = (KNOWLEDGE_DIR / "bibliography.md").read_text(encoding="utf-8")
    out: list[tuple[list[str], str]] = []
    for path, section in _split_markdown(text, (1,)):
        heading = _path_list(path)
        for entry in re.split(r"(?m)^(?=\* \*\*)", section):
            entry = entry.strip()
            if entry:
                out.append((heading, entry))
    return out


def _load_documents() -> list[tuple[list[str], str]] | None:
    """DOCUMENTS_PATH → (heading_path, chunk) pairs, H1/H2/H3 split with the
    file's relative path as first path element (demo parity:
    build_document_search). Returns None when the optional tool is off or
    the folder is missing/empty."""
    if not DOCUMENTS_PATH:
        return None
    doc_folder = Path(DOCUMENTS_PATH)
    if not doc_folder.exists():
        logger.warning(
            "DOCUMENTS_PATH %r does not exist — document search disabled.",
            DOCUMENTS_PATH,
        )
        return None
    md_files = sorted(doc_folder.rglob("*.md"))
    if not md_files:
        logger.warning(
            "No .md files found in %r — document search disabled.", DOCUMENTS_PATH
        )
        return None
    out: list[tuple[list[str], str]] = []
    for f in md_files:
        rel = str(f.relative_to(doc_folder))
        for path, chunk in _split_markdown(f.read_text(encoding="utf-8"), (1, 2, 3)):
            out.append(([rel, *_path_list(path)], chunk))
    logger.info(
        "Indexing %d chunks from %d files in %r.", len(out), len(md_files), DOCUMENTS_PATH
    )
    return out


def _load_starters() -> list[tuple[str, str]]:
    """Parse knowledge/prompts.md into (title, message) pairs (demo parity:
    H2 headers are the titles; the leading H1 '# Starters' is ignored).
    The message is the section content — the splitter already excludes the
    header line (strip_headers parity), so it goes straight into the prompt."""
    text = (KNOWLEDGE_DIR / "prompts.md").read_text(encoding="utf-8")
    out: list[tuple[str, str]] = []
    for path, chunk in _split_markdown(text, (2,)):
        if 2 in path:
            out.append((path[2], chunk))
    return out


def _context_chapters() -> list[tuple[list[str], str]]:
    """(heading_path, full text) per context chapter (R6.6).

    Re-splits knowledge/kontext.md (cheap: file read + split, no embedding)
    so build_server() can register resources before the lifespan builds the
    index. Repeated heading paths (rare) are merged so every chapter maps
    to exactly one resource URI; preamble text without a heading is not a
    chapter and is skipped.
    """
    merged: dict[tuple[str, ...], list[str]] = {}
    for path, chunk in _load_context():
        if not path:
            continue
        merged.setdefault(tuple(path), []).append(chunk)
    return [(list(p), "\n\n".join(texts)) for p, texts in merged.items()]

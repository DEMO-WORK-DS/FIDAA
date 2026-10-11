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

import functools
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("fidaa.server")

# The knowledge dir ships inside the repo; overridable for unusual layouts.
KNOWLEDGE_DIR = Path(
    os.getenv("KNOWLEDGE_DIR", str(Path(__file__).resolve().parent.parent / "knowledge"))
)
# Optional third tool: a folder of .md files to index (demo parity: the
# tool only exists when this is set; a missing folder disables it w/ warning).
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
# Public base URL of the hosted FIDAA endpoint + its HTTP file routes
# (/llms.txt, /knowledge/<file>, /kontext/<id>.md). Single definition, so
# server results, INDEX.md and docs reference the same link. Default: the
# public FIDAA-DEMO deployment; set PUBLIC_URL when an instance serves
# different knowledge.
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://fidaa.h2.de/mcp").rstrip("/")


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


def _strip_html_comments(text: str) -> str:
    """Remove <!-- … --> blocks before splitting/indexing.

    The knowledge files open with an SPDX license comment; unstripped it
    became its own 84-char "chapter-less" chunk that could surface as a
    junk search hit. Applied in the loaders (not in _split_markdown, which
    stays an exact port). Whole-file/chapter *delivery* keeps the license
    header — this only affects what gets indexed.
    """
    return re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)


def _load_context() -> list[tuple[list[str], str]]:
    """knowledge/kontext.md → (heading_path, chunk) pairs, H1/H2 split
    (demo parity: build_rag). One chunk = one complete H2 chapter; H2
    sections are never split (human-organized, complete units)."""
    text = _strip_html_comments((KNOWLEDGE_DIR / "kontext.md").read_text(encoding="utf-8"))
    return [(_path_list(path), chunk) for path, chunk in _split_markdown(text, (1, 2))]


def _load_bibliography() -> list[tuple[list[str], str]]:
    """knowledge/bibliography.md → one chunk per reference entry
    (demo parity: build_bibliography).

    Each entry ('* **Author (Year)**: …' bullet) becomes its own chunk: a
    citation must never be split across chunks, and the largest H1 section
    (~44 KB) would exceed the 8192-token embedding limit (HTTP 400). The
    enclosing H1 heading is kept as the heading path for source attribution.
    """
    # SPDX comment stripped before indexing (see _strip_html_comments).
    text = _strip_html_comments((KNOWLEDGE_DIR / "bibliography.md").read_text(encoding="utf-8"))
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
        # HTML comments (e.g. license headers) are not indexed content.
        text = _strip_html_comments(f.read_text(encoding="utf-8"))
        for path, chunk in _split_markdown(text, (1, 2, 3)):
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


# ---------------------------------------------------------------------------
# Chapters (complete H2 sections, verbatim source) + stable chapter IDs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Chapter:
    """One complete H2 chapter of kontext.md, as written in the source.

    Delivered whole by fetch / resources / HTTP — an H2 chapter is a
    human-organized, complete unit and is never split (maintainer rule).
    """

    id: str  # stable slug of the H2 heading, e.g. "muss-ich-verschwiegen-sein-…"
    heading_path: list[str]  # [H1, H2]
    text: str  # verbatim markdown incl. the "## …" heading line
    line_ranges: list[tuple[int, int]]  # 1-based, inclusive, in kontext.md


def _atx_header(stripped: str, levels: tuple[int, ...]) -> tuple[int, str] | None:
    """(level, text) if *stripped* is an ATX header of one of *levels* —
    the same rule _split_markdown applies (header text may be empty)."""
    for lv in sorted(levels, reverse=True):
        sep = "#" * lv
        if stripped.startswith(sep) and (len(stripped) == len(sep) or stripped[len(sep)] == " "):
            return lv, stripped[len(sep) :].strip()
    return None


def _scan_headers(lines: list[str], levels: tuple[int, ...]) -> list[tuple[int, int, str]]:
    """(line_index, level, text) of every header of *levels*, skipping
    fenced code blocks and normalizing lines exactly like _split_markdown
    (strip + drop non-printables), so both always agree on chapter starts."""
    out: list[tuple[int, int, str]] = []
    in_code, fence = False, ""
    for i, line in enumerate(lines):
        stripped = "".join(filter(str.isprintable, line.strip()))
        if not in_code:
            if stripped.startswith("```") and stripped.count("```") == 1:
                in_code, fence = True, "```"
            elif stripped.startswith("~~~"):
                in_code, fence = True, "~~~"
        elif stripped.startswith(fence):
            in_code, fence = False, ""
        if in_code:
            continue
        hit = _atx_header(stripped, levels)
        if hit:
            out.append((i, hit[0], hit[1]))
    return out


def _slug(heading: str, max_len: int = 80) -> str:
    """ASCII slug of a heading for IDs/URLs: lowercase, German umlauts
    transliterated (ä→ae …), other characters → '-', cut at a word
    boundary. Readable for agents, stable while the heading is unchanged."""
    s = heading.lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > max_len:
        s = s[:max_len].rsplit("-", 1)[0]
    return s or "kapitel"


def chapter_url(chapter_id: str) -> str:
    """Canonical, citable URL of one complete chapter (served by the
    server's /mcp/kontext/<id>.md route)."""
    return f"{PUBLIC_URL}/kontext/{chapter_id}.md"


@functools.lru_cache(maxsize=1)
def _context_chapters() -> tuple[Chapter, ...]:
    """All H2 chapters of knowledge/kontext.md, verbatim, in document order.

    Scans the raw source (no embedding, cheap) so build_server() can
    register resources before the lifespan builds the index; cached per
    process (the server never re-reads knowledge at runtime). Sections
    with an identical H2 heading are merged into one chapter (one ID ↔ one
    complete text); distinct headings whose slugs collide get "-2", "-3".
    Empty H2 sections (e.g. a bare "##") are not chapters.
    """
    lines = (KNOWLEDGE_DIR / "kontext.md").read_text(encoding="utf-8").split("\n")
    headers = _scan_headers(lines, (1, 2))
    by_heading: dict[str, dict] = {}
    used_ids: set[str] = set()
    h1 = ""
    for n, (i, lv, text) in enumerate(headers):
        if lv == 1:
            h1 = text
            continue
        end = headers[n + 1][0] if n + 1 < len(headers) else len(lines)
        # Trim trailing blank lines so ranges/texts end at the last content line.
        while end > i + 1 and not lines[end - 1].strip():
            end -= 1
        if not text or end <= i + 1:
            continue  # bare "##" or heading without content
        entry = by_heading.get(text)
        if entry is None:
            cid, k = _slug(text), 2
            while cid in used_ids:
                cid, k = f"{_slug(text)}-{k}", k + 1
            used_ids.add(cid)
            entry = by_heading[text] = {"id": cid, "path": [h1, text], "ranges": []}
        entry["ranges"].append((i + 1, end))  # 1-based inclusive
    return tuple(
        Chapter(
            id=e["id"],
            heading_path=e["path"],
            text="\n\n".join("\n".join(lines[s - 1 : t]) for s, t in e["ranges"]),
            line_ranges=e["ranges"],
        )
        for e in by_heading.values()
    )


# ---------------------------------------------------------------------------
# Whole-file delivery (option: ingest the complete knowledge base)
# ---------------------------------------------------------------------------
# Files an agent may read whole (tool, resources, HTTP). INDEX.md is the
# generated chapter map (scripts/build_index.py).
KNOWLEDGE_FILES = (
    "kontext.md",
    "bibliography.md",
    "systemprompt.md",
    "prompts.md",
    "welcome.md",
    "INDEX.md",
)
# Max chars per part when a file is read through the tool: ~60k chars
# ≈ 20k tokens (German), below common MCP client output caps (e.g. Claude
# Code's default 25k-token limit), so no client truncates a part silently.
PART_CHARS = 60_000


def _file_parts(name: str, max_chars: int = PART_CHARS) -> list[str]:
    """Split a knowledge file into parts of ≤ max_chars, cutting ONLY
    before an H1/H2 heading line — so an H2 chapter is never split across
    parts (a single oversized chapter becomes its own, larger part).
    Concatenating all parts with "\\n" reproduces the file exactly."""
    lines = (KNOWLEDGE_DIR / name).read_text(encoding="utf-8").split("\n")
    cuts = [i for i, _, _ in _scan_headers(lines, (1, 2)) if i > 0]
    bounds = [0, *cuts, len(lines)]
    units = ["\n".join(lines[a:b]) for a, b in zip(bounds, bounds[1:])]
    parts: list[str] = []
    for u in units:
        if parts and len(parts[-1]) + 1 + len(u) <= max_chars:
            parts[-1] += "\n" + u
        else:
            parts.append(u)
    return parts

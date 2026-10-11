# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors — https://eupl.eu/1.2/en/
"""FIDAA MCP server.

Exposes the FIDAA knowledge base (knowledge/*.md, Digital Streetwork) as MCP
tools and prompts:

  tools:    search_context, search_bibliography, list_sections, [search_documents],
            search + fetch (chapter IDs ↔ complete chapters), read_knowledge_file
  prompts:  fidaa_systemprompt, fidaa_starter_1 … fidaa_starter_9
  resources: fidaa://context/<H1>/<H2> — full chapter texts (R6.6),
             fidaa://knowledge/<file> — complete knowledge files
  HTTP:     /mcp/llms.txt, /mcp/knowledge/<file>, /mcp/kontext/<id>.md

Invariant: an H2 chapter of kontext.md is never split — every path that
hands knowledge to a model (search hit, fetch, resource, file part, HTTP)
delivers complete chapters.

Design (decisions Q1–Q10, see the FIDAA-DEMO repo's CURRENT_TASK / the
dev-wiki options analysis):
- v2 SDK (mcp>=2): MCPServer, stdio + streamable-http.
- In-memory numpy index, re-embedded on every startup — exact parity with
  the demo, which rebuilt its PGVector collection on every boot. No
  LangChain, no database, no credentials beyond the embedding endpoint.
- Hybrid search (R6.1): BM25 (rank-bm25, lexical — strong for German
  keywords / exact citation hits) fused with the cosine scores via
  Reciprocal Rank Fusion (k=60); top-k output (k=4) and result format
  unchanged. Pure-vector fallback when BM25 has no match at all.
- /healthz via custom_route (unauthenticated, for compose healthchecks);
  optional MCP_TOKEN bearer gate (default off); transport_security Host
  allowlist so the SDK's 421 guard works behind compose service names.
- Bare GET /mcp (no MCP session) serves mcp-landing.html — an info page
  for humans/agents opening the URL in a browser; the session-GET (SSE
  stream) and all POSTs always reach the MCP app.

Layout (split for reviewability, 2026-09-24; entry point unchanged):
  splitting.py — markdown splitter (verified port) + knowledge loaders
  index.py     — in-memory hybrid index, hybrid search, result models
  main.py      — server assembly (tools/prompts/resources/healthz),
                 token gate, transports
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import quote

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
# ToolError (not ValueError/RuntimeError): the SDK v2 only passes the
# message of an *anticipated* error to the model; other exceptions reach
# the client as a bare "Error executing tool …" without our guidance.
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse

# `import index` (module attribute access), NOT `from index import
# INDEX_READY`: build_indexes() *rebinds* INDEX_READY / INDEX_ERROR, and a
# from-import would be a stale snapshot taken at import time (a bool cannot
# be "mutated" across modules). INDEX (dict) would survive either way, but
# reading all three through the module keeps the rule uniform.
import index
from index import (
    MAX_TOP_K,
    TOP_K,
    ChapterDocument,
    CollectionSections,
    KnowledgeFilePart,
    SearchHit,
    SearchHits,
    SectionEntry,
    SectionListing,
    SearchResults,
    search_collection,
)
from splitting import (
    KNOWLEDGE_DIR,
    KNOWLEDGE_FILES,
    DOCUMENTS_PATH,
    Chapter,
    _context_chapters,
    _file_parts,
    _load_documents,
    _load_starters,
    _unique_paths,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fidaa.server")

# ---------------------------------------------------------------------------
# Configuration — HTTP-mode secrets / guards (embedding + knowledge config
# lives in index.py / splitting.py)
# ---------------------------------------------------------------------------
# HTTP mode only. MCP_TOKEN empty = open (compose-internal by default).
MCP_TOKEN = os.getenv("MCP_TOKEN", "")
# Host allowlist for the SDK's 421 guard, comma-separated hostnames
# (no ports). In docker compose this must contain the service name.
MCP_ALLOWED_HOSTS = [
    h.strip() for h in os.getenv("MCP_ALLOWED_HOSTS", "").split(",") if h.strip()
]


def _expand_hosts(hosts: list[str]) -> list[str]:
    """Expand each entry to exact + wildcard-port form.

    The SDK matches the full Host header (which carries ``:port``) by exact
    match, so ``fidaa`` alone would not admit ``Host: fidaa:8002`` — every
    entry is registered as both ``fidaa`` and ``fidaa:*``.
    """
    out: list[str] = []
    for h in hosts:
        if not h.endswith(":*"):
            out.append(h)
        out.append(h if h.endswith(":*") else f"{h}:*")
    return out


# Info page served for a bare GET /mcp (see _McpLanding). Lives in the repo
# root — /app/mcp-landing.html in the standalone image.
_MCP_LANDING = Path(__file__).resolve().parent.parent / "mcp-landing.html"
# llms.txt (repo root): agent entry point, also served at /mcp/llms.txt.
_LLMS_TXT = Path(__file__).resolve().parent.parent / "llms.txt"

# Public base URL of the HTTP file routes (S10/D3), used for the canonical
# chapter/file URLs in tool results (ChatGPT only cites results with a
# url). Routes live under /mcp/… because the FIDAA-DEMO Caddy proxies only
# /mcp* to this server. Default = the public FIDAA-DEMO deployment, which
# serves the same knowledge; self-hosters with changed content set this.
PUBLIC_URL = os.getenv("PUBLIC_URL", "https://fidaa.h2.de/mcp").rstrip("/")

# Hits returned by the ID-only `search` tool (S3): no texts, so a few more
# than search_context's default are cheap and help deep-research clients.
SEARCH_HITS = 8

# MCP tool annotations (S7): every FIDAA tool only reads the bundled,
# static knowledge — clients may run them without confirmation prompts.
_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


# S4: optional result count, shared by the search tools. Each hit is a
# complete H2 chapter (never cut), so keep it small. Module level: with
# `from __future__ import annotations` the SDK resolves annotations in the
# module globals.
TopK = Annotated[
    int,
    Field(
        ge=1,
        le=MAX_TOP_K,
        description=(
            f"Anzahl der Treffer (1–{MAX_TOP_K}, Standard {TOP_K}). Jeder "
            "Treffer ist ein vollständiges Kapitel (oft mehrere tausend Zeichen)."
        ),
    ),
]


def _chapter_url(chapter_id: str) -> str:
    """Canonical URL of one complete chapter (served by /mcp/kontext/…)."""
    return f"{PUBLIC_URL}/kontext/{quote(chapter_id)}.md"


def _chapter_markdown(ch: Chapter) -> str:
    """A chapter as a standalone markdown file: license/source header
    (CC-BY-SA attribution travels with the excerpt) + the verbatim text."""
    lines = ", ".join(f"{a}–{b}" for a, b in ch.line_ranges)
    return (
        "<!--\nSPDX-License-Identifier: CC-BY-SA-4.0\n"
        "Copyright (C) 2026 FIDAA contributors\n"
        f"Quelle: knowledge/kontext.md, Zeilen {lines} — "
        "https://github.com/DEMO-WORK-DS/FIDAA\n-->\n\n"
        f"{ch.text}\n"
    )


# ---------------------------------------------------------------------------
# Server assembly: tools, prompts, /healthz, lifespan
# ---------------------------------------------------------------------------
def build_server() -> MCPServer:
    """Create the FIDAA MCPServer and register all tools, prompts and the
    health route on it (single place so they cannot drift apart)."""
    server = MCPServer(
        name="fidaa",
        instructions=(
            "FIDAA (Fachinformation Digitale Aufsuchende Arbeit) is a curated, "
            "expert-checked knowledge base on Digital Streetwork (aufsuchende "
            "soziale Arbeit in digitalen Räumen), in German. "
            # Updated for S3/S4/whole-file: lists every tool and access path.
            "Every result is a complete H2 chapter — chapters are never cut. "
            "Tools: search_context returns the most relevant complete chapters "
            "(top_k, default 4) with heading paths and chapter IDs; "
            "fetch(id) returns one complete chapter by ID, and search(query) "
            "returns only chapter IDs/titles/URLs (search/fetch pair); "
            "list_sections lists the chapter structure with IDs; "
            "search_bibliography returns exact source citations "
            "(author/year/title) for a specific work; "
            "search_documents (if present) searches an extra document archive; "
            "read_knowledge_file reads a whole knowledge file (e.g. kontext.md, "
            "~150–175k tokens) in parts — the option to ingest the complete "
            "knowledge base instead of searching. "
            "Prompts: fidaa_systemprompt carries FIDAA's behavior rules; "
            "fidaa_starter_* are ready-made question templates. "
            "Resources: fidaa://context/… (complete chapters) and "
            "fidaa://knowledge/<file> (complete files). The same files are "
            f"plain HTTP downloads: {PUBLIC_URL}/llms.txt. "
            "Formulate queries in German for best recall, and cite sources "
            "(heading path or bibliography entry) when answering with this knowledge."
        ),
        version="0.1.0",
        lifespan=_server_lifespan,
    )
    _register_tools(server)
    _register_prompts(server)
    _register_chapter_resources(server)
    _register_file_routes(server)
    _register_healthz(server)
    return server


def _register_tools(server: MCPServer) -> None:
    """Search tools, the list_sections TOC tool (R6.5), the search/fetch
    chapter pair (S3) and whole-file reading. The optional
    search_documents (Q7) is registered only when the archive is enabled.
    All tools are annotated read-only (S7)."""

    @server.tool(annotations=_READ_ONLY)
    async def search_context(
        query: Annotated[str, Field(description="Die Suchanfrage für die interne Wissensdatenbank.")],
        top_k: TopK = TOP_K,
    ) -> SearchResults:
        """Durchsuche die interne Wissensdatenbank nach relevantem und geprüft richtigem Kontext. Jeder Treffer ist ein vollständiges Kapitel (H2) mit Überschriftenpfad und Kapitel-ID."""
        return SearchResults(results=await search_collection("rag_context", query, top_k))

    @server.tool(annotations=_READ_ONLY)
    async def search_bibliography(
        query: Annotated[
            str,
            Field(
                description="Titel, Autor:in oder Stichwort des Werks, dessen Quellenangabe gesucht wird."
            ),
        ],
        top_k: TopK = TOP_K,
    ) -> SearchResults:
        """Durchsuche die Bibliografie (knowledge/bibliography.md) nach der genauen Quellenangabe für ein bestimmtes Werk (Autor, Jahr, Titel)."""
        return SearchResults(results=await search_collection("rag_bibliography", query, top_k))

    # Optional third tool — registered only when DOCUMENTS_PATH is set (Q7).
    if DOCUMENTS_PATH and _load_documents() is not None:

        @server.tool(annotations=_READ_ONLY)
        async def search_documents(
            query: Annotated[str, Field(description="Die Suchanfrage an das Dokumentenarchiv.")],
            top_k: TopK = TOP_K,
        ) -> SearchResults:
            """Durchsuche das Dokumentenarchiv nach relevantem zusätzlichen Kontext."""
            return SearchResults(results=await search_collection("rag_documents", query, top_k))

    # --- S3: search/fetch chapter pair -----------------------------------
    # Shapes follow OpenAI's MCP "company knowledge compatibility" schema
    # (single string argument each; structuredContent + the same JSON as
    # text, which the SDK emits for pydantic return models). Chapters are
    # always delivered complete (maintainer rule: never split an H2).
    chapters = {c.id: c for c in _context_chapters()}

    @server.tool(annotations=_READ_ONLY)
    async def search(
        query: Annotated[str, Field(description="Suchanfrage (am besten auf Deutsch).")],
    ) -> SearchHits:
        """Finde die passendsten Kapitel der Kontextdatenbank — liefert nur Kapitel-ID, Titel und URL; den vollständigen Kapiteltext holt fetch(id). (search/fetch-Schnittstelle, z. B. für ChatGPT Deep Research.)"""
        hits = await search_collection("rag_context", query, SEARCH_HITS)
        out: list[SearchHit] = []
        for h in hits:
            # Merged duplicate headings map to one ID — list each once.
            if h.chapter_id and h.chapter_id not in {r.id for r in out}:
                out.append(
                    SearchHit(
                        id=h.chapter_id,
                        title=h.heading_path[-1],
                        url=_chapter_url(h.chapter_id),
                    )
                )
        return SearchHits(results=out)

    @server.tool(annotations=_READ_ONLY)
    async def fetch(
        id: Annotated[
            str,
            Field(description="Kapitel-ID aus search, search_context oder list_sections."),
        ],
    ) -> ChapterDocument:
        """Lies ein vollständiges Kapitel (H2-Abschnitt) der Kontextdatenbank anhand seiner Kapitel-ID — immer ungekürzt, wie in der Quelle."""
        ch = chapters.get(id.strip().removesuffix(".md"))
        if ch is None:
            raise ToolError(
                f"Unbekannte Kapitel-ID '{id}'. Gültige IDs liefern search, "
                "search_context und list_sections(collection='context')."
            )
        return ChapterDocument(
            id=ch.id,
            title=ch.heading_path[-1],
            text=ch.text,
            url=_chapter_url(ch.id),
            metadata={
                "heading_path": " > ".join(ch.heading_path),
                "source": "knowledge/kontext.md",
                "lines": ", ".join(f"{a}-{b}" for a, b in ch.line_ranges),
                "license": "CC-BY-SA-4.0",
            },
        )

    # --- Whole-file ingestion option -------------------------------------
    @server.tool(annotations=_READ_ONLY)
    async def read_knowledge_file(
        file: Annotated[
            Literal[KNOWLEDGE_FILES],  # type: ignore[valid-type]
            Field(
                description=(
                    "Wissensdatei: kontext.md (gesamte Kontextdatenbank, ~525k "
                    "Zeichen ≈ 150–175k Tokens), bibliography.md, systemprompt.md, "
                    "prompts.md, welcome.md oder INDEX.md (Kapitelübersicht)."
                )
            ),
        ] = "kontext.md",
        part: Annotated[int, Field(ge=1, description="Teil-Nummer (ab 1).")] = 1,
    ) -> KnowledgeFilePart:
        """Lies eine komplette Wissensdatei (Option: die gesamte Wissensbasis aufnehmen statt zu suchen). Große Dateien kommen in Teilen (≤ ~60k Zeichen, nur an Kapitelgrenzen getrennt); für die ganze Datei alle Teile 1…total_parts lesen. Lohnt sich nur bei ausreichend großem Kontextfenster."""
        parts = _file_parts(file)
        if part > len(parts):
            raise ToolError(f"{file} hat nur {len(parts)} Teil(e).")
        return KnowledgeFilePart(
            file=file,
            part=part,
            total_parts=len(parts),
            text=parts[part - 1],
            url=f"{PUBLIC_URL}/knowledge/{file}",
        )

    # --- TOC tool (R6.5): chapter structure before searching -------------
    # Friendly labels → internal collection names. "documents" is optional
    # (Q7), so it is only listed when actually indexed.
    _COLLECTION_ALIASES = {
        "context": "rag_context",
        "bibliography": "rag_bibliography",
        "documents": "rag_documents",
    }

    @server.tool(annotations=_READ_ONLY)
    async def list_sections(
        collection: Annotated[
            str,
            Field(
                description=(
                    "Sammlung auflisten: 'all' (Standard), 'context', "
                    "'bibliography' oder 'documents'."
                )
            ),
        ] = "all",
    ) -> SectionListing:
        """Liste die Kapitel- und Abschnittsstruktur der Wissensdatenbank als Überschriftenpfade (bei der Kontextdatenbank mit Kapitel-ID) auf — rufe sie auf, um Kapitel gezielt zu suchen oder per fetch(id) vollständig zu lesen."""
        if collection not in _COLLECTION_ALIASES and collection != "all":
            raise ToolError(
                f"Unbekannte Sammlung '{collection}' — erlaubt sind: "
                "all, context, bibliography, documents."
            )
        if not index.INDEX_READY:
            raise ToolError(
                "Wissensdatenbank ist nicht (noch) indexiert"
                + (f" — Fehler: {index.INDEX_ERROR}" if index.INDEX_ERROR else "")
                + ". Bitte später erneut versuchen."
            )
        wanted = (
            list(_COLLECTION_ALIASES.items())
            if collection == "all"
            else [(collection, _COLLECTION_ALIASES[collection])]
        )
        out: list[CollectionSections] = []
        for label, internal in wanted:
            col = index.INDEX.get(internal)
            if col is None:
                # Optional document archive not enabled (DOCUMENTS_PATH unset):
                # skip it for "all", but explain when it was requested.
                if collection == "documents":
                    raise ToolError(
                        "Dokumentenarchiv ist nicht aktiv (DOCUMENTS_PATH "
                        "ist nicht gesetzt)."
                    )
                continue
            # S3: heading path → chapter ID (context only), for fetch.
            ids = {tuple(p): cid for p, cid in zip(col.paths, col.chapter_ids)}
            # Empty path = preamble before the first heading: no chapter.
            out.append(
                CollectionSections(
                    name=label,
                    headings=[
                        SectionEntry(heading_path=p, id=ids.get(tuple(p)))
                        for p in _unique_paths(col.paths)
                        if p
                    ],
                )
            )
        return SectionListing(collections=out)


def _register_prompts(server: MCPServer) -> None:
    """FIDAA's behavior rules + the 9 chat starters (R6.2)."""

    # --- prompts (R6.2) ---------------------------------------------------
    @server.prompt(
        description=(
            "Systemprompt der FIDAA-App: Verhaltensregeln für Antworten, die auf "
            "der FIDAA-Wissensdatenbank basieren. Für Agenten, die wie FIDAA "
            "antworten sollen — den Text als eigenen Systemprompt verwenden. "
            "(MCP-Prompt-Nachrichten kennen nur die Rollen user/assistant, "
            "der Text wird daher als einzelne user-Nachricht transportiert.)"
        )
    )
    def fidaa_systemprompt() -> str:
        """FIDAA's behavior rules (knowledge/systemprompt.md), verbatim.

        The consuming agent is expected to apply this text as its own system
        prompt. A plain string is returned (rendered as one user message)
        because MCP prompt messages only support the user/assistant roles.
        """
        return (KNOWLEDGE_DIR / "systemprompt.md").read_text(encoding="utf-8").strip()

    for i, (title, msg) in enumerate(_load_starters(), start=1):
        server.prompt(
            name=f"fidaa_starter_{i}",
            title=title,
            description=f"Startervorlage {i}: {title}",
        )(_make_starter(msg))


def _register_chapter_resources(server: MCPServer) -> None:
    """Chapter resources (R6.6) + whole-file resources.

    One static resource per context chapter (factory pattern, like the
    starter prompts above). Static URIs — not a {heading} template —
    because heading paths contain '/' (two path segments); a template
    would depend on segment-capture semantics we don't want to rely on.
    Bibliography/documents stay search-only by design (R6.6 scope).
    Chapters are served verbatim from the source (S3: the complete H2
    incl. its heading line, with a license header) instead of the
    splitter's re-joined chunk text.
    """
    chapters = _context_chapters()
    for ch in chapters:
        heading = " > ".join(ch.heading_path)
        server.resource(
            uri="fidaa://context/"
            + "/".join(quote(part, safe="") for part in ch.heading_path),
            name=heading,
            title=f"Kontext: {heading}",
            description=(
                "Volltext eines Kapitels der FIDAA-Kontextdatenbank "
                f"(Markdown, Kapitel-ID {ch.id}). Für Agenten, die "
                "MCP-Ressourcen lesen können."
            ),
            mime_type="text/markdown",
        )(_make_chapter_resource(_chapter_markdown(ch)))
    # Whole-file ingestion option: each knowledge file complete, unsplit
    # (resources have no tool-output cap; read_knowledge_file pages instead).
    for name in KNOWLEDGE_FILES:
        if not (KNOWLEDGE_DIR / name).exists():
            continue  # e.g. INDEX.md not generated in a custom KNOWLEDGE_DIR
        server.resource(
            uri=f"fidaa://knowledge/{name}",
            name=f"knowledge/{name}",
            title=f"Wissensdatei {name} (vollständig)",
            description=(
                "Vollständige Wissensdatei der FIDAA-Wissensbasis (Markdown) — "
                "zum Aufnehmen der gesamten Datei statt einzelner Suchtreffer."
            ),
            mime_type="text/markdown",
        )(_make_file_resource(name))
    logger.info(
        "Registered %d chapter resources (fidaa://context/…) + knowledge files "
        "(fidaa://knowledge/…)",
        len(chapters),
    )


def _register_file_routes(server: MCPServer) -> None:
    """Plain-HTTP knowledge files (S10/D3, HTTP mode only), for agents that
    can fetch URLs but do not speak MCP — and the canonical citation URLs
    of search/fetch. Under /mcp/… because the FIDAA-DEMO Caddy routes
    only /mcp* here. Generated on the fly from knowledge/, never stored in
    git. Announced via llms.txt, the /mcp landing page and README/SKILL.md.

      /mcp/llms.txt            — agent entry point (repo llms.txt)
      /mcp/knowledge/<file>    — a complete knowledge file (KNOWLEDGE_FILES)
      /mcp/kontext/<id>.md     — one complete chapter (IDs: INDEX.md)
    """
    md = "text/markdown"  # Starlette appends "; charset=utf-8" for text/*

    @server.custom_route("/mcp/llms.txt", methods=["GET"])
    async def llms_txt(request) -> PlainTextResponse:  # noqa: ARG001
        if not _LLMS_TXT.exists():  # custom layouts without the repo root
            return PlainTextResponse("llms.txt not available", status_code=404)
        return PlainTextResponse(_LLMS_TXT.read_text(encoding="utf-8"), media_type=md)

    @server.custom_route("/mcp/knowledge/{name}", methods=["GET"])
    async def knowledge_file(request) -> PlainTextResponse:
        name = request.path_params["name"]
        if name not in KNOWLEDGE_FILES or not (KNOWLEDGE_DIR / name).exists():
            return PlainTextResponse(
                f"Unbekannte Datei. Verfügbar: {', '.join(KNOWLEDGE_FILES)}",
                status_code=404,
            )
        return PlainTextResponse(
            (KNOWLEDGE_DIR / name).read_text(encoding="utf-8"), media_type=md
        )

    chapters = {c.id: c for c in _context_chapters()}

    @server.custom_route("/mcp/kontext/{name}", methods=["GET"])
    async def chapter_file(request) -> PlainTextResponse:
        ch = chapters.get(request.path_params["name"].removesuffix(".md"))
        if ch is None:
            return PlainTextResponse(
                "Unbekannte Kapitel-ID — siehe /mcp/knowledge/INDEX.md",
                status_code=404,
            )
        return PlainTextResponse(_chapter_markdown(ch), media_type=md)


def _register_healthz(server: MCPServer) -> None:
    """/healthz (R6.7): unauthenticated, for compose healthchecks."""

    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request) -> JSONResponse:  # noqa: ARG001 — Starlette signature
        """Liveness + index state. 200 only when the index is fully built;
        503 with the error otherwise, so compose `depends_on: healthy`
        gates the app on a working knowledge base."""
        payload: dict[str, Any] = {
            "status": "ok" if index.INDEX_READY else "degraded",
            "ready": index.INDEX_READY,
            "collections": {n: len(c.chunks) for n, c in index.INDEX.items()},
        }
        if index.INDEX_ERROR:
            payload["error"] = index.INDEX_ERROR
        return JSONResponse(payload, status_code=200 if index.INDEX_READY else 503)


def _make_starter(message: str):
    """Closure factory: gives each starter prompt its own function object,
    so the message is captured per iteration (and the prompt has no
    arguments — the SDK rejects defaulted parameters for prompts)."""

    def starter() -> str:
        """One of the 9 FIDAA chat starters (knowledge/prompts.md)."""
        return message

    return starter


def _make_file_resource(name: str):
    """Closure factory for a whole-file resource (read on access, so the
    file is never held twice in memory)."""

    def knowledge_file() -> str:
        """Vollständige Wissensdatei (Markdown)."""
        return (KNOWLEDGE_DIR / name).read_text(encoding="utf-8")

    return knowledge_file


def _make_chapter_resource(text: str):
    """Closure factory: gives each chapter resource its own function object,
    so the text is captured per chapter (pattern shared with _make_starter)."""

    def chapter() -> str:
        """Volltext eines Kontextkapitels (Markdown)."""
        return text

    return chapter


@asynccontextmanager
async def _server_lifespan(server: MCPServer):  # noqa: ARG001 — SDK signature
    """Server lifespan (SDK constructor hook): the SDK calls
    ``lifespan(server)`` and ``async with`` the result — so this function
    (via @asynccontextmanager) *is* the context manager factory. Builds the
    in-memory index exactly once per process, before any transport serves a
    request."""
    await index.build_indexes()
    yield {"ready": index.INDEX_READY}


# ---------------------------------------------------------------------------
# Token gate (optional, Q8: MCP_TOKEN-ready)
# ---------------------------------------------------------------------------
class _BearerGate:
    """Pure-ASGI bearer-token gate for HTTP mode.

    Active only when MCP_TOKEN is set: every request except /healthz
    (compose healthcheck, monitoring) must carry
    `Authorization: Bearer <MCP_TOKEN>`. Comparison is constant-time.
    """

    def __init__(self, app, token: str) -> None:
        self.app = app
        self.token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http" and scope.get("path") != "/healthz":
            headers = dict(scope.get("headers", []))
            auth = headers.get(b"authorization", b"")
            if not hmac.compare_digest(auth, b"Bearer " + self.token.encode()):
                response = JSONResponse({"error": "unauthorized"}, status_code=401)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


class _McpLanding:
    """Info page for a bare GET /mcp (browser, or an agent following the
    link): without it the MCP app answers 400 "Missing session ID".

    Only a session-less GET on exactly /mcp or /mcp/ is intercepted; the
    streamable-HTTP SSE leg (GET with an Mcp-Session-Id) and every POST
    (e.g. initialize) pass through untouched. Served by the server itself
    so the behaviour is identical standalone and in the FIDAA-DEMO
    deployment (Caddy just proxies /mcp*).
    """

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send) -> None:
        if (
            scope["type"] == "http"
            and scope.get("method") == "GET"
            and scope.get("path") in ("/mcp", "/mcp/")
            and b"mcp-session-id" not in dict(scope.get("headers", []))
        ):
            if _MCP_LANDING.exists():
                response = FileResponse(_MCP_LANDING)
                await response(scope, receive, send)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="FIDAA MCP server (stdio or streamable-http)"
    )
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
        help="stdio for local agents (default); streamable-http for docker compose",
    )
    parser.add_argument("--host", default=os.getenv("MCP_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MCP_PORT", "8002")))
    args = parser.parse_args()

    server = build_server()

    if args.transport == "stdio":
        server.run(transport="stdio")
        return

    # HTTP mode. We run the returned Starlette app directly with uvicorn
    # (NOT mounted under another app): the app's own lifespan then runs,
    # which starts the session manager AND our index-building lifespan —
    # mounting it under a host app would kill the lifespan (the classic v2
    # trap: "Task group is not initialized").
    ts = None
    if MCP_ALLOWED_HOSTS:
        ts = TransportSecuritySettings(allowed_hosts=_expand_hosts(MCP_ALLOWED_HOSTS))
    app = server.streamable_http_app(transport_security=ts)
    if not _MCP_LANDING.exists():
        logger.warning(
            "Landing page %s not found — bare GET /mcp falls back to the "
            "MCP app's 400 'Missing session ID'.",
            _MCP_LANDING,
        )
    # add_middleware() prepends: the landing page goes in first, so the
    # optional token gate (added second) stays outermost — a protected
    # endpoint refuses the browser before any page is served.
    app.add_middleware(_McpLanding)
    if MCP_TOKEN:
        app.add_middleware(_BearerGate, token=MCP_TOKEN)
    logger.info(
        "Serving streamable-http on %s:%d (token gate: %s, host allowlist: %s)",
        args.host,
        args.port,
        "on" if MCP_TOKEN else "off",
        MCP_ALLOWED_HOSTS or "default (localhost)",
    )
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()

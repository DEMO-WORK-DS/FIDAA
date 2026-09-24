# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors — https://eupl.eu/1.2/en/
"""FIDAA MCP server.

Exposes the FIDAA knowledge base (knowledge/*.md, Digital Streetwork) as MCP
tools and prompts:

  tools:    search_context, search_bibliography, list_sections, [search_documents]
  prompts:  fidaa_systemprompt, fidaa_starter_1 … fidaa_starter_8
  resources: fidaa://context/<H1>/<H2> — full chapter texts (R6.6)

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
from typing import Annotated, Any
from urllib.parse import quote

import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.responses import JSONResponse

# `import index` (module attribute access), NOT `from index import
# INDEX_READY`: build_indexes() *rebinds* INDEX_READY / INDEX_ERROR, and a
# from-import would be a stale snapshot taken at import time (a bool cannot
# be "mutated" across modules). INDEX (dict) would survive either way, but
# reading all three through the module keeps the rule uniform.
import index
from index import (
    CollectionSections,
    SectionListing,
    SearchResults,
    search_collection,
)
from splitting import (
    KNOWLEDGE_DIR,
    DOCUMENTS_PATH,
    _context_chapters,
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
            "Tools: search_context returns the 4 most relevant passages with "
            "their heading paths; search_bibliography returns exact source "
            "citations (author/year/title) for a specific work; "
            "search_documents (if present) searches an extra document archive. "
            "list_sections lists the chapter structure (heading paths) of a "
            "collection — call it first to target your searches. "
            "Prompts: fidaa_systemprompt carries FIDAA's behavior rules; "
            "fidaa_starter_* are ready-made question templates. "
            "Resources: fidaa://context/… serves full chapter texts for "
            "clients that read MCP resources. "
            "Formulate queries in German for best recall, and cite sources "
            "(heading path or bibliography entry) when answering with this knowledge."
        ),
        version="0.1.0",
        lifespan=_server_lifespan,
    )
    _register_tools(server)
    _register_prompts(server)
    _register_chapter_resources(server)
    _register_healthz(server)
    return server


def _register_tools(server: MCPServer) -> None:
    """Search tools + the list_sections TOC tool (R6.5). The optional
    search_documents (Q7) is registered only when the archive is enabled."""

    @server.tool()
    async def search_context(
        query: Annotated[str, Field(description="Die Suchanfrage für die interne Wissensdatenbank.")]
    ) -> SearchResults:
        """Durchsuche die interne Wissensdatenbank nach relevantem und geprüft richtigem Kontext."""
        return SearchResults(results=await search_collection("rag_context", query))

    @server.tool()
    async def search_bibliography(
        query: Annotated[
            str,
            Field(
                description="Titel, Autor:in oder Stichwort des Werks, dessen Quellenangabe gesucht wird."
            ),
        ]
    ) -> SearchResults:
        """Durchsuche die Bibliografie (knowledge/bibliography.md) nach der genauen Quellenangabe für ein bestimmtes Werk (Autor, Jahr, Titel)."""
        return SearchResults(results=await search_collection("rag_bibliography", query))

    # Optional third tool — registered only when DOCUMENTS_PATH is set (Q7).
    if DOCUMENTS_PATH and _load_documents() is not None:

        @server.tool()
        async def search_documents(
            query: Annotated[str, Field(description="Die Suchanfrage an das Dokumentenarchiv.")]
        ) -> SearchResults:
            """Durchsuche das Dokumentenarchiv nach relevantem zusätzlichen Kontext."""
            return SearchResults(results=await search_collection("rag_documents", query))

    # --- TOC tool (R6.5): chapter structure before searching -------------
    # Friendly labels → internal collection names. "documents" is optional
    # (Q7), so it is only listed when actually indexed.
    _COLLECTION_ALIASES = {
        "context": "rag_context",
        "bibliography": "rag_bibliography",
        "documents": "rag_documents",
    }

    @server.tool()
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
        """Liste die Kapitel- und Abschnittsstruktur der Wissensdatenbank als Überschriftenpfade auf — rufe sie auf, um Kapitel gezielt zu benennen, bevor search_context o. ä. gesucht wird."""
        if collection not in _COLLECTION_ALIASES and collection != "all":
            raise ValueError(
                f"Unbekannte Sammlung '{collection}' — erlaubt sind: "
                "all, context, bibliography, documents."
            )
        if not index.INDEX_READY:
            raise RuntimeError(
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
                    raise RuntimeError(
                        "Dokumentenarchiv ist nicht aktiv (DOCUMENTS_PATH "
                        "ist nicht gesetzt)."
                    )
                continue
            # Empty path = preamble before the first heading: no chapter.
            out.append(
                CollectionSections(
                    name=label,
                    headings=[p for p in _unique_paths(col.paths) if p],
                )
            )
        return SectionListing(collections=out)


def _register_prompts(server: MCPServer) -> None:
    """FIDAA's behavior rules + the 8 chat starters (R6.2)."""

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
    """Chapter resources (R6.6): readable chapter texts.

    One static resource per context chapter (factory pattern, like the
    starter prompts above). Static URIs — not a {heading} template —
    because heading paths contain '/' (two path segments); a template
    would depend on segment-capture semantics we don't want to rely on.
    Bibliography/documents stay search-only by design (R6.6 scope).
    """
    chapters = _context_chapters()
    for path, text in chapters:
        heading = " > ".join(path)
        server.resource(
            uri="fidaa://context/"
            + "/".join(quote(part, safe="") for part in path),
            name=heading,
            title=f"Kontext: {heading}",
            description=(
                "Volltext eines Kapitels der FIDAA-Kontextdatenbank "
                "(Markdown). Für Agenten, die MCP-Ressourcen lesen können."
            ),
            mime_type="text/markdown",
        )(_make_chapter_resource(text))
    logger.info("Registered %d chapter resources (fidaa://context/…)", len(chapters))


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
        """One of the 8 FIDAA chat starters (knowledge/prompts.md)."""
        return message

    return starter


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

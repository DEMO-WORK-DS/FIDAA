# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors — https://eupl.eu/1.2/en/
"""FIDAA MCP server.

Exposes the FIDAA knowledge base (knowledge/*.md, Digital Streetwork) as MCP
tools and prompts:

  tools:    search_context, search_bibliography, [search_documents]
  prompts:  fidaa_systemprompt, fidaa_starter_1 … fidaa_starter_8

Design (decisions Q1–Q8, see the FIDAA-DEMO repo's CURRENT_TASK / the
dev-wiki options analysis):
- v2 SDK (mcp>=2): MCPServer, stdio + streamable-http.
- In-memory numpy index, re-embedded on every startup — exact parity with
  the demo, which rebuilt its PGVector collection on every boot. No
  LangChain, no database, no credentials beyond the embedding endpoint.
- /healthz via custom_route (unauthenticated, for compose healthchecks);
  optional MCP_TOKEN bearer gate (default off); transport_security Host
  allowlist so the SDK's 421 guard works behind compose service names.
"""

from __future__ import annotations

import argparse
import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Any

import httpx2
import numpy as np
import uvicorn
from mcp.server import MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from starlette.responses import JSONResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("fidaa.server")

# ---------------------------------------------------------------------------
# Configuration — hard-fail on the two required secrets (same as the demo)
# ---------------------------------------------------------------------------
LLM_URL = os.environ["LLM_URL"]
LLM_KEY = os.environ["LLM_KEY"]
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-4B")
# Instruction-aware query embedding (demo parity: enabled by default).
USE_EMBED_INSTRUCTIONS = os.getenv("USE_EMBED_INSTRUCTIONS", "true").lower() in (
    "true",
    "1",
    "yes",
)
DEFAULT_TASK_INSTRUCTION = os.getenv(
    "DEFAULT_TASK_INSTRUCTION",
    "Given a web search query, retrieve relevant passages that answer the query",
)
# The knowledge dir ships inside the repo; overridable for unusual layouts.
KNOWLEDGE_DIR = Path(
    os.getenv("KNOWLEDGE_DIR", str(Path(__file__).resolve().parent.parent / "knowledge"))
)
# Optional third tool: a folder of .md files to index (demo parity: the
# tool only exists when this is set; a missing folder disables it w/ warning).
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
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
# 8192-token embedding limit, approximated in characters (≈3 chars/token
# for German). Chunks above this risk the historical HTTP 400, so we warn
# at startup instead of failing (R6.9).
MAX_CHUNK_CHARS = 8192 * 3
TOP_K = 4  # demo parity: as_retriever(k=4)
EMBED_BATCH = 32  # demo parity: OpenAIEmbeddings(chunk_size=32)


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


# ---------------------------------------------------------------------------
# In-memory index (Q4: rebuild at startup, numpy cosine — no LangChain/PG)
# ---------------------------------------------------------------------------
@dataclass
class Collection:
    """In-memory vector index for one knowledge source."""

    name: str
    chunks: list[str] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    matrix: np.ndarray | None = None  # (n_chunks, dim), L2-normalized rows


INDEX: dict[str, Collection] = {}
INDEX_READY = False
INDEX_ERROR: str | None = None


def _make_client() -> AsyncOpenAI:
    """OpenAI-compatible client for the self-hosted embedding endpoint
    (same timeout profile the demo uses for its OpenAI clients)."""
    return AsyncOpenAI(
        base_url=LLM_URL,
        api_key=LLM_KEY,
        timeout=httpx2.Timeout(120.0, connect=10.0),
    )


async def _embed_texts(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Embed in batches of EMBED_BATCH (demo parity: chunk_size=32)."""
    out: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = await client.embeddings.create(
            model=EMBEDDING_MODEL, input=texts[i : i + EMBED_BATCH]
        )
        out.extend(d.embedding for d in resp.data)
    return out


def _with_instruction(query: str) -> str:
    """Instruction-aware query prefix (demo parity: InstructionAwareEmbeddings).

    Documents are embedded without the prefix; only queries get it.
    """
    if USE_EMBED_INSTRUCTIONS:
        return f"Instruct: {DEFAULT_TASK_INSTRUCTION}\nQuery: {query}"
    return query


async def build_indexes() -> None:
    """Embed every knowledge chunk once at startup (server lifespan).

    The demo hard-failed startup when RAG building failed. Here we log the
    error, keep the server up (healthz reports it; tools raise a clear
    error) so the MCP surface stays inspectable even if the embedding API
    is unreachable.
    """
    global INDEX_READY, INDEX_ERROR
    client = _make_client()
    t0 = time.monotonic()
    try:
        jobs: list[tuple[str, list[tuple[list[str], str]]]] = [
            ("rag_context", _load_context()),
            ("rag_bibliography", _load_bibliography()),
        ]
        docs = _load_documents()
        if docs is not None:
            jobs.append(("rag_documents", docs))

        for name, pairs in jobs:
            chunks = [c for _, c in pairs]
            for c in chunks:
                if len(c) > MAX_CHUNK_CHARS:
                    # Startup guard (R6.9): flag chunks that risk the
                    # 8192-token embedding HTTP 400 instead of failing late.
                    logger.warning(
                        "chunk in %s exceeds ~%d chars — may hit the embedding "
                        "8192-token limit: %.60s…",
                        name,
                        MAX_CHUNK_CHARS,
                        c.replace("\n", " "),
                    )
            logger.info("Embedding %s: %d chunks…", name, len(chunks))
            vectors = await _embed_texts(client, chunks)
            mat = np.asarray(vectors, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)  # cosine via dot
            INDEX[name] = Collection(
                name=name, chunks=chunks, paths=[p for p, _ in pairs], matrix=mat
            )
        INDEX_READY = True
        logger.info(
            "Index ready in %.1f s: %s",
            time.monotonic() - t0,
            ", ".join(f"{n}={len(c.chunks)}" for n, c in INDEX.items()),
        )
    except Exception as e:  # noqa: BLE001 — keep the server up, report via healthz
        INDEX_ERROR = f"{type(e).__name__}: {e}"
        logger.exception("Index build failed — tools will error until restarted")


# ---------------------------------------------------------------------------
# Tool / prompt result models (structured output, R6.4: heading paths make
# sources citable by any agent)
# ---------------------------------------------------------------------------
class SearchResult(BaseModel):
    """Ein Treffer aus der FIDAA-Wissensdatenbank."""

    heading_path: list[str] = Field(
        description=(
            "Pfad der Markdown-Überschriften des Abschnitts (von H1 abwärts; "
            "im Dokumentenarchiv beginnt er mit dem Dateinamen)."
        )
    )
    text: str = Field(description="Volltext des Abschnitts inkl. Überschriftszeile.")


class SearchResults(BaseModel):
    """Suchergebnisse (top 4 nach semantischem Cosine-Score)."""

    results: list[SearchResult]


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
            "Prompts: fidaa_systemprompt carries FIDAA's behavior rules; "
            "fidaa_starter_* are ready-made question templates. "
            "Formulate queries in German for best recall, and cite sources "
            "(heading path or bibliography entry) when answering with this knowledge."
        ),
        version="0.1.0",
        lifespan=_server_lifespan,
    )

    _client: dict[str, AsyncOpenAI | None] = {"c": None}

    def _get_client() -> AsyncOpenAI:
        """Shared embedding client — built lazily, one per process."""
        if _client["c"] is None:
            _client["c"] = _make_client()
        return _client["c"]

    async def _search(name: str, query: str) -> list[SearchResult]:
        """Cosine top-k over one in-memory collection (demo parity: k=4)."""
        col = INDEX.get(name)
        if col is None or col.matrix is None:
            raise RuntimeError(
                f"Wissensdatenbank '{name}' ist nicht (noch) indexiert"
                + (f" — Fehler: {INDEX_ERROR}" if INDEX_ERROR else "")
                + ". Bitte später erneut versuchen."
            )
        (qvec,) = await _embed_texts(_get_client(), [_with_instruction(query)])
        q = np.asarray(qvec, dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = col.matrix @ q
        top = np.argsort(-scores)[:TOP_K]
        return [
            SearchResult(heading_path=col.paths[i], text=col.chunks[i]) for i in top
        ]

    @server.tool()
    async def search_context(
        query: Annotated[str, Field(description="Die Suchanfrage für die interne Wissensdatenbank.")]
    ) -> SearchResults:
        """Durchsuche die interne Wissensdatenbank nach relevantem und geprüft richtigem Kontext."""
        return SearchResults(results=await _search("rag_context", query))

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
        return SearchResults(results=await _search("rag_bibliography", query))

    # Optional third tool — registered only when DOCUMENTS_PATH is set (Q7).
    if DOCUMENTS_PATH and _load_documents() is not None:

        @server.tool()
        async def search_documents(
            query: Annotated[str, Field(description="Die Suchanfrage an das Dokumentenarchiv.")]
        ) -> SearchResults:
            """Durchsuche das Dokumentenarchiv nach relevantem zusätzlichen Kontext."""
            return SearchResults(results=await _search("rag_documents", query))

    # --- prompts (R6.2): FIDAA's behavior rules + the 8 chat starters -----
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

    # --- /healthz (R6.7): unauthenticated, for compose healthchecks -------
    @server.custom_route("/healthz", methods=["GET"])
    async def healthz(request) -> JSONResponse:  # noqa: ARG001 — Starlette signature
        """Liveness + index state. 200 only when the index is fully built;
        503 with the error otherwise, so compose `depends_on: healthy`
        gates the app on a working knowledge base."""
        payload: dict[str, Any] = {
            "status": "ok" if INDEX_READY else "degraded",
            "ready": INDEX_READY,
            "collections": {n: len(c.chunks) for n, c in INDEX.items()},
        }
        if INDEX_ERROR:
            payload["error"] = INDEX_ERROR
        return JSONResponse(payload, status_code=200 if INDEX_READY else 503)

    return server


def _make_starter(message: str):
    """Closure factory: gives each starter prompt its own function object,
    so the message is captured per iteration (and the prompt has no
    arguments — the SDK rejects defaulted parameters for prompts)."""

    def starter() -> str:
        """One of the 8 FIDAA chat starters (knowledge/prompts.md)."""
        return message

    return starter


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


@asynccontextmanager
async def _server_lifespan(server: MCPServer):  # noqa: ARG001 — SDK signature
    """Server lifespan (SDK constructor hook): the SDK calls
    ``lifespan(server)`` and ``async with`` the result — so this function
    (via @asynccontextmanager) *is* the context manager factory. Builds the
    in-memory index exactly once per process, before any transport serves a
    request."""
    await build_indexes()
    yield {"ready": INDEX_READY}


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

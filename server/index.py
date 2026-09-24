# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors — https://eupl.eu/1.2/en/
"""In-memory hybrid index (Q4: rebuild at startup, numpy cosine — no
LangChain, no Postgres, no credentials beyond the embedding endpoint).

Embeds every knowledge chunk once per process (server lifespan), fuses
the BM25 lexical ranking with the cosine ranking via RRF (R6.1), and
defines the structured result models shared by the search and TOC tools.
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field

import httpx2
import numpy as np
from openai import AsyncOpenAI
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from splitting import _load_bibliography, _load_context, _load_documents

logger = logging.getLogger("fidaa.server")

# Configuration — hard-fail on the two required secrets (same as the demo)
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
# 8192-token embedding limit, approximated in characters (≈3 chars/token
# for German). Chunks above this risk the historical HTTP 400, so we warn
# at startup instead of failing (R6.9).
MAX_CHUNK_CHARS = 8192 * 3
TOP_K = 4  # demo parity: as_retriever(k=4)
# RRF fusion constant (Cormack et al. 2009): fusion quality is flat for
# k in 40–60; 60 is the standard value (R6.1 hybrid search).
RRF_K = 60
EMBED_BATCH = 32  # demo parity: OpenAIEmbeddings(chunk_size=32)


# ---------------------------------------------------------------------------
# In-memory index
# ---------------------------------------------------------------------------
@dataclass
class Collection:
    """In-memory vector index for one knowledge source."""

    name: str
    chunks: list[str] = field(default_factory=list)
    paths: list[list[str]] = field(default_factory=list)
    matrix: np.ndarray | None = None  # (n_chunks, dim), L2-normalized rows
    # BM25 lexical index over the same chunks (R6.1, hybrid search).
    bm25: BM25Okapi | None = None


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


def _rrf_scores(scores: np.ndarray) -> np.ndarray:
    """Reciprocal rank of one ranking: 1/(k + rank), rank 1 = best score.

    Summing the RRF scores of the BM25 and the vector ranking fuses both
    without normalizing their (incompatible) raw score scales (R6.1)."""
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return 1.0 / (RRF_K + ranks)


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
            logger.info("Indexing %s: %d chunks (embeddings + BM25)…", name, len(chunks))
            vectors = await _embed_texts(client, chunks)
            mat = np.asarray(vectors, dtype=np.float32)
            mat /= np.linalg.norm(mat, axis=1, keepdims=True)  # cosine via dot
            # BM25 over the same chunks (R6.1): plain word tokens, no
            # stemmer; built once per startup, query cost is negligible.
            bm25 = BM25Okapi([re.findall(r"\w+", c.lower()) for c in chunks])
            INDEX[name] = Collection(
                name=name,
                chunks=chunks,
                paths=[p for p, _ in pairs],
                matrix=mat,
                bm25=bm25,
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
# Tool result models (structured output, R6.4: heading paths make
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
    """Suchergebnisse (top 4, hybride Sortierung: BM25 + Cosine via RRF)."""

    results: list[SearchResult]


class CollectionSections(BaseModel):
    """Abschnittsstruktur einer Sammlung (R6.5)."""

    name: str = Field(
        description="Sammlung: 'context', 'bibliography' oder 'documents'."
    )
    headings: list[list[str]] = Field(
        description=(
            "Überschriftenpfade in Dokumentenreihenfolge (von H1 abwärts; "
            "im Dokumentenarchiv beginnt der Pfad mit dem Dateinamen), "
            "ohne Wiederholungen."
        )
    )


class SectionListing(BaseModel):
    """Inhaltsverzeichnis der FIDAA-Wissensdatenbank (zum gezielten Suchen)."""

    collections: list[CollectionSections]


# ---------------------------------------------------------------------------
# Hybrid search (R6.1): BM25 + cosine via RRF over one collection
# ---------------------------------------------------------------------------
_client: dict[str, AsyncOpenAI | None] = {"c": None}


def _get_client() -> AsyncOpenAI:
    """Shared embedding client — built lazily, one per process."""
    if _client["c"] is None:
        _client["c"] = _make_client()
    return _client["c"]


async def search_collection(name: str, query: str) -> list[SearchResult]:
    """Hybrid top-k (BM25 + cosine via RRF) over one in-memory
    collection (demo parity: k=4, result format unchanged)."""
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
    # Hybrid ranking (R6.1): fuse the lexical BM25 ranking with the
    # cosine ranking via RRF — BM25 wins on exact keyword/citation
    # hits, the vectors on semantic queries. Falls back to pure
    # vector order when BM25 matches nothing at all.
    if col.bm25 is not None:
        bm25_scores = np.asarray(
            col.bm25.get_scores(re.findall(r"\w+", query.lower())),
            dtype=np.float64,
        )
        if bm25_scores.max() > 0:
            scores = _rrf_scores(bm25_scores) + _rrf_scores(
                scores.astype(np.float64)
            )
    top = np.argsort(-scores)[:TOP_K]
    return [
        SearchResult(heading_path=col.paths[i], text=col.chunks[i]) for i in top
    ]

---
name: fidaa
description: >-
  FIDAA knowledge base and semantic search for Digital Streetwork (digitale
  aufsuchende soziale Arbeit; German). Use when answering questions about
  digital street work, online outreach to young people,
  Radikalisierungsprävention, digital empowerment or networking work; when
  working with FIDAA / DEMO-WORK project material; or when the FIDAA MCP
  server is connected (tools: search_context, search, fetch, list_sections,
  search_bibliography, read_knowledge_file, search_documents; prompts:
  fidaa_systemprompt, fidaa_starter_1..9; resources: fidaa://context/…,
  fidaa://knowledge/…).
license: EUPL-1.2-only
---

# FIDAA

FIDAA = *Fachinformation Digitale Aufsuchende Arbeit*: a curated,
expert-checked knowledge base on **Digital Streetwork** (German).

## Mode 1 — FIDAA MCP server available (preferred)

If an MCP server named `fidaa` is connected, use its tools instead of
reading files. No server running? Connect to the hosted endpoint
`https://fidaa.h2.de/mcp` (streamable-HTTP, open — see README,
"Nutzung durch einen Agenten → Gehosteter Endpunkt") instead of
self-hosting.

1. `search_context(query, top_k?)` — the most relevant **complete
   chapters** of the knowledge database (default 4; heading path, chapter
   ID, text).
2. `search(query)` → `fetch(id)` — find chapter IDs/titles/URLs, then read
   one chapter complete by ID (also the ChatGPT deep-research interface).
3. `list_sections(collection?)` — the chapter structure with chapter IDs
   (`all`/`context`/`bibliography`/`documents`); call it first when you
   want to target a chapter, then `fetch(id)` it.
4. `search_bibliography(query, top_k?)` — the exact citation
   (author/year/title) of a specific work.
5. `read_knowledge_file(file?, part?)` — read a whole knowledge file
   (`kontext.md` ≈ 150–175k tokens) in parts split only at chapter
   boundaries; read all parts `1…total_parts` to ingest the complete
   knowledge base when your context window allows it.
6. `search_documents(query)` — only present when the server additionally
   indexes a document archive (`DOCUMENTS_PATH`).

If your client reads MCP **resources**: complete chapters are
`fidaa://context/<H1>/<H2>`, complete files `fidaa://knowledge/<file>`.

**Never use part of a chapter.** Every `##` chapter is a complete,
human-edited unit; all access paths deliver whole chapters — keep them
whole when you quote or summarize, so no qualification gets lost.

Rules (carried verbatim in the `fidaa_systemprompt` prompt):

- Formulate queries in **German** (the knowledge base is in German).
- Always search before answering; answer strictly on the basis of the
  retrieved context — no inventing, no off-source advice.
- Cite sources: heading path for context, bibliography entry for works.
- FIDAA gives orientation, **not counseling** (no psychological or legal
  advice) and never communicates with clients.

`fidaa_starter_1` … `fidaa_starter_9` are ready-made question templates.

## Mode 2 — no MCP server (read the files directly)

Start with `llms.txt` (repo root) or `knowledge/INDEX.md`. Then choose:

- **Whole files** (most complete — use it if your context window allows):
  `knowledge/kontext.md` ≈ 525k characters ≈ 150–175k tokens;
  `bibliography.md` ≈ 29k tokens; all files ≈ 205k tokens.
- **Chapter by chapter**: `knowledge/INDEX.md` lists every chapter of
  `kontext.md` with ID, **line range**, size and subsections — read the
  matching chapters completely by line range (always the whole range).

| File | Read it when |
|---|---|
| `knowledge/INDEX.md` | first: which chapters exist, where, how big (generated) |
| `knowledge/welcome.md` | quick orientation (small) |
| `knowledge/systemprompt.md` | you should behave *like* FIDAA (behavior rules) |
| `knowledge/prompts.md` | typical questions (the 9 starters) |
| `knowledge/kontext.md` | content questions — the knowledge database |
| `knowledge/bibliography.md` | you need exact sources / citations |

`bibliography.md` is grouped by thematic field (`# 1. Migration, …` …
`# 6. Übergreifende Grundlagen, …`; line ranges in `INDEX.md`). Each
reference entry is a `* **Author (Year)**: …` bullet with a 2–3 sentence
abstract and (where present) a URL.

Without the repo, the same files are plain HTTP downloads:
`https://fidaa.h2.de/mcp/llms.txt`, `https://fidaa.h2.de/mcp/knowledge/<file>`,
`https://fidaa.h2.de/mcp/kontext/<ID>.md`.

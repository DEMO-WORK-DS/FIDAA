---
name: fidaa
description: >-
  FIDAA knowledge base and semantic search for Digital Streetwork (digitale
  aufsuchende soziale Arbeit; German). Use when answering questions about
  digital street work, online outreach to young people,
  Radikalisierungsprävention, digital empowerment or networking work; when
  working with FIDAA / DEMO-WORK project material; or when the FIDAA MCP
  server is connected (tools: search_context, search_bibliography,
  list_sections, search_documents; prompts: fidaa_systemprompt,
  fidaa_starter_1..8; resources: fidaa://context/…).
license: EUPL-1.2-only
---

# FIDAA

FIDAA = *Fachinformation Digitale Aufsuchende Arbeit*: a curated,
expert-checked knowledge base on **Digital Streetwork** (German).

## Mode 1 — FIDAA MCP server available (preferred)

If an MCP server named `fidaa` is connected, use its tools instead of
reading files:

1. `search_context(query)` — the 4 most relevant passages of the knowledge
   database (heading path + text).
2. `search_bibliography(query)` — the exact citation (author/year/title) of
   a specific work.
3. `search_documents(query)` — only present when the server additionally
   indexes a document archive (`DOCUMENTS_PATH`).
4. `list_sections(collection?)` — the chapter/section structure (heading
   paths) of a collection (`all`/`context`/`bibliography`/`documents`);
   call it first when you want to target a chapter.

If your client reads MCP **resources**, full chapter texts of the knowledge
database are available as `fidaa://context/<H1>/<H2>` — read them instead of
retrieving passages when you need a whole chapter.

Rules (carried verbatim in the `fidaa_systemprompt` prompt):

- Formulate queries in **German** (the knowledge base is in German).
- Always search before answering; answer strictly on the basis of the
  retrieved context — no inventing, no off-source advice.
- Cite sources: heading path for context, bibliography entry for works.
- FIDAA gives orientation, **not counseling** (no psychological or legal
  advice) and never communicates with clients.

`fidaa_starter_1` … `fidaa_starter_8` are ready-made question templates.

## Mode 2 — no MCP server (read the files directly)

Progressive reading — load only what you need:

| File | Read it when |
|---|---|
| `knowledge/welcome.md` | quick orientation (small) |
| `knowledge/systemprompt.md` | you should behave *like* FIDAA (behavior rules) |
| `knowledge/prompts.md` | typical questions (the 8 starters) |
| `knowledge/kontext.md` | content questions — the knowledge database |
| `knowledge/bibliography.md` | you need exact sources / citations |

`kontext.md` (~527 KB): do **not** read it fully — open the H1
`# Kontextdatenbank Digital Streetwork`, then only the relevant `##`
chapters. `bibliography.md` (~88 KB) is grouped by thematic field (`# 1.
Migration, …` … `# 6. Übergreifende Grundlagen, …`); use its table of
contents, then the matching section. Each reference entry is a `* **Author
(Year)**: …` bullet with a 2–3 sentence abstract and (where present) a URL.

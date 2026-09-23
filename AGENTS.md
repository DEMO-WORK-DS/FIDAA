# AGENTS.md

Instructions for AI coding agents working in this repository.

## Project

FIDAA (*Fachinformation Digitale Aufsuchende Arbeit*): the knowledge base of
the DEMO-WORK project on **Digital Streetwork** (aufsuchende soziale Arbeit
in digitalen Räumen) in `knowledge/*.md` (German, CC-BY-SA-4.0), plus a thin MCP
server (`server/main.py`, ~700 LOC) that exposes it as tools
(`search_context`, `search_bibliography`, `list_sections`, optional
`search_documents`), prompts (FIDAA system prompt + the 8 chat starters)
and resources (full context-chapter texts, `fidaa://context/…`). Any AI agent can use
this knowledge via the MCP server (stdio or streamable-http) or by reading
`knowledge/` directly (see `skills/fidaa/SKILL.md` for the agent-facing
usage guide).

## Commands

- Dependencies: `uv sync` (the repo ships a `.venv`)
- Local secrets: `cp secrets.env.example secrets.env`, fill it in
- Start (stdio, for agents): `uv run server/main.py`
- Start (HTTP, for docker compose): `uv run server/main.py --transport streamable-http`
- MCP Inspector: `uv run mcp dev server/main.py` (dev-group extra `mcp[cli]`;
  the inspector UI needs Node)
- Health (HTTP mode): `curl localhost:8002/healthz`
- No test suite exists. Verify changes by starting the server and checking
  the startup log (index build line) plus a manual tool call (Inspector or
  any MCP client).

## Conventions

- `secrets.env` (gitignored) is the single source of truth for secrets
  (`LLM_URL`, `LLM_KEY`, `EMBEDDING_MODEL`, `MCP_TOKEN`, …).
- The embedding model has an 8192-token context window: markdown sections
  longer than that make the embeddings call fail with HTTP 400. The server
  warns at startup for chunks above `MAX_CHUNK_CHARS` (~24k chars). When
  restructuring `knowledge/*.md`, keep chapters under that limit.
- `_split_markdown` in `server/main.py` is an **exact port** of the
  `MarkdownHeaderTextSplitter` semantics (langchain-text-splitters 1.1.2,
  the version the pre-M2 demo ran; the port was made by reading that
  source). FIDAA-DEMO's M2 H2 starter splitter is a second, independent
  port of the same behavior. The two are verified byte-identical — if one
  changes, change both and re-verify the chunk outputs.
- The server rebuilds its in-memory index on every startup (parity with the
  demo, which re-embeds into PGVector on every boot). Do **not** add
  persistence to the server.
- `uv.lock` is checked into git; the Dockerfile's uv image tag must match
  the local uv that generated the lock (lockfile format).
- License split: knowledge content is **CC-BY-SA-4.0** (root `LICENSE`,
  per-file SPDX headers in `knowledge/`); code files are
  **EUPL-1.2-only** (per-file SPDX headers). Do not mix the two.

## Change workflow (required)

- For major changes, always propose a structured plan to the user (maintainer).
  Explain what changes should be made, what their purpose is and how it will
  be kept minimal, to avoid an explosion in complexity.
- Before implementing big parts of new functionality, check online whether
  well-known, widely used libraries already fulfill that purpose. Evaluate
  them against each other and against custom code, present the finding, and
  support the maintainer in making an informed decision before proceeding.
- For every code change, add a short comment at the changed location that
  documents its purpose (what it does and why).
- Show the diff to the maintainer and wait for review. Never `git push` to
  `origin` before the maintainer has approved the change.

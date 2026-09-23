# FIDAA — Fachinformation Digitale Aufsuchende Arbeit

The knowledge base of the [DEMO-WORK](https://demo-work.h2.de) project on
**Digital Streetwork** (aufsuchende soziale Arbeit in digitalen Räumen),
packaged so that **any AI agent** can use it: as an [MCP](https://modelcontextprotocol.io)
server (tools + prompts), or by simply reading `knowledge/`.

- `knowledge/` — the curated, expert-checked knowledge base (German,
  CC BY-SA 4.0):
  `kontext.md` (context database), `bibliography.md` (140+ cited sources),
  `systemprompt.md`, `prompts.md` (8 chat starters), `welcome.md`.
- `server/main.py` — thin MCP server on the official `mcp` SDK v2
  (`MCPServer`), stdio + streamable-http, in-memory hybrid index (numpy
  cosine + BM25, RRF fusion) rebuilt on startup (no database, no LangChain).
- `skills/fidaa/SKILL.md` — [Agent Skill](https://agentskills.io): usage
  guide for coding agents that open this repo directly.

## Project & Team

FIDAA is part of **DEMO-WORK**, a research and transfer project funded
by the **VolkswagenStiftung** ("Transformationswissen über Demokratien
im Wandel – transdisziplinäre Perspektiven"), carried out in cooperation
between the **Hochschule Magdeburg-Stendal (h2)**, the **Amadeu Antonio
Stiftung** and the **Katholische Hochschule Nordrhein-Westfalen
(katho)**, and affiliated with the Institut für demokratische Kultur (IdK)
at h2.

The team unites scientific and practical expertise in social work,
sociology of technology, and computer science / AI research:

**Wissenschaft (research)**

- **Prof. Dr. Nele Wulf** (katho Aachen) — professor for digitalization
  and social work; leads the consolidation of scientific and practical
  findings; responsible for the concept and evaluation of FIDAA.
- **Christina Dinar** (h2; PhD at KHSB Berlin / TU Berlin) — research
  associate; co-developed the "Digital Streetwork" concept; doctoral
  research on digital streetwork as outreach work in social media.
- **David Döring** (h2) — research associate (computer science);
  technical implementation of the knowledge-based system; technical
  advisory for the team.
- **Nils Fietkau** (h2) — research associate (social work, health,
  media); co-responsible for the empirical research on professional
  digital streetwork practice.

**Praxis (practice)**

- **Cornelia Heyken** (Amadeu Antonio Stiftung) — co-developer of the
  "Digital Streetwork" concept; analyzes the academic and formalized
  knowledge base on digital streetwork and prepares it for the AI
  prototype.
- **Jerome Trebing** (Amadeu Antonio Stiftung) — violence-prevention
  practitioner; ethnographic observation and analysis of
  digital-streetwork practice; development of professional standards.

Project information, full team bios and contact (imprint):
<https://demo-work.h2.de> (English: <https://demo-work.h2.de/en/>)

## What the server offers

| MCP surface | Name | What it does |
|---|---|---|
| tool | `search_context(query)` | 4 most relevant passages, with heading paths |
| tool | `search_bibliography(query)` | exact source citations (author/year/title) |
| tool | `search_documents(query)` | optional — only if `DOCUMENTS_PATH` is set |
| tool | `list_sections(collection?)` | chapter structure (heading paths), per collection |
| prompt | `fidaa_systemprompt` | FIDAA's behavior rules (system prompt) |
| prompt | `fidaa_starter_1…8` | the 8 chat starters, ready to send |
| resource | `fidaa://context/<H1>/<H2>` | full chapter texts (context only) |
| route | `GET /healthz` | index state (HTTP mode only) |

Queries in **German** give the best recall.

## Quickstart (laptop)

```bash
git clone https://github.com/DEMO-WORK-DS/FIDAA.git && cd FIDAA
cp secrets.env.example secrets.env   # fill LLM_URL / LLM_KEY / EMBEDDING_MODEL
uv sync
uv run server/main.py                              # stdio (for an agent)
uv run server/main.py --transport streamable-http  # HTTP on 0.0.0.0:8002
```

Try it in the MCP Inspector: `uv run mcp dev server/main.py`.

## Using it from an agent

**Claude Code / Codex** (`.mcp.json`, project dir — stdio):

```json
{
  "mcpServers": {
    "fidaa": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/FIDAA", "server/main.py"],
      "env": {
        "LLM_URL": "https://ai.h2.de/llm/v1",
        "LLM_KEY": "<key>",
        "EMBEDDING_MODEL": "Qwen3-Embedding-4B"
      }
    }
  }
}
```

**OpenCode** (opencode.json):

```json
{
  "mcp": {
    "fidaa": {
      "type": "local",
      "command": ["uv", "run", "--project", "/path/to/FIDAA", "server/main.py"],
      "environment": {
        "LLM_URL": "https://ai.h2.de/llm/v1",
        "LLM_KEY": "<key>",
        "EMBEDDING_MODEL": "Qwen3-Embedding-4B"
      },
      "enabled": true
    }
  }
}
```

**Claude Desktop**: same shape as `.mcp.json`, in
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).

**Running over HTTP** (e.g. from the FIDAA-DEMO compose network):
point any streamable-http client at `http://<host>:8002/mcp`.

## Exposing the HTTP endpoint publicly (roadmap)

v1 ships **internal-only** (compose network, no token). To expose it:

1. Set `MCP_TOKEN` (strong random value) in the environment — the `/mcp`
   route then requires `Authorization: Bearer <token>`; `/healthz` stays open.
2. Add the public hostname to `MCP_ALLOWED_HOSTS` (the SDK's 421 guard).
3. Route it in Caddy and add rate limiting in front; note that every call
   costs embedding-API usage, and a shared token is only a weak gate —
   the spec-compliant upgrade is OAuth 2.1 (MCP authorization spec).
4. Publish to the [MCP registry](https://modelcontextprotocol.io/registry)
   so agents researching digital streetwork can discover it.

## License

Split by content type: the **knowledge content** (`knowledge/*.md`) is
**CC BY-SA 4.0** (root `LICENSE`; each file also carries a
`SPDX-License-Identifier: CC-BY-SA-4.0` header). The **code**
(`server/`, `Dockerfile`, `pyproject.toml`, …) is **EUPL-1.2-only**
(per-file SPDX headers).

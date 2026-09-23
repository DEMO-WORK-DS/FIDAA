# FIDAA — Fachinformation Digitale Aufsuchende Arbeit

The knowledge base of the [DEMO-WORK](https://h2.de/) project on
**Digital Streetwork** (aufsuchende soziale Arbeit in digitalen Räumen),
packaged so that **any AI agent** can use it: as an [MCP](https://modelcontextprotocol.io)
server (tools + prompts), or by simply reading `knowledge/`.

- `knowledge/` — the curated, expert-checked knowledge base (German, EUPL-1.2):
  `kontext.md` (context database), `bibliography.md` (140+ cited sources),
  `systemprompt.md`, `prompts.md` (8 chat starters), `welcome.md`.
- `server/main.py` — thin MCP server on the official `mcp` SDK v2
  (`MCPServer`), stdio + streamable-http, in-memory numpy index rebuilt on
  startup (no database, no LangChain).
- `skills/fidaa/SKILL.md` — [Agent Skill](https://agentskills.io): usage
  guide for coding agents that open this repo directly.

## What the server offers

| MCP surface | Name | What it does |
|---|---|---|
| tool | `search_context(query)` | 4 most relevant passages, with heading paths |
| tool | `search_bibliography(query)` | exact source citations (author/year/title) |
| tool | `search_documents(query)` | optional — only if `DOCUMENTS_PATH` is set |
| prompt | `fidaa_systemprompt` | FIDAA's behavior rules (system prompt) |
| prompt | `fidaa_starter_1…8` | the 8 chat starters, ready to send |
| route | `GET /healthz` | index state (HTTP mode only) |

Queries in **German** give the best recall.

## Quickstart (laptop)

```bash
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

Code and knowledge base: EUPL-1.2 (`LICENSE`). `knowledge/prompts.md`
carries its own CC-BY-SA-4.0 header.

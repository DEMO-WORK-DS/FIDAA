<p align="center">
  <img src="assets/logo.svg" width="128" alt="FIDAA logo" />
</p>

# FIDAA — Fachinformation Digitale Aufsuchende Arbeit

[**Click here, to read this page in English**](README.en.md)

Dieses Projekt enthält die Wissensdatenbank des [DEMO-WORK](https://demo-work.h2.de)-Projekts zu
**Digital Streetwork** (aufsuchende soziale Arbeit in digitalen Räumen),
verpackt so, dass **jeder KI-Agent** sie nutzen kann: als
[MCP](https://modelcontextprotocol.io)-Server (Tools + Prompts) oder
einfach, indem der Agent `knowledge/` liest.

- `knowledge/` — die kuratierte, fachlich von Expert\*innen geprüfte Wissensdatenbank
  (Deutsch, CC BY-SA 4.0):
  - `kontext.md` (Kontextdatenbank)
  - `bibliography.md` (140+ Referenzen zu externen Quellen)
  - `systemprompt.md` (Geprüfte Anweisungen für einen KI-Agenten)  
  - `prompts.md` (9 Chat-Startervorlagen)
  - `welcome.md` (Kurze Projektübersicht)
- `server/main.py` — schlanker MCP-Server auf dem offiziellen `mcp`-SDK
  v2 (`MCPServer`), stdio + streamable-http, In-Memory-Hybridindex
  (numpy-Cosine + BM25, RRF-Fusion), bei jedem Start neu aufgebaut (keine
  Datenbank, kein LangChain).
- `skills/fidaa/SKILL.md` — [Agent Skill](https://agentskills.io):
  Nutzungsanleitung für Coding-Agenten, die dieses Repo direkt öffnen.

## Projekt & Team

FIDAA ist Teil von [**DEMO-WORK**](https://demo-work.h2.de/), einem von der [**VolkswagenStiftung**](https://www.volkswagenstiftung.de/)
geförderten Forschungs- und Transferprojekt ([„Transformationswissen über
Demokratien im Wandel – transdisziplinäre Perspektiven"](https://www.volkswagenstiftung.de/de/news/aktuelles/demokratiewandel-im-fokus-neun-neue-forschungsprojekte-erhalten-foerderung)), durchgeführt in
Kooperation zwischen der [**Hochschule Magdeburg-Stendal (h2)**](https://www.h2.de/), der
[**Amadeu Antonio Stiftung**](https://www.amadeu-antonio-stiftung.de/) und der [**Katholischen Hochschule
Nordrhein-Westfalen (katho)**](https://katho-nrw.de).

Das Team vereint wissenschaftliche und praktische Expertise aus Sozialer
Arbeit, Techniksoziologie sowie Informatik / KI-Forschung:

**Wissenschaft**

- **Prof. Dr. Nele Wulf** (katho Aachen) — Professorin für Digitalisierung
  und Soziale Arbeit; leitet die Zusammenführung wissenschaftlicher und
  praktischer Befunde; verantwortlich für Konzept und Evaluation von FIDAA.
- **Christina Dinar** (Amadeu Antonio Stiftung; Promotion an der KHSB
  Berlin / TU Berlin) — wissenschaftliche Mitarbeitende; Mitentwicklerin
  des Konzepts „Digital Streetwork"; Dissertation über digitalen Streetwork
  als Outreach-Arbeit in sozialen Medien.
- **David Döring** (h2) — wissenschaftlicher Mitarbeiter (Informatik);
  technische Umsetzung des wissensbasierten Systems; technisches Advisory
  für das Team.
- **Nils Fietkau** (h2) — wissenschaftlicher Mitarbeiter (Soziale Arbeit,
  Gesundheit, Medien); mitverantwortlich für die empirische Forschung zur
  professionellen Praxis digitalen Streetworks.

**Praxis**

- **Cornelia Heyken** (Amadeu Antonio Stiftung) — Mitentwicklerin des
  Konzepts „Digital Streetwork"; analysiert die wissenschaftliche und
  formalisierte Wissensgrundlage des digitalen Streetwork und bereitet sie
  für den KI-Prototyp auf.
- **Jerome Trebing** (Amadeu Antonio Stiftung) — Praktiker der
  Gewaltprävention; ethnografische Beobachtung und Analyse von
  Streetwork-Praxis im Digitalen; Entwicklung von Berufsstandards.

Projektinformationen, vollständige Team-Profile und Kontakt (Impressum):
<https://demo-work.h2.de> (Englisch: <https://demo-work.h2.de/en/>)

## Was der MCP-Server bietet

| MCP-Oberfläche | Name                         | Funktion                                             |
| -------------- | ---------------------------- | ---------------------------------------------------- |
| tool           | `search_context(query)`      | die 4 relevantesten Passagen, mit Überschriftspfaden |
| tool           | `search_bibliography(query)` | exakte Quellenangaben (Autor:in/Jahr/Titel)          |
| tool           | `search_documents(query)`    | optional — nur wenn `DOCUMENTS_PATH` gesetzt ist     |
| tool           | `list_sections(collection?)` | Kapitelstruktur (Überschriftspfade), pro Sammlung    |
| prompt         | `fidaa_systemprompt`         | FIDAAs Verhaltensregeln (System-Prompt)              |
| prompt         | `fidaa_starter_1…9`          | die 9 Chat-Startervorlagen, versandbereit            |
| resource       | `fidaa://context/<H1>/<H2>`  | vollständige Kapiteltexte (nur Kontext)              |
| route          | `GET /healthz`               | Index-Status (nur HTTP-Modus)                        |

Abfragen auf **Deutsch** liefern die besten Treffer.

## Schnellstart (Laptop)

```bash
git clone https://github.com/DEMO-WORK-DS/FIDAA.git && cd FIDAA
cp secrets.env.example secrets.env   # fill LLM_URL / LLM_KEY / EMBEDDING_MODEL
uv sync
uv run server/main.py                              # stdio (for an agent)
uv run server/main.py --transport streamable-http  # HTTP on 0.0.0.0:8002
```

Im MCP Inspector testen: `uv run mcp dev server/main.py`.

## Nutzung durch einen Agenten

### Gehosteter Endpunkt (ohne Self-Hosting)

Der [FIDAA-DEMO](https://github.com/DEMO-WORK-DS/FIDAA-DEMO)-Betrieb stellt
den Server unter **`https://fidaa.h2.de/mcp`** bereit (streamable-HTTP,
über Caddy geroutet). Er ist **offen — ohne Authentifizierung**. Keine
eigenen LLM-Keys nötig: die Embeddings laufen auf der Hosting-Seite.

**Claude Code / Codex** (`.mcp.json`, Projektverzeichnis):

```json
{
  "mcpServers": {
    "fidaa": {
      "url": "https://fidaa.h2.de/mcp"
    }
  }
}
```

**OpenCode** (`opencode.json`):

```json
{
  "mcp": {
    "fidaa": {
      "type": "remote",
      "url": "https://fidaa.h2.de/mcp",
      "enabled": true
    }
  }
}
```

**Claude Desktop**: gleiche `mcpServers`-Struktur, in
`~/Library/Application Support/Claude/claude_desktop_config.json`
(macOS).

### Selbst gehostet (stdio — volle Kontrolle, eigene LLM-Keys)

**Claude Code / Codex** (`.mcp.json`, Projektverzeichnis — stdio):

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

**OpenCode** (`opencode.json`):

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

**Claude Desktop**: wie `.mcp.json`, in
`~/Library/Application Support/Claude/claude_desktop_config.json` (macOS).

**Per HTTP** (z. B. aus dem FIDAA-DEMO-Compose-Netzwerk): jeden
streamable-http-Client auf `http://<host>:8002/mcp` zeigen lassen.

## Öffentlicher Endpunkt

Der FIDAA-DEMO-Betrieb stellt den HTTP-Endpunkt unter
**`https://fidaa.h2.de/mcp`** bereit (Caddy-Route; siehe „Gehosteter
Endpunkt" oben). Er ist bewusst **offen** (ohne Authentifizierung), um ihn
so zugänglich wie möglich zu halten.

Damit kann FIDAA als Werkzeug ganz einfach durch angabe des Links in andere KI-Agenten integriert werden.

## Lizenz

Aufteilung nach Inhaltstyp: Die **Wissensinhalte** (`knowledge/*.md`) sind
**CC BY-SA 4.0** (root `LICENSE`; jede Datei zusätzlich mit
`SPDX-License-Identifier: CC-BY-SA-4.0`-Header). Der **Code** (`server/`,
`Dockerfile`, `pyproject.toml`, …) ist **EUPL-1.2-only**
(per-Datei-SPDX-Header).

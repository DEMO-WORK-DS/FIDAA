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
  - `INDEX.md` (generierte Kapitelübersicht: ID, Zeilenbereich und Größe
    jedes Kapitels — erzeugt mit `uv run scripts/build_index.py`)
- `llms.txt` — Einstiegspunkt für KI-Agenten ([llms.txt](https://llmstxt.org)):
  welche Datei wofür, und die drei Zugangswege (ganze Dateien, einzelne
  Kapitel, Suche per MCP).
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

| MCP-Oberfläche | Name                                  | Funktion                                                       |
| -------------- | ------------------------------------- | -------------------------------------------------------------- |
| tool           | `search_context(query, top_k?)`       | die relevantesten vollständigen Kapitel (Standard 4), mit Überschriftspfad und Kapitel-ID |
| tool           | `search(query)` + `fetch(id)`         | Kapitel-IDs/Titel/URLs finden, dann ein Kapitel vollständig lesen (kompatibel mit ChatGPT Deep Research) |
| tool           | `list_sections(collection?)`          | Kapitelstruktur (Überschriftspfade + Kapitel-IDs), pro Sammlung |
| tool           | `search_bibliography(query, top_k?)`  | exakte Quellenangaben (Autor:in/Jahr/Titel)                    |
| tool           | `read_knowledge_file(file?, part?)`   | eine ganze Wissensdatei lesen (in Teilen, nur an Kapitelgrenzen getrennt) |
| tool           | `search_documents(query, top_k?)`     | optional — nur wenn `DOCUMENTS_PATH` gesetzt ist               |
| prompt         | `fidaa_systemprompt`                  | FIDAAs Verhaltensregeln (System-Prompt)                        |
| prompt         | `fidaa_starter_1…9`                   | die 9 Chat-Startervorlagen, versandbereit                      |
| resource       | `fidaa://context/<H1>/<H2>`           | vollständige Kapiteltexte (nur Kontext)                        |
| resource       | `fidaa://knowledge/<datei>`           | vollständige Wissensdateien                                    |
| route          | `GET /mcp/llms.txt`, `/mcp/knowledge/<datei>`, `/mcp/kontext/<id>.md` | dieselben Inhalte als einfache HTTP-Dateien (nur HTTP-Modus) |
| route          | `GET /healthz`                        | Index-Status (nur HTTP-Modus)                                  |

Alle Tools sind als nur lesend markiert (`readOnlyHint`). **Ein Kapitel
(`##`-Abschnitt) wird nie zerteilt:** Jeder Weg — Suchtreffer, `fetch`,
Ressource, Dateiteil, HTTP — liefert vollständige Kapitel, weil jedes
Kapitel eine von Menschen redigierte, in sich vollständige Einheit ist.

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

## Direkter Zugriff ohne MCP / ganze Wissensbasis aufnehmen

Agenten müssen nicht suchen: Wer ein ausreichend großes Kontextfenster hat,
kann die **gesamte Wissensbasis** aufnehmen — `kontext.md` hat ca. 525.000
Zeichen (≈ 150–175k Tokens), alle Dateien zusammen ≈ 205k Tokens. Das
vermeidet Lücken durch verpasste Suchtreffer. Wege:

- im Repo: `knowledge/kontext.md` lesen (Kapitelübersicht vorher in
  `knowledge/INDEX.md`, Einstieg über `llms.txt`);
- per HTTP (ohne MCP-Client): `https://fidaa.h2.de/mcp/llms.txt`,
  `https://fidaa.h2.de/mcp/knowledge/kontext.md`, einzelne Kapitel unter
  `https://fidaa.h2.de/mcp/kontext/<ID>.md` (IDs in `INDEX.md`);
- per MCP: Ressource `fidaa://knowledge/kontext.md` oder Tool
  `read_knowledge_file` (liefert große Dateien in Teilen ≤ ~60k Zeichen,
  damit Client-Limits für Tool-Ausgaben nicht greifen).

Die Dateien werden vom Server zur Laufzeit aus `knowledge/` ausgeliefert,
nicht doppelt im Repo abgelegt. Die Kapitel-URLs nutzt der Server auch als
zitierbare Quellen-URL in `search`/`fetch`; für eine eigene Instanz mit
geändertem Wissen `PUBLIC_URL` setzen (Standard: `https://fidaa.h2.de/mcp`).

**Dieses Repo in einem Agenten öffnen:** Die `.mcp.json` im Wurzelverzeichnis
verbindet Claude Code (und kompatible Clients) automatisch mit dem
gehosteten Endpunkt.

## Lizenz

Aufteilung nach Inhaltstyp: Die **Wissensinhalte** (`knowledge/*.md`) sind
**CC BY-SA 4.0** (root `LICENSE`; jede Datei zusätzlich mit
`SPDX-License-Identifier: CC-BY-SA-4.0`-Header). Der **Code** (`server/`,
`Dockerfile`, `pyproject.toml`, …) ist **EUPL-1.2-only**
(per-Datei-SPDX-Header).

# FIDAA -- **F**ach**i**nformation **D**igitale **A**ufsuchende **A**rbeit

Chainlit RAG Agent für einfaches Information Retrieval im Chat-Format aus einer vorgegebenen, geprüften Wissensadatenbank.
Verwendet OpenAI-kompatibles Backend für Chat-Streaming (mit Thinking & Tooluse) sowie Embedding.
Langchain RAG wird bei Start aus Dokumenten geladen und in via pgvector in Postgresdatenbank aufgebaut.
Containerisiert mit Docker. Optionaler Dockercontainer um einen gezielten VPN-Tunnel zum OpenAI-compatiblen API-Server aufzubauen.

## Quickstart

```bash
# Copy secrets template
cp secrets.env.example secrets.env

# Fill out secrets.env!

# Start services
docker compose up -d

# Visit http://localhost:8000
```

## Configuration

Siehe [`secrets.env.example`](secrets.env.example) für Einstellungen durch Environment Variablen.
Siehe [`knowledge`](knowledge) für Markdown-Dateien die Wissen und Verhalten des Chatbots steuern.

## License

Dieses Projekt benutzt ein Dual-Lizenz-Modell, um Software- und Textinhalte mit entsprechend angemessenen Lizenzen zu versehen.

| Component                                                        | License          | File                                   |
|------------------------------------------------------------------|------------------|----------------------------------------|
| Source code (`app.py`, `Dockerfile`, `docker-compose.yml`, etc.) | **EUPL 1.2**     | [LICENSE](LICENSE)                     |
| Knowledge content (`knowledge/*.md`)                             | **CC BY-SA 4.0** | [knowledge/LICENSE](knowledge/LICENSE) |

*Both licenses are open source (OSI-approved) and permit commercial use with attribution. The EUPL closes the SaaS loophole. Modified versions hosted as a service must remain open source.*

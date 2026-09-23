# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors
#
# Licensed under the EUPL, Version 1.2 only (the "Licence").
# See: https://eupl.eu/1.2/en/

FROM python:3.13-slim

# Copy uv binary from official distroless image (statically linked, no Python deps).
# Keep this tag in sync with the local uv that generated uv.lock (lock format).
COPY --from=ghcr.io/astral-sh/uv:0.12.15 /uv /uvx /bin/

WORKDIR /app

# Configure uv for production builds
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_NO_DEV=1 \
    UV_PROJECT_ENVIRONMENT=/opt/venv

# Copy pyproject.toml + uv.lock first (besser für Docker layer caching)
# uv.lock is checked into git. It pins all transitive deps and hashes,
# ensuring identical docker builds across machines (uv sync --locked).
# Regenerate it with `uv lock`, whenever packages should be updated.
COPY pyproject.toml uv.lock ./

# Install only third-party dependencies.
# --no-install-project: the repo is not a package (server/ runs as a script)
RUN uv sync --locked --no-dev --no-install-project

# Add venv to PATH
ENV PATH="/opt/venv/bin:$PATH"

# Knowledge base + server code. Baked in so the image also works
# standalone (`docker run`, future registry image). In the FIDAA-DEMO
# deployment the working tree is mounted over /app (`./fidaa:/app:ro`),
# so knowledge updates need only a pull + restart — no rebuild.
COPY knowledge ./knowledge
COPY server ./server
COPY LICENSE ./LICENSE

# streamable-http on 8002 (compose default). Local agents use stdio instead:
#   docker compose run --rm fidaa python server/main.py
EXPOSE 8002

CMD ["python", "server/main.py", "--transport", "streamable-http"]

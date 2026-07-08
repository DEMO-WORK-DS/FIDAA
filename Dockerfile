# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors
#
# Licensed under the EUPL, Version 1.2 only (the "Licence").
# See: https://eupl.eu/1.2/en/

FROM python:3.13-slim

# Copy uv binary from official distroless image (statically linked, no Python deps)
COPY --from=ghcr.io/astral-sh/uv:0.11.21 /uv /uvx /bin/

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
# --no-install-project: skip installing the project itself as a package
# --no-editable: bake wheels instead of symlinks (cleaner for containers)
RUN uv sync --locked --no-dev --no-install-project --no-editable

# Add venv to PATH so chainlit and other CLI tools are available
ENV PATH="/opt/venv/bin:$PATH"

CMD ["chainlit", "run", "app.py", "--host", "0.0.0.0", "--port", "8000"]

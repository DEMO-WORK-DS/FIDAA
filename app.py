# SPDX-License-Identifier: EUPL-1.2-only
# Copyright (C) 2026 FIDAA contributors
# SPDX-FileCopyrightText: 2026 FIDAA contributors
#
# Licensed under the EUPL, Version 1.2 only (the "Licence");
# You may not use this work except in compliance with the Licence.
# You may obtain a copy of the Licence at:
#   https://eupl.eu/1.2/en/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the Licence is distributed on an "AS IS" basis,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

"""Chainlit RAG Agent Client (OpenAI-native streaming + LangChain RAG).

Run:        docker compose up
Create user: docker compose exec app python -c "import app; app.add_user('email', 'password')"
Delete user: docker compose exec app python -c "import app; app.delete_user('email')"
"""

# pyright: reportUnknownMemberType=false
# Chainlit Funktionen returnen immer Callable ohne Parameter anstatt z.B. Callable[[str, str]].
# Das regt pyright auf. Darum ignorieren.

import json
import logging
import os
import re
import secrets
import time
from datetime import datetime
from pathlib import Path

import bcrypt
import chainlit as cl
import chevron
import httpx
import peewee as pw
import playhouse.db_url as ph_url  # pyright: ignore[reportMissingTypeStubs]  # bundled with peewee, no stubs
from email_validator import EmailNotValidError, validate_email
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter
from openai import AsyncClient, AsyncOpenAI, DefaultAsyncHttpxClient, DefaultHttpxClient

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration — hard fail on secrets, SQLite fallback for DATABASE_URL
# ---------------------------------------------------------------------------
LLM_URL = os.environ["LLM_URL"]
LLM_KEY = os.environ["LLM_KEY"]
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "Qwen-3.6")
ASSISTANT_NAME = os.getenv("ASSISTANT_NAME", "Assistant")

if not os.environ.get("CHAINLIT_AUTH_SECRET"):
    os.environ["CHAINLIT_AUTH_SECRET"] = secrets.token_urlsafe(32)

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///users.db")
DOCUMENTS_PATH = os.getenv("DOCUMENTS_PATH")
PROXY_URL = os.getenv("PROXY_URL")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "Qwen3-Embedding-4B")

# Instructions for instruction-aware embeddings
USE_EMBED_INSTRUCTIONS = os.getenv("USE_EMBED_INSTRUCTIONS", "true").lower() in ("true", "1", "yes")
DEFAULT_TASK_INSTRUCTION = os.getenv("DEFAULT_TASK_INSTRUCTION", "Given a web search query, retrieve relevant passages that answer the query")
MAX_AGENT_STEPS = 7

MAX_LOGIN_ATTEMPTS = 5
LOGIN_LOCKOUT_SECONDS = 300  # 5 minutes
MAX_PASSWORD_LENGTH = int(os.getenv("MAX_PASSWORD_LENGTH", "128"))
MAX_MESSAGE_LENGTH = int(os.getenv("MAX_MESSAGE_LENGTH", "4000"))

# Instrument OpenAI
# https://docs.chainlit.io/integrations/openai
# Macht effektiv fast alles, was die App im Frontend ausmacht, bei OpenAI backend.
cl.instrument_openai()


def _make_openai_client(timeout: float = 120.0):
    """Erstelle AsyncOpenAI client. Optional mit VPN-Proxy Verbindung, wenn PROXY_URL (env variable) gesetzt ist."""
    if PROXY_URL:
        # Siehe https://github.com/openai/openai-python/blob/main/README.md#configuring-the-http-client
        # Verweist auf das Überladen von DefaultHttpxClient, um Proxies nach https://www.python-httpx.org/advanced/proxies/ einzurichten.
        return AsyncOpenAI(
            http_client=DefaultAsyncHttpxClient(proxy=PROXY_URL),
            base_url=LLM_URL,
            api_key=LLM_KEY,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )
    else:
        return AsyncOpenAI(
            base_url=LLM_URL,
            api_key=LLM_KEY,
            timeout=httpx.Timeout(timeout, connect=10.0),
        )


# ------------------------------------------------
# Peewee Database Management
# ------------------------------------------------
# Playhouse wechselt je nach URL den Datenbank-Typ.
#  Default: Postgres via DATABASE_URL env variable
#  Fallback: Sqlite users.db file
#
# 2 Tabellen:
#   User (email + bcrypt hash passwort)
#   LoginAttempt (email + failed attempts + last attempt)

_db: pw.Database = ph_url.connect(DATABASE_URL)


class User(pw.Model):
    email: pw.CharField = pw.CharField(primary_key=True)
    password: pw.CharField = pw.CharField()

    class Meta:
        database = _db


class LoginAttempt(pw.Model):
    email = pw.CharField(primary_key=True)
    attempts = pw.IntegerField(default=0)
    last_attempt = pw.FloatField(default=0)

    class Meta:
        database = _db


def add_user(email: str, password: str):
    """Füge Nutzer zur Datenbank (z.B. Postgres) hinzu.
    on_conflict() überschreibt User, wenn bereits verfügbar. Kann also für Passwort-reset benutzt werden.

    bcrypt generiert einen eigenen Salt pro User. Der Salt wird gemeinsam mit dem Hash gespeichert.
    Scheint so industriestandard zu sein, um rainbowtable-attacks zu verhindern.

    Aufruf extern:
        docker compose exec app python -c "import app; app.add_user('david@email.de', 'password')"
    """
    try:
        vmail = validate_email(email, check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError(str(exc)) from exc
    hashed = bcrypt.hashpw(
        password[:MAX_PASSWORD_LENGTH].encode(), bcrypt.gensalt()
    ).decode()
    User.insert(email=email, password=hashed).on_conflict(
        conflict_target=[User.email],
        update={User.password: hashed},
    ).execute()


def delete_user(email: str):
    """
    Aufruf extern:
        docker compose exec app python -c "import app; app.delete_user('email')"
    """
    User.delete().where(User.email == email).execute()
    LoginAttempt.delete().where(LoginAttempt.email == email).execute()


def _check_rate_limit(email: str) -> tuple[bool, int]:
    """Returns (allowed, seconds_remaining)."""
    attempt = LoginAttempt.get_or_none(LoginAttempt.email == email)
    if not attempt:
        return True, 0
    if attempt.attempts >= MAX_LOGIN_ATTEMPTS:
        elapsed = time.time() - attempt.last_attempt
        if elapsed < LOGIN_LOCKOUT_SECONDS:
            return False, int(LOGIN_LOCKOUT_SECONDS - elapsed)
        attempt.delete_instance()
    return True, 0


def _record_failed_attempt(email: str):
    now = time.time()
    LoginAttempt.insert(email=email, attempts=1, last_attempt=now).on_conflict(
        conflict_target=[LoginAttempt.email],
        update={
            LoginAttempt.attempts: LoginAttempt.attempts + 1,
            LoginAttempt.last_attempt: now,
        },
    ).execute()


def _record_successful_attempt(email: str):
    LoginAttempt.delete().where(LoginAttempt.email == email).execute()


def login(email: str, password: str) -> tuple[bool, str]:
    """Verify credentials with rate limiting. Returns (success, error_message)."""
    if len(password) > MAX_PASSWORD_LENGTH:
        return False, "Invalid credentials"

    allowed, seconds_remaining = _check_rate_limit(email)
    if not allowed:
        return False, f"Account locked. Try again in {seconds_remaining} seconds."

    user = User.get_or_none(User.email == email)
    if not user or not bcrypt.checkpw(password.encode(), user.password.encode()):
        _record_failed_attempt(email)
        return False, "Invalid credentials"

    _record_successful_attempt(email)
    return True, ""


# ---------------------------------------------------------------------------
# OpenAI-compatible tool definitions
# ---------------------------------------------------------------------------
_TOOL_SEARCH_CONTEXT = {
    "type": "function",
    "function": {
        "name": "search_context",
        "description": "Durchsuche die interne Wissensdatenbank nach relevantem und geprüft richtigem Kontext.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Suchanfrage für die interne Wissensdatenbank.",
                }
            },
            "required": ["query"],
        },
    },
}

_TOOL_SEARCH_DOCUMENTS = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "Durchsuche das Dokumentenarchiv nach relevantem zusätzlichen Kontext.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Suchanfrage an das Dokumentenarchiv.",
                }
            },
            "required": ["query"],
        },
    },
}

# Populated at startup; read-only after that
_startup_error: str | None = None
_available_models: list[str] = []
_tools: list[dict] = [_TOOL_SEARCH_CONTEXT, _TOOL_SEARCH_DOCUMENTS]

# OpenAIEmbeddings instance — populated at first call


# OpenAIEmbeddings instance — populated at first call
_embeddings: OpenAIEmbeddings | None = None


# ---------------------------------------------------------------------------
# Instruction-aware embeddings for retrieval quality
# ---------------------------------------------------------------------------
class InstructionAwareEmbeddings(OpenAIEmbeddings):
    """OpenAIEmbeddings that prepends an instruction prefix to queries.

    Uses the client's instruction-aware embedding endpoint when enabled.
    Documents are embedded without the instruction prefix.
    """

    def embed_query(self, text: str, **kwargs) -> list[float]:
        """Embed a single query - adds instruction prefix if enabled."""
        if USE_EMBED_INSTRUCTIONS:
            text = f"Instruct: {DEFAULT_TASK_INSTRUCTION}\nQuery: {text}"
        return super().embed_query(text, **kwargs)

    def embed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Embed multiple documents - no instruction prefix for documents."""
        return super().embed_documents(texts, **kwargs)

    async def aembed_query(self, text: str, **kwargs) -> list[float]:
        """Async version of embed_query - adds instruction prefix if enabled."""
        if USE_EMBED_INSTRUCTIONS:
            text = f"Instruct: {DEFAULT_TASK_INSTRUCTION}\nQuery: {text}"
        return await super().aembed_query(text, **kwargs)

    async def aembed_documents(self, texts: list[str], **kwargs) -> list[list[float]]:
        """Async version of embed_documents - no instruction prefix for documents."""
        return await super().aembed_documents(texts, **kwargs)


def _get_embeddings():
    """Shared OpenAI-compatible embeddings instance — singleton.

    Uses OpenAIEmbeddings from langchain_openai. The singleton is lost on
    file-watcher reload, but the cached retrievers already hold their reference.
    Instruction-aware embeddings are enabled via USE_EMBED_INSTRUCTIONS.
    """
    global _embeddings
    if _embeddings is None:
        kwargs = dict(
            base_url=LLM_URL,
            api_key=LLM_KEY,
            model=EMBEDDING_MODEL,
            check_embedding_ctx_length=False,
            tiktoken_enabled=False,
        )
        if PROXY_URL:
            kwargs["http_client"] = DefaultHttpxClient(proxy=PROXY_URL)
            kwargs["http_async_client"] = DefaultAsyncHttpxClient(proxy=PROXY_URL)

        # Use InstructionAwareEmbeddings for instruction-aware embeddings
        _embeddings = InstructionAwareEmbeddings(**kwargs)
    return _embeddings


def _build_vectorstore(docs, collection_name):
    """Index docs into PGVector (production) or Chroma (SQLite fallback)."""
    embeddings = _get_embeddings()

    if isinstance(_db, pw.PostgresqlDatabase):
        try:
            from langchain_postgres.vectorstores import PGVector
        except ImportError:
            from langchain_community.vectorstores import PGVector

        vectorstore = PGVector(
            embeddings=embeddings,
            connection=DATABASE_URL,
            collection_name=collection_name,
            pre_delete_collection=True,  # Drop old 384-dim collection, re-index
        )
        # Instructions: Rebuild with 2560-dim vectors
        # pre_delete_collection=True drops the old 384-dim collection and creates new 2560-dim collection
        if not vectorstore.similarity_search("test", k=1):
            logger.info(
                "Embedding documents into PGVector (collection=%r)...", collection_name
            )
            vectorstore.add_documents(docs)
    else:
        from langchain_chroma import Chroma

        vectorstore = Chroma.from_documents(documents=docs, embedding=embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


@cl.cache
def build_rag():
    """Load knowledge/kontext.md, split and index. Returns (system_prompt, retriever)."""
    system_prompt = Path("knowledge/systemprompt.md").read_text(encoding="utf-8")
    context_text = Path("knowledge/kontext.md").read_text(encoding="utf-8")

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2")]
    )
    docs = splitter.split_text(context_text)
    return (system_prompt, _build_vectorstore(docs, collection_name="rag_context"))


@cl.cache
def build_document_search():
    """Load .md files from DOCUMENTS_PATH and index. Returns retriever or None."""
    if not DOCUMENTS_PATH:
        return None

    doc_folder = Path(DOCUMENTS_PATH)
    if not doc_folder.exists():
        logger.warning(
            "DOCUMENTS_PATH %r does not exist — document search disabled.",
            DOCUMENTS_PATH,
        )
        return None

    md_files = sorted(doc_folder.rglob("*.md"))
    if not md_files:
        logger.warning(
            "No .md files found in %r — document search disabled.", DOCUMENTS_PATH
        )
        return None

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "Header 1"), ("##", "Header 2"), ("###", "Header 3")]
    )
    docs = []
    for f in md_files:
        chunks = splitter.split_text(f.read_text(encoding="utf-8"))
        for chunk in chunks:
            chunk.metadata["source"] = str(f.relative_to(doc_folder))
        docs.extend(chunks)

    logger.info(
        "Indexing %d chunks from %d files in %r.",
        len(docs),
        len(md_files),
        DOCUMENTS_PATH,
    )
    return _build_vectorstore(docs, collection_name="rag_documents")


def strip_reasoning_tags(text: str) -> str:
    """Remove <thinking>...</thinking> tags from displayed text."""
    return re.sub(r"<thinking>.*?</thinking>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Model list
# ---------------------------------------------------------------------------
async def _fetch_models():
    """Fetch available models from the API server."""
    try:
        client = _make_openai_client(timeout=10.0)
        models = await client.models.list()
        return [m.id for m in models.data]
    except Exception:
        return [DEFAULT_MODEL]


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------
@cl.author_rename
def rename(orig_author: str):
    """Rename the assistant in the UI."""
    if orig_author in ["Assistant", "on_message"]:
        return ASSISTANT_NAME
    return orig_author


@cl.on_stop
async def on_stop():
    """Handle user clicking the stop button."""
    cl.user_session.set("stop_requested", True)


# ---------------------------------------------------------------------------
# User seeding
# ---------------------------------------------------------------------------
def _seed_users():
    """Populate users from SEED_USERS env var. Format: email:password,email:password"""
    raw = os.getenv("SEED_USERS", "").strip()
    if not raw:
        return
    count = 0
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        email, _, password = pair.partition(":")
        email, password = email.strip(), password.strip()
        if not email or not password:
            logger.warning("Unmatched user/password pair. Not seeded.")
            continue
        try:
            add_user(email, password)
            count += 1
            logger.info("Seeded user %s", email)
        except Exception:
            logger.warning("Failed to seed user: %s", email)
    if count:
        logger.info("Seeded %d users from SEED_USERS", count)


# ---------------------------------------------------------------------------
# Session initialisation — shared by on_chat_start and on_chat_resume
# ---------------------------------------------------------------------------
async def _init_session():
    """Set up model, client, api_messages, Modes picker and transcript command."""
    model = (
        DEFAULT_MODEL
        if DEFAULT_MODEL in _available_models
        else (_available_models[0] if _available_models else DEFAULT_MODEL)
    )
    cl.user_session.set("model", model)
    cl.user_session.set("client", _make_openai_client())
    cl.user_session.set("stop_requested", False)
    cl.user_session.set("api_messages", [{"role": "system", "content": build_rag()[0]}])

    await cl.context.emitter.set_modes(
        [
            cl.Mode(
                id="model",
                name="Modell",
                options=[
                    cl.ModeOption(id=m, name=m, default=(m == model))
                    for m in (_available_models or [DEFAULT_MODEL])
                ],
            )
        ]
    )
    await cl.context.emitter.set_commands(
        [
            {
                "id": "transcript",
                "icon": "file-text",
                "description": "Konversation als HTML exportieren",
                "button": True,
                "persistent": True,
            }
        ]
    )


# ---------------------------------------------------------------------------
# App startup
# ---------------------------------------------------------------------------
@cl.on_app_startup
async def startup():
    """Initialize database, RAG, document search, and model list at server start.

    On failure, sets _startup_error so on_chat_start can surface it to users.
    The GUI still starts so users get a visible error instead of a dead server.
    """
    global _startup_error, _tools, _available_models

    if not isinstance(_db, pw.PostgresqlDatabase):
        logger.warning(
            "Not using PostgreSQL — SQLite/Chroma fallback active. "
            "Not recommended for production."
        )

    try:
        _db.create_tables([User, LoginAttempt], safe=True)
    except Exception:
        logger.exception("STARTUP FAILED: database initialization error")
        _startup_error = "Datenbankfehler beim Start. Bitte Administrator kontaktieren."
        return

    _seed_users()

    try:
        build_rag()
    except Exception:
        logger.exception("STARTUP FAILED: RAG initialization error")
        _startup_error = "Wissensdatenbank konnte nicht geladen werden. Bitte Administrator kontaktieren."
        return

    try:
        build_document_search()
    except Exception:
        logger.exception(
            "STARTUP WARNING: document search initialization failed — disabled"
        )

    _tools = [_TOOL_SEARCH_CONTEXT]
    if build_document_search() is not None:
        _tools.append(_TOOL_SEARCH_DOCUMENTS)

    _available_models = await _fetch_models()
    if not _available_models:
        logger.error("STARTUP FAILED: no models available from LLM API")
        _startup_error = (
            "Keine Sprachmodelle verfügbar. Bitte LLM-API-Verbindung prüfen."
        )


# ---------------------------------------------------------------------------
# Step functions
# ---------------------------------------------------------------------------
@cl.step(type="llm", name="💭 Denkprozess…")
async def _thinking_step(message_content):
    """Stream reasoning tokens and accumulate text/tool calls from the LLM.

    cl.instrument_openai() records the API call as a nested step automatically,
    so only reasoning streaming and tool call accumulation are done manually here.
    Returns (text_acc, tool_calls_acc).
    """
    current_step = cl.context.current_step
    current_step.input = message_content

    model = cl.user_session.get("model")
    client = cl.user_session.get("client")
    api_messages = cl.user_session.get("api_messages")

    stream = await client.chat.completions.create(
        model=model,
        messages=api_messages,
        tools=_tools,
        stream=True,
    )

    text_acc = ""
    tool_calls_acc = []
    current_tool_call = None

    async for chunk in stream:
        if cl.user_session.get("stop_requested"):
            break

        delta = chunk.choices[0].delta

        if getattr(delta, "reasoning_content", None):
            await current_step.stream_token(delta.reasoning_content)

        if delta.content:
            text_acc += delta.content

        tc_list = getattr(delta, "tool_calls", None)
        if tc_list:
            for tc in tc_list:
                if tc.id:
                    current_tool_call = {
                        "id": tc.id,
                        "name": tc.function.name or "",
                        "args": tc.function.arguments or "",
                    }
                    tool_calls_acc.append(current_tool_call)
                elif current_tool_call is not None and tc.function.arguments:
                    current_tool_call["args"] += tc.function.arguments

    return text_acc, tool_calls_acc


async def _run_retrieval(tc, retriever):
    """Shared retrieval logic for both tool step functions."""
    current_step = cl.context.current_step

    raw_args = tc["args"]
    try:
        parsed_args = json.loads(raw_args)
    except Exception:
        parsed_args = {"query": raw_args}
    query = parsed_args.get("query", "")

    current_step.input = query

    try:
        docs = await cl.make_async(retriever.invoke)(query)
        result = "\n\n---\n\n".join(d.page_content for d in docs)
    except Exception:
        result = "Fehler beim Abrufen des Kontexts."
        docs = []

    sources = []
    for i, doc in enumerate(docs, 1):
        path_parts = [
            doc.metadata.get(h, "") for h in ("Header 1", "Header 2", "Header 3")
        ]
        heading = " > ".join(filter(None, path_parts))
        sources.append(f"#### {i}. `{heading}`\n```markdown\n{doc.page_content}\n```")

    current_step.output = (
        f"**Suchanfrage:** `{query}`\n### 🔍 Gefundene Quellen\n" + "\n".join(sources)
    )
    return result


@cl.step(type="tool", name="🔍 Kontextsuche")
async def _context_step(tc):
    """Search the internal knowledge base."""
    return await _run_retrieval(tc, build_rag()[1])


@cl.step(type="tool", name="📄 Dokumentensuche")
async def _doc_step(tc):
    """Search the external documents folder."""
    retriever = build_document_search()
    if retriever is None:
        return "Dokumentensuche ist nicht verfügbar."
    return await _run_retrieval(tc, retriever)


# ---------------------------------------------------------------------------
# Promptvorschläge ("Starters")
# ---------------------------------------------------------------------------
@cl.set_starters
async def set_starters(user: cl.User | None = None):
    """Parse knowledge/prompts.md into cl.Starter list."""
    # ignoring user-parameter, cause everyone get's the same starter
    content = Path("knowledge/prompts.md").read_text(encoding="utf-8")
    splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("##", "title")])
    docs = splitter.split_text(content)
    return [
        cl.Starter(label=doc.metadata["title"], message=doc.page_content.strip())
        for doc in docs
        if "title" in doc.metadata
    ]


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------
@cl.password_auth_callback
async def auth_callback(username: str, password: str) -> cl.User | None:
    success, error_message = login(username, password)
    if success:
        return cl.User(identifier=username, metadata={"email": username})
    logger.warning("Auth failed for %s: %s", username, error_message)
    return None


# ---------------------------------------------------------------------------
# HTML transcript export
# ---------------------------------------------------------------------------
_STEP_STYLES = {
    "user_message": ("👤 User", "#2563eb"),
    "assistant_message": ("🤖 FIADA", "#059669"),
    "llm": ("💭 Thinking", "#7c3aed"),
    "tool": ("🔧 Recherche", "#d97706"),
    "run": ("▶ Run", "#6b7280"),
}


def _build_export_html(thread_id: str) -> str:
    query = """
    SELECT s.type, s.name, s.input, s.output, s."createdAt",
           f.value AS fb_value, f.comment AS fb_comment
    FROM "Step" s
    LEFT JOIN "Feedback" f ON f."stepId" = s.id
    WHERE s."threadId" = %s
    ORDER BY s."startTime"
    """
    rows = _db.execute_sql(query, (thread_id,)).fetchall()

    steps = []
    for row in rows:
        step_type, step_name, inp, outp, created_at, fb_value, fb_comment = row
        label, color = _STEP_STYLES.get(step_type, (step_type or "?", "#6b7280"))
        has_feedback = fb_value is not None
        positive = has_feedback and fb_value > 0

        steps.append(
            {
                "display_name": step_name or label,
                "color": color,
                "content": outp or inp or "",
                "time": created_at.strftime("%H:%M:%S") if created_at else "",
                "has_feedback": has_feedback,
                "feedback_class": "up" if positive else "down",
                "feedback_icon": "👍" if positive else "👎",
                "feedback_comment": fb_comment or "",
            }
        )

    with open("templates/export.html", encoding="utf-8") as f:
        return chevron.render(
            f,
            {
                "exported_at": datetime.now().strftime("%d.%m.%Y um %H:%M"),
                "step_count": len(rows),
                "thread_id": thread_id,
                "steps": steps,
            },
        )


# ---------------------------------------------------------------------------
# Chat lifecycle
# ---------------------------------------------------------------------------
@cl.on_chat_start
async def on_chat_start():
    """Set up per-user session state. Startup already built the RAG and model list."""
    if _startup_error:
        await cl.ErrorMessage(content=_startup_error).send()
        return
    await _init_session()


@cl.on_chat_resume
async def on_chat_resume(thread: cl.types.ThreadDict):
    """Restore session state when opening an existing chat — no welcome message."""
    if _startup_error:
        await cl.ErrorMessage(content=_startup_error).send()
        return
    await _init_session()
    api_messages = cl.user_session.get("api_messages")
    for step in thread["steps"]:
        if step["type"] == "user_message":
            api_messages.append({"role": "user", "content": step["output"]})
        elif step["type"] == "assistant_message":
            api_messages.append({"role": "assistant", "content": step["output"]})
    cl.user_session.set("api_messages", api_messages)


@cl.on_chat_end
async def on_chat_end():
    """Clean up session data when user disconnects."""
    cl.user_session.set("api_messages", [])


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------
@cl.on_message
async def on_message(message: cl.Message):
    if message.command == "transcript":
        thread_id = cl.context.session.thread_id
        html_content = await cl.make_async(_build_export_html)(thread_id)
        await cl.Message(
            content="📄 Konversation exportiert:",
            elements=[
                cl.File(
                    name=f"export_{thread_id[:8]}.html",
                    content=html_content.encode("utf-8"),
                    mime="text/html",
                )
            ],
        ).send()
        return

    if len(message.content) > MAX_MESSAGE_LENGTH:
        await cl.Message(content="Nachricht zu lang.").send()
        return

    try:
        # Sync model from Modes picker (user may change it per message)
        model = (
            (message.modes or {}).get("model")
            or cl.user_session.get("model")
            or DEFAULT_MODEL
        )
        cl.user_session.set("model", model)

        client = cl.user_session.get("client")
        if client is None:
            client = _make_openai_client()
            cl.user_session.set("client", client)

        api_messages = cl.user_session.get("api_messages")
        if api_messages is None:
            api_messages = [{"role": "system", "content": build_rag()[0]}]
            cl.user_session.set("api_messages", api_messages)

        api_messages.append({"role": "user", "content": message.content})

        for _ in range(MAX_AGENT_STEPS):
            text_acc, tool_calls_acc = await _thinking_step(message.content)

            if tool_calls_acc:
                api_messages.append(
                    {
                        "role": "assistant",
                        "content": text_acc,
                        "tool_calls": [
                            {
                                "id": tc["id"],
                                "type": "function",
                                "function": {
                                    "name": tc["name"],
                                    "arguments": tc["args"],
                                },
                            }
                            for tc in tool_calls_acc
                        ],
                    }
                )
                for tc in tool_calls_acc:
                    if tc["name"] == "search_context":
                        result = await _context_step(tc)
                    elif tc["name"] == "search_documents":
                        result = await _doc_step(tc)
                    else:
                        result = f"Unbekanntes Tool: {tc['name']}"
                    api_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": result,
                        }
                    )
                continue

            final_text = strip_reasoning_tags(text_acc)
            api_messages.append({"role": "assistant", "content": text_acc})
            await cl.Message(content=final_text).send()
            return

        await cl.ErrorMessage(content="Fehler: Maximale Iterationen erreicht.").send()

    except Exception:
        logger.exception("Error processing message")
        await cl.ErrorMessage(
            content="Ein Fehler ist aufgetreten. Bitte versuche es erneut."
        ).send()
